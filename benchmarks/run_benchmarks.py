#!/usr/bin/env python3
"""Benchmark suite for Machine Boss dynamic programming implementations.

Times Forward and Viterbi algorithms across backends (C++ interpreter,
C++ compiled codegen, JAX CPU/GPU, JavaScript CPU, WebGPU), machine sizes,
and sequence lengths.
Outputs JSON results keyed by hostname.

Usage:
    python run_benchmarks.py [--backends cpp_interp,cpp_compiled,...] [--dry-run] [--reps N]
    python run_benchmarks.py --tables-only
"""

import argparse
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Subprocess resource measurement (for peak memory)
# ---------------------------------------------------------------------------

_HAS_WAIT4 = hasattr(os, "wait4")  # Unix only (no Windows)


def _ru_maxrss_bytes(ru_maxrss):
    """Normalize ru_maxrss from a struct_rusage to bytes.

    Darwin reports ru_maxrss in bytes; Linux (and most other Unices)
    report it in kilobytes.
    """
    if platform.system() == "Darwin":
        return int(ru_maxrss)
    return int(ru_maxrss) * 1024


def _run_measured(cmd, stdin_data=None, timeout=None, cwd=None, env=None):
    """Run a subprocess and capture stdout/stderr + child peak RSS.

    Returns (returncode, stdout, stderr, peak_rss_bytes). Uses os.wait4 to
    get the child's struct_rusage (ru_maxrss) — the max resident set size
    the child reached during its lifetime, normalized to bytes.

    On platforms without os.wait4 (Windows), falls back to subprocess.run
    and reports peak_rss_bytes = 0.

    Raises subprocess.TimeoutExpired on timeout (child is killed and reaped
    so the measurement stays clean).
    """
    if not _HAS_WAIT4:
        result = subprocess.run(
            cmd, input=stdin_data, capture_output=True, text=True,
            timeout=timeout, cwd=cwd, env=env)
        return (result.returncode, result.stdout, result.stderr, 0)

    with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
        stdin_arg = subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL
        proc = subprocess.Popen(
            cmd, stdin=stdin_arg, stdout=out_f, stderr=err_f,
            cwd=cwd, env=env)
        if stdin_data is not None:
            data = stdin_data.encode() if isinstance(stdin_data, str) else stdin_data
            try:
                proc.stdin.write(data)
            except BrokenPipeError:
                pass
            proc.stdin.close()

        deadline = time.perf_counter() + timeout if timeout is not None else None
        rusage = None
        status = 0
        while True:
            try:
                pid, status, rusage = os.wait4(proc.pid, os.WNOHANG)
            except ChildProcessError:
                return (1, "", "", 0)
            if pid != 0:
                break
            if deadline is not None and time.perf_counter() > deadline:
                proc.kill()
                try:
                    pid, status, rusage = os.wait4(proc.pid, 0)
                except ChildProcessError:
                    pass
                out_f.seek(0); err_f.seek(0)
                stdout = out_f.read().decode("utf-8", errors="replace")
                stderr = err_f.read().decode("utf-8", errors="replace")
                raise subprocess.TimeoutExpired(
                    cmd, timeout, output=stdout, stderr=stderr)
            time.sleep(0.01)

        out_f.seek(0); err_f.seek(0)
        stdout = out_f.read().decode("utf-8", errors="replace")
        stderr = err_f.read().decode("utf-8", errors="replace")
        if hasattr(os, "waitstatus_to_exitcode"):
            returncode = os.waitstatus_to_exitcode(status)
        else:
            returncode = (status >> 8) if status >= 0 else -1
        proc.returncode = returncode
        peak_bytes = _ru_maxrss_bytes(rusage.ru_maxrss) if rusage else 0
        return (returncode, stdout, stderr, peak_bytes)

# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def _cpu_model():
    """Best-effort CPU model string."""
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
            ).strip()
            return out
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _collect_machine_stats():
    """Collect platform/hardware stats for the report."""
    stats = {}
    stats["uname"] = platform.uname()._asdict()
    stats["platform"] = platform.platform()
    stats["python_version"] = platform.python_version()

    if platform.system() == "Darwin":
        keys = [
            "hw.cpufrequency", "hw.cpufrequency_max", "hw.ncpu",
            "hw.physicalcpu", "hw.logicalcpu", "hw.memsize",
            "machdep.cpu.brand_string",
        ]
        for key in keys:
            try:
                val = subprocess.check_output(
                    ["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL,
                ).strip()
                stats[key] = val
            except Exception:
                pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        stats["cpu_model"] = line.split(":", 1)[1].strip()
                        break
            stats["cpu_count"] = str(os.cpu_count())
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        stats["mem_total"] = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

    try:
        uname_full = subprocess.check_output(["uname", "-a"], text=True).strip()
        stats["uname_string"] = uname_full
    except Exception:
        pass

    return stats


def _gpu_model():
    """Return GPU model if JAX sees one, else None."""
    try:
        import jax
        devs = jax.devices("gpu")
        if devs:
            return str(devs[0])
    except Exception:
        pass
    return None


def _has_webgpu():
    """Check if Node.js with WebGPU support is available."""
    try:
        result = subprocess.run(
            ["node", "-e", "navigator.gpu ? process.exit(0) : process.exit(1)"],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def hardware_id():
    host = platform.node()
    cpu = _cpu_model()
    gpu = _gpu_model()
    parts = [host, cpu]
    if gpu:
        parts.append(gpu)
    return " / ".join(parts)


# ---------------------------------------------------------------------------
# Machine and sequence generation
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
BOSS = REPO_ROOT / "bin" / "boss"


def _generate_random_machine_json(n_states, alphabet, rng, is_generator=False,
                                   is_recognizer=False):
    """Generate a random transducer/generator/recognizer JSON dict.

    For transducers: both input and output alphabets.
    For generators: output alphabet only (no input).
    For recognizers: input alphabet only (no output).
    """
    states = []
    for s in range(n_states + 1):  # +1 for end state
        if s == n_states:
            states.append({"id": "End", "trans": []})
            continue

        trans = []
        if is_generator:
            # Output-only transitions: each symbol emitted to random dest
            for sym in alphabet:
                dest = rng.randint(0, n_states - 1)
                w = rng.uniform(0.1, 1.0)
                trans.append({"out": sym, "to": dest, "weight": round(w, 4)})
        elif is_recognizer:
            # Input-only transitions: each symbol consumed to random dest
            for sym in alphabet:
                dest = rng.randint(0, n_states - 1)
                w = rng.uniform(0.1, 1.0)
                trans.append({"in": sym, "to": dest, "weight": round(w, 4)})
        else:
            # Transducer: transitions for each (in, out) pair
            for in_sym in alphabet:
                for out_sym in alphabet:
                    dest = rng.randint(0, n_states - 1)
                    w = rng.uniform(0.1, 1.0)
                    trans.append({"in": in_sym, "out": out_sym, "to": dest,
                                  "weight": round(w, 4)})

        # Silent transition to end
        trans.append({"to": n_states, "weight": round(rng.uniform(0.01, 0.1), 4)})

        states.append({"id": f"S{s}", "trans": trans})

    return {"state": states}


def _generate_random_sequence(alphabet, length, rng):
    """Generate a random sequence of given length."""
    return "".join(rng.choice(alphabet) for _ in range(length))


# Alphabets for different machine sizes
ALPHABETS = {
    "binary": ["0", "1"],
    "dna": ["A", "C", "G", "T"],
}


# ---------------------------------------------------------------------------
# Backend wrappers
# ---------------------------------------------------------------------------

def _get_boss_path():
    """Get path to boss binary, building if needed."""
    if not BOSS.exists():
        print("Building boss binary...")
        subprocess.check_call(["make"], cwd=REPO_ROOT, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    return str(BOSS)


def _time_cpp_interp(machine_json, algorithm, input_seq=None, output_seq=None,
                     n_reps=3, timeout=60.0):
    """Time C++ interpreter (boss CLI with generic DP) for forward/viterbi.

    Returns (mean_s, std_s, n_reps_completed, peak_rss_bytes) or None on error.
    peak_rss_bytes is the max child RSS observed across probe + timed reps.
    """
    boss = _get_boss_path()

    # Write machine to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(machine_json, f)
        machine_file = f.name

    try:
        cmd = [boss, machine_file]
        if algorithm == "Forward":
            cmd.append("-L")
        elif algorithm == "Viterbi":
            cmd.append("-V")
        else:
            return None

        if input_seq:
            cmd.extend(["--input-chars", input_seq])
        if output_seq:
            cmd.extend(["--output-chars", output_seq])

        peak_bytes = 0

        # Probe call
        t0 = time.perf_counter()
        try:
            rc, _, _, probe_peak = _run_measured(cmd, timeout=timeout + 5)
        except subprocess.TimeoutExpired:
            return None
        probe_time = time.perf_counter() - t0
        if rc != 0:
            return None
        peak_bytes = max(peak_bytes, probe_peak)

        if probe_time > timeout:
            return (probe_time, 0.0, 1, peak_bytes)

        # Full timing
        times = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            try:
                rc, _, _, rep_peak = _run_measured(cmd, timeout=timeout + 5)
            except subprocess.TimeoutExpired:
                break
            elapsed = time.perf_counter() - t0
            if rc != 0:
                break
            times.append(elapsed)
            peak_bytes = max(peak_bytes, rep_peak)
            if elapsed > timeout:
                break

        if not times:
            return None
        return (float(np.mean(times)), float(np.std(times)), len(times), peak_bytes)
    except Exception:
        return None
    finally:
        os.unlink(machine_file)


def _compile_machine(machine_json, algorithm, is_generator=False):
    """Compile a machine into a standalone C++ binary using boss --cpp64.

    Returns path to compiled binary, or None on failure.
    The caller must clean up the temp directory.
    """
    boss = _get_boss_path()
    codegen_dir = tempfile.mkdtemp(prefix="boss_codegen_")

    # Write machine to temp file
    machine_file = os.path.join(codegen_dir, "machine.json")
    with open(machine_file, "w") as f:
        json.dump(machine_json, f)

    try:
        # Generate C++ code
        codegen_cmd = [boss, machine_file, "--cpp64",
                       "--inseq", "string", "--outseq", "string",
                       "--codegen", codegen_dir + "/"]
        if algorithm == "Viterbi":
            codegen_cmd.append("--compileviterbi")

        result = subprocess.run(codegen_cmd, capture_output=True, text=True,
                                timeout=30)
        if result.returncode != 0:
            shutil.rmtree(codegen_dir, ignore_errors=True)
            return None, None

        # Write a minimal test harness that times the computation
        harness = os.path.join(codegen_dir, "bench.cpp")
        with open(harness, "w") as f:
            f.write("""\
#include <fstream>
#include <iostream>
#include <chrono>
#include <string>
#include <map>
#include <cstdlib>
#include "computeForward.h"
using namespace std;
int main(int argc, char** argv) {
    if (argc < 4) { cerr << "Usage: bench inseq outseq nreps" << endl; return 1; }
    string inStr(argv[1]);
    string outStr(argv[2]);
    int nreps = atoi(argv[3]);
    map<string,double> params;
    // Warmup
    volatile double ll = computeForward(inStr, outStr, params);
    // Timed runs
    for (int i = 0; i < nreps; i++) {
        auto t0 = chrono::high_resolution_clock::now();
        ll = computeForward(inStr, outStr, params);
        auto t1 = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double>(t1 - t0).count();
        cout << elapsed << endl;
    }
    return 0;
}
""")

        # Compile: bench.cpp + all computeForward*.cpp
        import glob
        cf_files = glob.glob(os.path.join(codegen_dir, "computeForward*.cpp"))
        binary = os.path.join(codegen_dir, "bench")

        compile_cmd = ["clang++", "-std=c++11", "-O3",
                       "-I" + str(REPO_ROOT / "src"),
                       "-I" + str(REPO_ROOT / "ext"),
                       "-I" + str(REPO_ROOT / "ext" / "nlohmann_json"),
                       "-I" + codegen_dir,
                       "-o", binary, harness] + cf_files
        result = subprocess.run(compile_cmd, capture_output=True, text=True,
                                timeout=120)
        if result.returncode != 0:
            print(f" COMPILE ERROR: {result.stderr[:200]}")
            shutil.rmtree(codegen_dir, ignore_errors=True)
            return None, None

        return binary, codegen_dir
    except Exception as e:
        print(f" CODEGEN ERROR: {e}")
        shutil.rmtree(codegen_dir, ignore_errors=True)
        return None, None


def _time_cpp_compiled(machine_json, algorithm, input_seq=None, output_seq=None,
                       n_reps=3, timeout=60.0, is_generator=False):
    """Time compiled C++ code generated by boss --cpp64.

    Generates machine-specific C++ code, compiles it, and times execution.
    Returns (mean_s, std_s, n_reps_completed, peak_rss_bytes) or None on error.
    """
    binary, codegen_dir = _compile_machine(machine_json, algorithm,
                                           is_generator=is_generator)
    if binary is None:
        return None

    try:
        in_arg = input_seq or ""
        out_arg = output_seq or ""

        try:
            rc, stdout, _, peak_bytes = _run_measured(
                [binary, in_arg, out_arg, str(n_reps)],
                timeout=timeout * (n_reps + 2),
            )
        except subprocess.TimeoutExpired:
            return None
        if rc != 0:
            return None

        times = []
        for line in stdout.strip().split("\n"):
            line = line.strip()
            if line:
                times.append(float(line))

        if not times:
            return None

        return (float(np.mean(times)), float(np.std(times)), len(times), peak_bytes)
    except Exception as e:
        print(f" RUN ERROR: {e}")
        return None
    finally:
        shutil.rmtree(codegen_dir, ignore_errors=True)


# Cache for compiled binaries: (machine_key, algorithm) -> (binary, codegen_dir)
_compiled_cache = {}


def _time_cpp_compiled_cached(machine_json, algorithm, input_seq=None,
                              output_seq=None, n_reps=3, timeout=60.0,
                              is_generator=False, cache_key=None):
    """Like _time_cpp_compiled but caches the compiled binary across sequence lengths.

    cache_key should uniquely identify the (machine, algorithm) pair.
    Returns (mean_s, std_s, n_reps_completed, peak_rss_bytes) or None on error.
    """
    global _compiled_cache

    if cache_key and cache_key in _compiled_cache:
        binary, codegen_dir = _compiled_cache[cache_key]
    else:
        binary, codegen_dir = _compile_machine(machine_json, algorithm,
                                               is_generator=is_generator)
        if binary is None:
            return None
        if cache_key:
            _compiled_cache[cache_key] = (binary, codegen_dir)

    try:
        in_arg = input_seq or ""
        out_arg = output_seq or ""

        try:
            rc, stdout, _, peak_bytes = _run_measured(
                [binary, in_arg, out_arg, str(n_reps)],
                timeout=timeout * (n_reps + 2),
            )
        except subprocess.TimeoutExpired:
            return None
        if rc != 0:
            return None

        times = []
        for line in stdout.strip().split("\n"):
            line = line.strip()
            if line:
                times.append(float(line))

        if not times:
            return None

        return (float(np.mean(times)), float(np.std(times)), len(times), peak_bytes)
    except Exception as e:
        print(f" RUN ERROR: {e}")
        return None


def _cleanup_compiled_cache():
    """Remove all cached compiled binary directories."""
    global _compiled_cache
    for key, (binary, codegen_dir) in _compiled_cache.items():
        shutil.rmtree(codegen_dir, ignore_errors=True)
    _compiled_cache.clear()


def _get_jax_machine(machine_json, params=None):
    """Create a (JAXMachine, EvaluatedMachine) from JSON dict.

    Returns both so we can use EvaluatedMachine for tokenization.
    """
    from machineboss.machine import Machine
    from machineboss.eval import EvaluatedMachine
    from machineboss.jax.types import JAXMachine

    m = Machine.from_json(machine_json)
    em = EvaluatedMachine.from_machine(m, params or {})
    jm = JAXMachine.from_evaluated(em)
    return jm, em


def _tokenize_seq(seq_str, token_list):
    """Convert string sequence to token indices.

    token_list: list where index 0 = empty token, 1..N = alphabet symbols.
    """
    tok_map = {sym: i for i, sym in enumerate(token_list)}
    return np.array([tok_map[c] for c in seq_str], dtype=np.int32)


_JAX_BENCH_SHIM = r'''#!/usr/bin/env python3
"""JAX benchmark shim — runs one (machine, sequences, algorithm) config,
reports timing and (where available) GPU peak bytes to stdout as JSON.
Intended to be spawned as a subprocess so that the parent can measure
the child's peak RSS via os.wait4.
"""
import json, os, sys, time


def main():
    args = json.loads(sys.stdin.read())
    os.environ["JAX_PLATFORMS"] = "gpu" if args["use_gpu"] else "cpu"

    import numpy as np
    import jax
    import jax.numpy as jnp

    if args["use_gpu"]:
        gpu_devs = jax.devices("gpu")
        if not gpu_devs:
            print(json.dumps({"error": "no gpu device"}))
            sys.exit(1)

    from machineboss.machine import Machine
    from machineboss.eval import EvaluatedMachine
    from machineboss.jax.types import JAXMachine
    from machineboss.jax.forward import log_forward
    from machineboss.jax.viterbi import log_viterbi

    m = Machine.from_json(args["machine"])
    em = EvaluatedMachine.from_machine(m, args.get("params") or {})
    jm = JAXMachine.from_evaluated(em)

    def _tok(seq_str, token_list):
        tok_map = {sym: i for i, sym in enumerate(token_list)}
        return np.array([tok_map[c] for c in seq_str], dtype=np.int32)

    in_tokens = (jnp.array(_tok(args["input_seq"], em.input_tokens))
                 if args["input_seq"] is not None else None)
    out_tokens = (jnp.array(_tok(args["output_seq"], em.output_tokens))
                  if args["output_seq"] is not None else None)

    strategy = args["strategy"]
    kernel = args["kernel"]
    if args["algorithm"] == "Forward":
        fn = lambda: float(log_forward(jm, in_tokens, out_tokens,
                                       strategy=strategy, kernel=kernel))
    elif args["algorithm"] == "Viterbi":
        fn = lambda: float(log_viterbi(jm, in_tokens, out_tokens,
                                       strategy=strategy, kernel=kernel))
    else:
        print(json.dumps({"error": "unknown algorithm"}))
        sys.exit(1)

    timeout = float(args["timeout"])
    n_reps = int(args["n_reps"])

    def _gpu_stats():
        """Return (bytes_in_use, peak_bytes_in_use) for the first GPU device,
        or (None, None) if unsupported/unavailable.
        """
        if not args["use_gpu"]:
            return (None, None)
        try:
            stats = jax.devices("gpu")[0].memory_stats() or {}
            return (int(stats.get("bytes_in_use", 0)) if "bytes_in_use" in stats else None,
                    int(stats.get("peak_bytes_in_use", 0)) if "peak_bytes_in_use" in stats else None)
        except Exception:
            return (None, None)

    # Sample GPU allocator state BEFORE warmup so we can subtract out
    # one-time allocations from imports / machine load.
    use_before, peak_before = _gpu_stats()

    # Warmup / probe
    t0 = time.perf_counter()
    fn()
    probe = time.perf_counter() - t0

    if probe > timeout:
        times = [probe]
    else:
        times = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            fn()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            if elapsed > timeout:
                break

    use_after, peak_after = _gpu_stats()

    out = {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "n": len(times),
    }

    # GPU memory stats from XLA, where supported.
    #   gpu_peak_bytes:        peak_bytes_in_use at end of run (cumulative
    #                          max since the subprocess started, so includes
    #                          JAX imports + JIT compile + buffers).
    #   gpu_peak_delta_bytes:  peak growth during warmup + reps
    #                          (peak_after - peak_before). Isolates the
    #                          memory the timed run actually needed on top
    #                          of import/setup overhead.
    #   gpu_bytes_delta:       bytes_in_use_after - bytes_in_use_before.
    #                          Residual growth — memory retained by the
    #                          JIT cache / constants after the run ends.
    if peak_after is not None:
        gpu = {"peak_bytes": peak_after}
        if peak_before is not None:
            gpu["peak_delta_bytes"] = max(0, peak_after - peak_before)
        if use_after is not None and use_before is not None:
            gpu["bytes_delta"] = max(0, use_after - use_before)
        out["gpu"] = gpu

    print(json.dumps(out))


if __name__ == "__main__":
    main()
'''


def _time_jax(machine_json, algorithm, input_seq=None, output_seq=None,
              strategy="auto", kernel="auto", n_reps=3, timeout=60.0,
              use_gpu=False):
    """Time JAX implementation in a subprocess for memory isolation.

    Spawns a fresh Python process (via a shim script) that imports JAX,
    runs warmup + n_reps of the algorithm, and reports JSON timing. The
    parent captures peak RSS via os.wait4 (see _run_measured).

    Returns (mean_s, std_s, n_reps_completed, peak_rss_bytes, gpu_info)
    or None on error. gpu_info is None (CPU run / no stats) or a dict
    with keys {peak_bytes, peak_delta_bytes, bytes_delta}.
    """
    # Write shim script to a temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(_JAX_BENCH_SHIM)
        shim_path = f.name

    payload = {
        "machine": machine_json,
        "input_seq": input_seq,
        "output_seq": output_seq,
        "algorithm": algorithm,
        "strategy": strategy,
        "kernel": kernel,
        "n_reps": n_reps,
        "timeout": timeout,
        "use_gpu": use_gpu,
    }
    stdin_data = json.dumps(payload)

    # Budget the subprocess wall time generously: import + warmup + reps
    wall_timeout = max(timeout * (n_reps + 2) + 120, 60.0)

    try:
        try:
            rc, stdout, stderr, peak_bytes = _run_measured(
                [sys.executable, shim_path],
                stdin_data=stdin_data,
                timeout=wall_timeout,
            )
        except subprocess.TimeoutExpired:
            return None
        if rc != 0:
            if stderr:
                err_line = stderr.strip().splitlines()[-1] if stderr.strip() else ""
                print(f" JAX ERROR: {err_line[:200]}")
            return None

        try:
            data = json.loads(stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            print(f" JAX PARSE ERROR")
            return None

        if "error" in data:
            return None

        gpu_info = data.get("gpu")  # dict or None
        return (data["mean"], data["std"], data["n"], peak_bytes, gpu_info)
    finally:
        try:
            os.unlink(shim_path)
        except OSError:
            pass


def _time_js_cpu(machine_json, algorithm, input_seq=None, output_seq=None,
                 n_reps=3, timeout=60.0):
    """Time JavaScript CPU fallback via Node.js.

    Returns (mean_s, std_s, n_reps_completed, peak_rss_bytes) or None if
    Node.js unavailable.
    """
    js_dir = REPO_ROOT / "js" / "webgpu"
    if not (js_dir / "machineboss-gpu.mjs").exists():
        return None

    # Write temp machine file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(machine_json, f)
        machine_file = f.name

    # Build a Node.js benchmark script
    algo_method = "forward" if algorithm == "Forward" else "viterbi"
    in_arg = json.dumps(list(input_seq)) if input_seq else "null"
    out_arg = json.dumps(list(output_seq)) if output_seq else "null"

    script = f"""
import {{ readFileSync }} from 'fs';
import {{ MachineBoss }} from '{js_dir}/machineboss-gpu.mjs';

const machine = JSON.parse(readFileSync('{machine_file}', 'utf8'));
const mb = await MachineBoss.create(machine, {{}}, {{ backend: 'cpu' }});

const inToks = {in_arg};
const outToks = {out_arg};

// Warmup
await mb.{algo_method}(inToks, outToks);

const times = [];
for (let i = 0; i < {n_reps}; i++) {{
    const t0 = performance.now();
    await mb.{algo_method}(inToks, outToks);
    const elapsed = (performance.now() - t0) / 1000.0;
    times.push(elapsed);
    if (elapsed > {timeout}) break;
}}

const mean = times.reduce((a, b) => a + b) / times.length;
const std = Math.sqrt(times.reduce((a, b) => a + (b - mean) ** 2, 0) / times.length);
console.log(JSON.stringify({{ mean, std, n: times.length }}));

mb.destroy();
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".mjs", delete=False) as f:
        f.write(script)
        script_file = f.name

    try:
        try:
            rc, stdout, _, peak_bytes = _run_measured(
                ["node", script_file],
                timeout=timeout * (n_reps + 2),
            )
        except subprocess.TimeoutExpired:
            return None
        if rc != 0:
            return None
        data = json.loads(stdout.strip())
        return (data["mean"], data["std"], data["n"], peak_bytes)
    except Exception:
        return None
    finally:
        os.unlink(machine_file)
        os.unlink(script_file)


def _time_js_fused_plan7(hmm_path, transducer_json, params, algorithm,
                          output_seq, n_reps=3, timeout=60.0):
    """Time JavaScript fused Plan7 kernel via Node.js.

    Returns (mean_s, std_s, n_reps_completed, loglike, peak_rss_bytes) or None.
    """
    js_dir = REPO_ROOT / "js" / "webgpu"
    if not (js_dir / "cpu" / "fused-plan7.mjs").exists():
        return None

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(transducer_json, f)
        td_file = f.name

    algo_fn = "fusedPlan7Forward" if algorithm == "Forward" else "fusedPlan7Viterbi"
    out_str = json.dumps(list(output_seq)) if output_seq else "[]"
    params_str = json.dumps(params)

    script = f"""
import {{ readFileSync }} from 'fs';
import {{ parseHmmer }} from '{js_dir}/internal/hmmer-parse.mjs';
import {{ prepareMachine, tokenize }} from '{js_dir}/internal/machine-prep.mjs';
import {{ buildFusedPlan7, fusedPlan7Forward, fusedPlan7Viterbi }} from '{js_dir}/cpu/fused-plan7.mjs';

const hmm = parseHmmer(readFileSync('{hmm_path}', 'utf8'));
const td = JSON.parse(readFileSync('{td_file}', 'utf8'));
const params = {params_str};
const prepared = prepareMachine(td, params);
const fused = buildFusedPlan7(hmm, prepared);
const outSeq = tokenize({out_str}, prepared.outputAlphabet);

// Warmup
let ll = {algo_fn}(fused, outSeq);

const times = [];
for (let i = 0; i < {n_reps}; i++) {{
    const t0 = performance.now();
    ll = {algo_fn}(fused, outSeq);
    const elapsed = (performance.now() - t0) / 1000.0;
    times.push(elapsed);
    if (elapsed > {timeout}) break;
}}

const mean = times.reduce((a, b) => a + b) / times.length;
const std = Math.sqrt(times.reduce((a, b) => a + (b - mean) ** 2, 0) / times.length);
console.log(JSON.stringify({{ mean, std, n: times.length, ll }}));
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".mjs", delete=False) as f:
        f.write(script)
        script_file = f.name

    try:
        try:
            rc, stdout, _, peak_bytes = _run_measured(
                ["node", script_file],
                timeout=timeout * (n_reps + 2),
            )
        except subprocess.TimeoutExpired:
            return None
        if rc != 0:
            return None
        data = json.loads(stdout.strip())
        return (data["mean"], data["std"], data["n"], data["ll"], peak_bytes)
    except Exception:
        return None
    finally:
        os.unlink(td_file)
        os.unlink(script_file)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def time_fn(fn, n_reps, timeout=None):
    """Time fn() over n_reps calls. Returns (mean_s, std_s)."""
    times = []
    for i in range(n_reps):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)
        if timeout and elapsed > timeout:
            break
    return float(np.mean(times)), float(np.std(times))


def _unpack_timing(timing):
    """Normalize a backend _time_* return tuple.

    Accepts 3-, 4-, or 5-tuples and pads missing slots:
      (mean, std, n)                           -> (_, _, _, 0, None)
      (mean, std, n, peak_rss)                 -> (_, _, _, peak_rss, None)
      (mean, std, n, peak_rss, gpu_info_dict)  -> as-is (JAX)
    Returns (mean_s, std_s, n_completed, peak_rss_bytes, gpu_info)
    where gpu_info is None or a dict with keys
    {peak_bytes, peak_delta_bytes, bytes_delta} (only populated for
    JAX GPU runs when XLA memory_stats supports them).
    """
    if len(timing) == 3:
        mean_s, std_s, n = timing
        return (mean_s, std_s, n, 0, None)
    if len(timing) == 4:
        mean_s, std_s, n, peak = timing
        return (mean_s, std_s, n, peak, None)
    mean_s, std_s, n, peak, gpu_info = timing
    return (mean_s, std_s, n, peak, gpu_info)


def _h_bytes(n):
    """Human-readable bytes (e.g. '1.4 GB')."""
    if n is None:
        return "?"
    if n <= 0:
        return "—"
    units = [("B", 1), ("KB", 1024), ("MB", 1024**2),
             ("GB", 1024**3), ("TB", 1024**4)]
    for name, scale in reversed(units):
        if n >= scale:
            return f"{n/scale:.1f} {name}"
    return f"{n} B"


def _fmt_bytes(peak_rss_bytes, gpu_info=None):
    """Format memory usage for console output (RSS + optional GPU)."""
    out = f"RSS {_h_bytes(peak_rss_bytes)}"
    if gpu_info:
        peak = gpu_info.get("peak_bytes")
        delta = gpu_info.get("peak_delta_bytes")
        resid = gpu_info.get("bytes_delta")
        parts = []
        if delta is not None and delta > 0:
            parts.append(f"Δpeak {_h_bytes(delta)}")
        elif peak is not None and peak > 0:
            parts.append(f"peak {_h_bytes(peak)}")
        if resid is not None and resid > 0:
            parts.append(f"Δuse {_h_bytes(resid)}")
        if parts:
            out += ", GPU " + " / ".join(parts)
    return out


# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

ALGORITHMS = ["Forward", "Viterbi"]

# 1D problems: generator with output sequence only
PARAM_GRID_1D = {
    "L": [100, 500, 2000, 10000],
    "S": [4, 16, 64],
}

# 2D problems: transducer with input and output sequences
PARAM_GRID_2D = {
    "Li": [10, 50, 200],
    "Lo": [10, 50, 200],
    "S": [4, 16, 64],
}

# Backend definitions
ALL_BACKENDS_1D = [
    "cpp_interp",
    "cpp_compiled",
    "jax_1d_simple",
    "jax_1d_optimal",
    "jax_gpu_1d",
    "js_cpu",
]

ALL_BACKENDS_2D = [
    "cpp_interp",
    "cpp_compiled",
    "jax_2d_simple",
    "jax_2d_optimal",
    "jax_gpu_2d",
    "js_cpu",
]

DEFAULT_BACKENDS = [
    "cpp_interp", "cpp_compiled",
    "jax_1d_simple", "jax_1d_optimal",
    "jax_2d_simple", "jax_2d_optimal",
    "jax_gpu_1d", "jax_gpu_2d",
    "js_cpu",
]

ALPHABET = ALPHABETS["dna"]


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmarks(backends, n_reps, dry_run=False, timeout=60.0):
    """Run the full benchmark grid. Returns list of result dicts."""
    results = []
    hw = hardware_id()

    # Separate backends into 1D and 2D
    backends_1d = [b for b in backends if b in ALL_BACKENDS_1D]
    backends_2d = [b for b in backends if b in ALL_BACKENDS_2D]

    # Count total configs
    total_1d = (len(PARAM_GRID_1D["L"]) * len(PARAM_GRID_1D["S"])
                * len(ALGORITHMS) * len(backends_1d))
    total_2d = (len(PARAM_GRID_2D["Li"]) * len(PARAM_GRID_2D["Lo"])
                * len(PARAM_GRID_2D["S"]) * len(ALGORITHMS) * len(backends_2d))
    total = total_1d + total_2d
    done = 0

    rng = np.random.RandomState(42)

    # Track timeout keys: (backend, algorithm, S) -> skip larger
    timed_out = set()

    # ---- 1D benchmarks (generators) ----
    if backends_1d:
        print("\n=== 1D Benchmarks (generators) ===\n")

    for S in PARAM_GRID_1D["S"]:
        machine_json = _generate_random_machine_json(
            S, ALPHABET, rng, is_generator=True)

        for L in PARAM_GRID_1D["L"]:
            seq = _generate_random_sequence(ALPHABET, L, rng)

            for algorithm in ALGORITHMS:
                for backend in backends_1d:
                    done += 1
                    label = (f"[{done}/{total}] 1D {backend} {algorithm} "
                             f"S={S} L={L}")

                    if dry_run:
                        print(f"  DRY RUN: {label}")
                        results.append({
                            "problem": "1D", "backend": backend,
                            "algorithm": algorithm, "S": S, "L": L,
                            "Li": 0, "Lo": L,
                            "mean_seconds": 0.0, "std_seconds": 0.0,
                            "peak_rss_bytes": 0,
                            "gpu_peak_bytes": None,
                            "gpu_peak_delta_bytes": None,
                            "gpu_bytes_delta": None,
                            "n_reps": n_reps, "hardware_id": hw,
                        })
                        continue

                    skip_key = (backend, algorithm, S)
                    if skip_key in timed_out:
                        print(f"  {label} ... SKIPPED (smaller config exceeded {timeout}s)")
                        continue

                    print(f"  {label} ...", end="", flush=True)

                    timing = None
                    try:
                        if backend == "cpp_interp":
                            timing = _time_cpp_interp(
                                machine_json, algorithm,
                                output_seq=seq,
                                n_reps=n_reps, timeout=timeout)
                        elif backend == "cpp_compiled":
                            timing = _time_cpp_compiled_cached(
                                machine_json, algorithm,
                                output_seq=seq,
                                n_reps=n_reps, timeout=timeout,
                                is_generator=True,
                                cache_key=("1D", algorithm, S))
                        elif backend == "jax_1d_simple":
                            timing = _time_jax(
                                machine_json, algorithm,
                                output_seq=seq,
                                strategy="simple", kernel="dense",
                                n_reps=n_reps, timeout=timeout)
                        elif backend == "jax_1d_optimal":
                            timing = _time_jax(
                                machine_json, algorithm,
                                output_seq=seq,
                                strategy="optimal", kernel="dense",
                                n_reps=n_reps, timeout=timeout)
                        elif backend == "jax_gpu_1d":
                            timing = _time_jax(
                                machine_json, algorithm,
                                output_seq=seq,
                                strategy="auto", kernel="auto",
                                n_reps=n_reps, timeout=timeout,
                                use_gpu=True)
                        elif backend == "js_cpu":
                            timing = _time_js_cpu(
                                machine_json, algorithm,
                                output_seq=seq,
                                n_reps=n_reps, timeout=timeout)
                    except Exception as e:
                        print(f" ERROR: {e}")
                        continue

                    if timing is None:
                        print(f" SKIPPED (backend unavailable)")
                        continue

                    mean_s, std_s, n_completed, peak_bytes, gpu_info = _unpack_timing(timing)

                    if mean_s > timeout:
                        print(f" {mean_s:.2f}s (probe > {timeout}s, skipping larger)")
                        timed_out.add(skip_key)

                        results.append({
                            "problem": "1D", "backend": backend,
                            "algorithm": algorithm, "S": S, "L": L,
                            "Li": 0, "Lo": L,
                            "mean_seconds": mean_s, "std_seconds": std_s,
                            "peak_rss_bytes": peak_bytes,
                            "gpu_peak_bytes": (gpu_info or {}).get("peak_bytes"),
                            "gpu_peak_delta_bytes": (gpu_info or {}).get("peak_delta_bytes"),
                            "gpu_bytes_delta": (gpu_info or {}).get("bytes_delta"),
                            "n_reps": 1, "hardware_id": hw,
                        })
                        continue

                    print(f" {mean_s:.4f} +/- {std_s:.4f} s  "
                          f"[{_fmt_bytes(peak_bytes, gpu_info)}]")
                    results.append({
                        "problem": "1D", "backend": backend,
                        "algorithm": algorithm, "S": S, "L": L,
                        "Li": 0, "Lo": L,
                        "mean_seconds": mean_s, "std_seconds": std_s,
                        "peak_rss_bytes": peak_bytes,
                        "gpu_peak_bytes": (gpu_info or {}).get("peak_bytes"),
                        "gpu_peak_delta_bytes": (gpu_info or {}).get("peak_delta_bytes"),
                        "gpu_bytes_delta": (gpu_info or {}).get("bytes_delta"),
                        "n_reps": n_completed, "hardware_id": hw,
                    })

    # ---- 2D benchmarks (transducers) ----
    if backends_2d:
        print("\n=== 2D Benchmarks (transducers) ===\n")

    for S in PARAM_GRID_2D["S"]:
        machine_json = _generate_random_machine_json(
            S, ALPHABET, rng, is_generator=False, is_recognizer=False)

        for Li in PARAM_GRID_2D["Li"]:
            in_seq = _generate_random_sequence(ALPHABET, Li, rng)

            for Lo in PARAM_GRID_2D["Lo"]:
                out_seq = _generate_random_sequence(ALPHABET, Lo, rng)

                for algorithm in ALGORITHMS:
                    for backend in backends_2d:
                        done += 1
                        label = (f"[{done}/{total}] 2D {backend} {algorithm} "
                                 f"S={S} Li={Li} Lo={Lo}")

                        if dry_run:
                            print(f"  DRY RUN: {label}")
                            results.append({
                                "problem": "2D", "backend": backend,
                                "algorithm": algorithm, "S": S,
                                "L": 0, "Li": Li, "Lo": Lo,
                                "mean_seconds": 0.0, "std_seconds": 0.0,
                                "peak_rss_bytes": 0,
                                "gpu_peak_bytes": None,
                                "gpu_peak_delta_bytes": None,
                                "gpu_bytes_delta": None,
                                "n_reps": n_reps, "hardware_id": hw,
                            })
                            continue

                        skip_key = (backend, algorithm, S)
                        if skip_key in timed_out:
                            print(f"  {label} ... SKIPPED (smaller config exceeded {timeout}s)")
                            continue

                        print(f"  {label} ...", end="", flush=True)

                        timing = None
                        try:
                            if backend == "cpp_interp":
                                timing = _time_cpp_interp(
                                    machine_json, algorithm,
                                    input_seq=in_seq, output_seq=out_seq,
                                    n_reps=n_reps, timeout=timeout)
                            elif backend == "cpp_compiled":
                                timing = _time_cpp_compiled_cached(
                                    machine_json, algorithm,
                                    input_seq=in_seq, output_seq=out_seq,
                                    n_reps=n_reps, timeout=timeout,
                                    is_generator=False,
                                    cache_key=("2D", algorithm, S))
                            elif backend == "jax_2d_simple":
                                timing = _time_jax(
                                    machine_json, algorithm,
                                    input_seq=in_seq, output_seq=out_seq,
                                    strategy="simple", kernel="dense",
                                    n_reps=n_reps, timeout=timeout)
                            elif backend == "jax_2d_optimal":
                                timing = _time_jax(
                                    machine_json, algorithm,
                                    input_seq=in_seq, output_seq=out_seq,
                                    strategy="optimal", kernel="dense",
                                    n_reps=n_reps, timeout=timeout)
                            elif backend == "jax_gpu_2d":
                                timing = _time_jax(
                                    machine_json, algorithm,
                                    input_seq=in_seq, output_seq=out_seq,
                                    strategy="auto", kernel="auto",
                                    n_reps=n_reps, timeout=timeout,
                                    use_gpu=True)
                            elif backend == "js_cpu":
                                timing = _time_js_cpu(
                                    machine_json, algorithm,
                                    input_seq=in_seq, output_seq=out_seq,
                                    n_reps=n_reps, timeout=timeout)
                        except Exception as e:
                            print(f" ERROR: {e}")
                            continue

                        if timing is None:
                            print(f" SKIPPED (backend unavailable)")
                            continue

                        mean_s, std_s, n_completed, peak_bytes, gpu_info = _unpack_timing(timing)

                        if mean_s > timeout:
                            print(f" {mean_s:.2f}s (probe > {timeout}s, skipping larger)")
                            timed_out.add(skip_key)

                            results.append({
                                "problem": "2D", "backend": backend,
                                "algorithm": algorithm, "S": S,
                                "L": 0, "Li": Li, "Lo": Lo,
                                "mean_seconds": mean_s, "std_seconds": std_s,
                                "peak_rss_bytes": peak_bytes,
                                "gpu_peak_bytes": (gpu_info or {}).get("peak_bytes"),
                                "gpu_peak_delta_bytes": (gpu_info or {}).get("peak_delta_bytes"),
                                "gpu_bytes_delta": (gpu_info or {}).get("bytes_delta"),
                                "n_reps": 1, "hardware_id": hw,
                            })
                            continue

                        print(f" {mean_s:.4f} +/- {std_s:.4f} s  "
                              f"[{_fmt_bytes(peak_bytes, gpu_info)}]")
                        results.append({
                            "problem": "2D", "backend": backend,
                            "algorithm": algorithm, "S": S,
                            "L": 0, "Li": Li, "Lo": Lo,
                            "mean_seconds": mean_s, "std_seconds": std_s,
                            "peak_rss_bytes": peak_bytes,
                            "gpu_peak_bytes": (gpu_info or {}).get("peak_bytes"),
                            "gpu_peak_delta_bytes": (gpu_info or {}).get("peak_delta_bytes"),
                            "gpu_bytes_delta": (gpu_info or {}).get("bytes_delta"),
                            "n_reps": n_completed, "hardware_id": hw,
                        })

    _cleanup_compiled_cache()
    return results


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def save_results(results, out_dir="results"):
    """Save results, merging with any existing data for this host.

    New results replace old results for the same (problem, backend, algorithm,
    S, L, Li, Lo) key; results for backends not in the new run are preserved.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    hostname = platform.node() or "unknown"
    filepath = out_path / f"{hostname}.json"

    # Load existing results if any
    existing = []
    if filepath.exists():
        try:
            with open(filepath) as f:
                old_data = json.load(f)
            existing = old_data.get("results", [])
        except (json.JSONDecodeError, KeyError):
            pass

    # Build set of keys being replaced by the new run
    def _result_key(r):
        return (r["problem"], r["backend"], r["algorithm"],
                r.get("S", 0), r.get("L", 0), r.get("Li", 0), r.get("Lo", 0))

    new_keys = {_result_key(r) for r in results}

    # Keep old results whose keys are not in the new run
    merged = [r for r in existing if _result_key(r) not in new_keys]
    merged.extend(results)

    data = {
        "hardware_id": results[0]["hardware_id"] if results else hardware_id(),
        "hostname": hostname,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "machine_stats": _collect_machine_stats(),
        "results": merged,
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {filepath} ({len(merged)} records, "
          f"{len(results)} new/updated, {len(merged) - len(results)} preserved)")
    return filepath


def load_all_results(results_dir="results"):
    """Load all JSON result files. Returns list of record dicts."""
    records = []
    results_path = Path(results_dir)
    if not results_path.exists():
        return records
    for fp in sorted(results_path.glob("*.json")):
        with open(fp) as f:
            data = json.load(f)
        for r in data["results"]:
            r.setdefault("hardware_id", data.get("hardware_id", fp.stem))
            records.append(r)
    return records


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def _tex_escape(s):
    """Escape special LaTeX characters in a string."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("$", r"\$"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s


def _fmt_time(mean, std):
    """Format a timing value for LaTeX."""
    if mean < 0.01:
        return f"{mean:.4f} $\\pm$ {std:.4f}"
    elif mean < 1.0:
        return f"{mean:.3f} $\\pm$ {std:.3f}"
    else:
        return f"{mean:.2f} $\\pm$ {std:.2f}"


def _fmt_mem_tex(peak_rss_bytes, gpu_peak_delta_bytes=None,
                 gpu_peak_bytes=None):
    """Format a memory value for LaTeX.

    Preference order (for GPU backends):
      1. gpu_peak_delta_bytes  — XLA peak growth across the timed run,
                                 subtracting one-time import / machine
                                 load overhead (most informative).
      2. gpu_peak_bytes        — XLA peak_bytes_in_use at end of run
                                 (fallback when delta unavailable).
      3. peak_rss_bytes        — child subprocess host RSS (CPU backends).
    Returns a human-readable MB/GB string.
    """
    def _h(n):
        if n is None or n <= 0:
            return "---"
        if n >= 1024**3:
            return f"{n / 1024**3:.2f} GB"
        if n >= 1024**2:
            return f"{n / 1024**2:.0f} MB"
        if n >= 1024:
            return f"{n / 1024:.0f} KB"
        return f"{n} B"

    if gpu_peak_delta_bytes is not None and gpu_peak_delta_bytes > 0:
        return _h(gpu_peak_delta_bytes)
    if gpu_peak_bytes is not None and gpu_peak_bytes > 0:
        return _h(gpu_peak_bytes)
    return _h(peak_rss_bytes)


def _generate_machine_stats_table(results_dir, tables_dir):
    """Generate a LaTeX table fragment with machine stats per host."""
    results_path = Path(results_dir)
    tables_path = Path(tables_dir)
    tables_path.mkdir(parents=True, exist_ok=True)

    host_stats = {}
    for fp in sorted(results_path.glob("*.json")):
        with open(fp) as f:
            data = json.load(f)
        hw_id = data.get("hardware_id", fp.stem)
        stats = data.get("machine_stats", {})
        host_stats[hw_id] = stats

    if not host_stats:
        return

    display_keys = [
        ("uname_string", "System"),
        ("machdep.cpu.brand_string", "CPU"),
        ("cpu_model", "CPU"),
        ("hw.physicalcpu", "Physical cores"),
        ("hw.logicalcpu", "Logical cores"),
        ("cpu_count", "CPU count"),
        ("hw.cpufrequency_max", "CPU frequency"),
        ("hw.memsize", "Memory"),
        ("mem_total", "Memory"),
        ("python_version", "Python"),
        ("platform", "Platform"),
    ]

    filepath = tables_path / "machine_stats.tex"
    lines = []
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l p{0.7\textwidth}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Property} & \textbf{Value} \\")
    lines.append(r"\midrule")

    for hw_id, stats in host_stats.items():
        lines.append(r"\multicolumn{2}{l}{\textbf{" + _tex_escape(hw_id) + r"}} \\")
        lines.append(r"\midrule")
        seen_labels = set()
        for key, label in display_keys:
            if key in stats and label not in seen_labels:
                val = str(stats[key])
                if key == "hw.memsize":
                    try:
                        gb = int(val) / (1024 ** 3)
                        val = f"{gb:.0f} GB"
                    except ValueError:
                        pass
                elif key == "hw.cpufrequency_max":
                    try:
                        ghz = int(val) / 1e9
                        val = f"{ghz:.2f} GHz"
                    except ValueError:
                        pass
                lines.append(f"{_tex_escape(label)} & {_tex_escape(val)} \\\\")
                seen_labels.add(label)
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines.pop()
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Machine stats table written: {filepath}")


def generate_tables(results_dir="results", tables_dir="tables"):
    """Generate LaTeX table fragments from results JSON."""
    records = load_all_results(results_dir)
    if not records:
        print("No results found -- skipping table generation.")
        return

    tables_path = Path(tables_dir)
    tables_path.mkdir(parents=True, exist_ok=True)

    _generate_machine_stats_table(results_dir, tables_dir)

    hw_map = {}
    for r in records:
        hw = r["hardware_id"]
        hw_map.setdefault(hw, []).append(r)

    all_backends = sorted({r["backend"] for r in records})

    for hw, hw_records in hw_map.items():
        # Build lookups: timing + memory
        time_lookup = {}
        mem_lookup = {}
        for r in hw_records:
            key = (r["problem"], r["algorithm"], r["backend"],
                   r.get("S", 0), r.get("L", 0), r.get("Li", 0), r.get("Lo", 0))
            time_lookup[key] = (r["mean_seconds"], r["std_seconds"])
            mem_lookup[key] = (r.get("peak_rss_bytes", 0),
                               r.get("gpu_peak_delta_bytes"),
                               r.get("gpu_peak_bytes"))

        safe_hw = hw.replace("/", "-").replace(" ", "_")[:40]

        # --- 1D tables ---
        backends_1d = sorted({r["backend"] for r in hw_records if r["problem"] == "1D"})
        if backends_1d:
            for algo in ALGORITHMS:
                # Timing table
                _write_grid_table(
                    tables_path / f"{safe_hw}_1D_{algo}.tex",
                    backends_1d, PARAM_GRID_1D, algo, "1D",
                    time_lookup, _fmt_time,
                    row_cols=("S", "L"))
                # Memory table
                _write_grid_table(
                    tables_path / f"{safe_hw}_1D_{algo}_mem.tex",
                    backends_1d, PARAM_GRID_1D, algo, "1D",
                    mem_lookup, _fmt_mem_tex,
                    row_cols=("S", "L"))

        # --- 2D tables ---
        backends_2d = sorted({r["backend"] for r in hw_records if r["problem"] == "2D"})
        if backends_2d:
            for algo in ALGORITHMS:
                # Timing table
                _write_grid_table(
                    tables_path / f"{safe_hw}_2D_{algo}.tex",
                    backends_2d, PARAM_GRID_2D, algo, "2D",
                    time_lookup, _fmt_time,
                    row_cols=("S", "Li", "Lo"))
                # Memory table
                _write_grid_table(
                    tables_path / f"{safe_hw}_2D_{algo}_mem.tex",
                    backends_2d, PARAM_GRID_2D, algo, "2D",
                    mem_lookup, _fmt_mem_tex,
                    row_cols=("S", "Li", "Lo"))

    # 1D includes
    _write_include_file(tables_path, hw_map, "1D", "Forward",
                        "1D Forward timings (generators)")
    _write_include_file(tables_path, hw_map, "1D", "Viterbi",
                        "1D Viterbi timings (generators)")
    _write_include_file(tables_path, hw_map, "1D", "Forward",
                        "1D Forward peak memory (generators)", suffix="_mem")
    _write_include_file(tables_path, hw_map, "1D", "Viterbi",
                        "1D Viterbi peak memory (generators)", suffix="_mem")
    # 2D includes
    _write_include_file(tables_path, hw_map, "2D", "Forward",
                        "2D Forward timings (transducers)")
    _write_include_file(tables_path, hw_map, "2D", "Viterbi",
                        "2D Viterbi timings (transducers)")
    _write_include_file(tables_path, hw_map, "2D", "Forward",
                        "2D Forward peak memory (transducers)", suffix="_mem")
    _write_include_file(tables_path, hw_map, "2D", "Viterbi",
                        "2D Viterbi peak memory (transducers)", suffix="_mem")


def _write_grid_table(filepath, backends, param_grid, algo, problem,
                      lookup, cell_formatter, row_cols):
    """Write one LaTeX table over a parameter grid.

    row_cols is ("S", "L") for 1D or ("S", "Li", "Lo") for 2D. cell_formatter
    is called with the unpacked lookup value (time: mean, std; mem: rss_bytes,
    gpu_bytes). Writes the file only if at least one cell has data.
    """
    col_spec = "r" * len(row_cols) + "r" * len(backends)
    lines = []
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    if row_cols == ("S", "L"):
        header_cells = [r"$S$", r"$L$"]
    else:
        header_cells = [r"$S$", r"$L_{\mathrm{in}}$", r"$L_{\mathrm{out}}$"]
    header_cells.extend(
        r"\texttt{" + b.replace("_", r"\_") + "}" for b in backends)
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    has_data = False
    if problem == "1D":
        param_iter = [(S, L) for S in param_grid["S"] for L in param_grid["L"]]
    else:
        param_iter = [(S, Li, Lo)
                      for S in param_grid["S"]
                      for Li in param_grid["Li"]
                      for Lo in param_grid["Lo"]]

    for params in param_iter:
        if problem == "1D":
            S, L = params
            cells = [str(S), str(L)]
            key_base = (problem, algo)
            key_suffix = lambda b: (S, L, 0, L)
        else:
            S, Li, Lo = params
            cells = [str(S), str(Li), str(Lo)]
            key_suffix = lambda b: (S, 0, Li, Lo)
        for b in backends:
            key = (problem, algo, b) + key_suffix(b)
            if key in lookup:
                has_data = True
                cells.append(cell_formatter(*lookup[key]))
            else:
                cells.append("---")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    if has_data:
        with open(filepath, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Table written: {filepath}")


def _write_include_file(tables_path, hw_map, problem, algo, caption,
                        suffix=""):
    """Write a LaTeX include file that wraps per-host tables.

    suffix is appended to the filename stem before .tex (e.g. "_mem" for
    memory tables) and to the include-file name.
    """
    is_mem = suffix == "_mem"
    caption_tail = "" if is_mem else r" (seconds, mean $\pm$ std)"
    include_lines = []
    for hw in hw_map:
        safe_hw = hw.replace("/", "-").replace(" ", "_")[:40]
        filename = f"{safe_hw}_{problem}_{algo}{suffix}.tex"
        tex_path = tables_path / filename
        if tex_path.exists():
            include_lines.append(
                r"\begin{table}[H]"
                "\n" r"\centering"
                "\n" r"\caption{" + caption + caption_tail + r".}"
                "\n" r"\small"
                "\n" r"\resizebox{\textwidth}{!}{\input{tables/" + filename + "}}"
                "\n" r"\end{table}"
                "\n"
            )
    inc_file = (tables_path /
                f"{problem.lower()}_{algo.lower()}{suffix}_includes.tex")
    with open(inc_file, "w") as f:
        f.write("\n".join(include_lines) + "\n")
    print(f"Include file written: {inc_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Machine Boss DP benchmark suite")
    parser.add_argument(
        "--backends",
        default=",".join(DEFAULT_BACKENDS),
        help=f"Comma-separated backends (default: {','.join(DEFAULT_BACKENDS)})",
    )
    parser.add_argument(
        "--reps", type=int, default=5,
        help="Number of timing repetitions (default: 5)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print config without running",
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0,
        help="Max seconds per single call (default: 60)",
    )
    parser.add_argument(
        "--tables-only", action="store_true",
        help="Generate LaTeX tables from existing results (no benchmarking)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir / "results"
    tables_dir = script_dir / "tables"

    if args.tables_only:
        generate_tables(results_dir, tables_dir)
        return

    backends = [b.strip() for b in args.backends.split(",")]
    print(f"Hardware: {hardware_id()}")
    print(f"Backends: {backends}")
    print(f"Reps: {args.reps}")
    print(f"1D grid: L={PARAM_GRID_1D['L']} S={PARAM_GRID_1D['S']}")
    print(f"2D grid: Li={PARAM_GRID_2D['Li']} Lo={PARAM_GRID_2D['Lo']} S={PARAM_GRID_2D['S']}")
    print()

    results = run_benchmarks(backends, args.reps, dry_run=args.dry_run,
                             timeout=args.timeout)
    save_results(results, results_dir)
    generate_tables(results_dir, tables_dir)


if __name__ == "__main__":
    main()
