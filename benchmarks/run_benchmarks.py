#!/usr/bin/env python3
"""Benchmark suite for Machine Boss dynamic programming implementations.

Times Forward and Viterbi algorithms across backends (C++ native, JAX CPU/GPU,
JavaScript CPU, WebGPU), machine sizes, and sequence lengths.
Outputs JSON results keyed by hostname.

Usage:
    python run_benchmarks.py [--backends cpp,jax_1d_simple,...] [--dry-run] [--reps N]
    python run_benchmarks.py --tables-only
"""

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

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


def _time_cpp_native(machine_json, algorithm, input_seq=None, output_seq=None,
                     n_reps=3, timeout=60.0):
    """Time C++ native boss CLI for forward/viterbi.

    Returns (mean_s, std_s, n_reps_completed) or None on error.
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

        # Probe call
        t0 = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout + 5)
        probe_time = time.perf_counter() - t0

        if result.returncode != 0:
            return None

        if probe_time > timeout:
            return (probe_time, 0.0, 1)

        # Full timing
        times = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            subprocess.run(cmd, capture_output=True, timeout=timeout + 5)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            if elapsed > timeout:
                break

        return (float(np.mean(times)), float(np.std(times)), len(times))
    except (subprocess.TimeoutExpired, Exception):
        return None
    finally:
        os.unlink(machine_file)


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


def _time_jax(machine_json, algorithm, input_seq=None, output_seq=None,
              strategy="auto", kernel="auto", n_reps=3, timeout=60.0,
              use_gpu=False):
    """Time JAX implementation.

    Returns (mean_s, std_s, n_reps_completed) or None on error.
    """
    if use_gpu:
        os.environ["JAX_PLATFORMS"] = "gpu"
    else:
        os.environ["JAX_PLATFORMS"] = "cpu"

    try:
        import jax
        import jax.numpy as jnp

        if use_gpu and not jax.devices("gpu"):
            return None

        from machineboss.jax.forward import log_forward
        from machineboss.jax.viterbi import log_viterbi

        jm, em = _get_jax_machine(machine_json)

        # Tokenize sequences using EvaluatedMachine's token lists
        in_tokens = None
        out_tokens = None
        if input_seq is not None:
            in_tokens = jnp.array(_tokenize_seq(input_seq, em.input_tokens))
        if output_seq is not None:
            out_tokens = jnp.array(_tokenize_seq(output_seq, em.output_tokens))

        if algorithm == "Forward":
            fn = lambda: float(log_forward(jm, in_tokens, out_tokens,
                                           strategy=strategy, kernel=kernel))
        elif algorithm == "Viterbi":
            fn = lambda: float(log_viterbi(jm, in_tokens, out_tokens,
                                           strategy=strategy, kernel=kernel))
        else:
            return None

        # Warmup / probe
        t0 = time.perf_counter()
        fn()
        probe_time = time.perf_counter() - t0

        if probe_time > timeout:
            return (probe_time, 0.0, 1)

        # Full timing (post-JIT)
        times = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            fn()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            if elapsed > timeout:
                break

        return (float(np.mean(times)), float(np.std(times)), len(times))
    except Exception as e:
        print(f" JAX ERROR: {e}")
        return None


def _time_js_cpu(machine_json, algorithm, input_seq=None, output_seq=None,
                 n_reps=3, timeout=60.0):
    """Time JavaScript CPU fallback via Node.js.

    Returns (mean_s, std_s, n_reps_completed) or None if Node.js unavailable.
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
        result = subprocess.run(
            ["node", script_file],
            capture_output=True, text=True,
            timeout=timeout * (n_reps + 2),
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout.strip())
        return (data["mean"], data["std"], data["n"])
    except Exception:
        return None
    finally:
        os.unlink(machine_file)
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
    "cpp",
    "jax_1d_simple",
    "jax_1d_optimal",
    "jax_gpu_1d",
    "js_cpu",
]

ALL_BACKENDS_2D = [
    "cpp",
    "jax_2d_simple",
    "jax_2d_optimal",
    "jax_gpu_2d",
    "js_cpu",
]

DEFAULT_BACKENDS = [
    "cpp",
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
                        if backend == "cpp":
                            timing = _time_cpp_native(
                                machine_json, algorithm,
                                output_seq=seq,
                                n_reps=n_reps, timeout=timeout)
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

                    mean_s, std_s, n_completed = timing

                    if mean_s > timeout:
                        print(f" {mean_s:.2f}s (probe > {timeout}s, skipping larger)")
                        timed_out.add(skip_key)

                        results.append({
                            "problem": "1D", "backend": backend,
                            "algorithm": algorithm, "S": S, "L": L,
                            "Li": 0, "Lo": L,
                            "mean_seconds": mean_s, "std_seconds": std_s,
                            "n_reps": 1, "hardware_id": hw,
                        })
                        continue

                    print(f" {mean_s:.4f} +/- {std_s:.4f} s")
                    results.append({
                        "problem": "1D", "backend": backend,
                        "algorithm": algorithm, "S": S, "L": L,
                        "Li": 0, "Lo": L,
                        "mean_seconds": mean_s, "std_seconds": std_s,
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
                            if backend == "cpp":
                                timing = _time_cpp_native(
                                    machine_json, algorithm,
                                    input_seq=in_seq, output_seq=out_seq,
                                    n_reps=n_reps, timeout=timeout)
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

                        mean_s, std_s, n_completed = timing

                        if mean_s > timeout:
                            print(f" {mean_s:.2f}s (probe > {timeout}s, skipping larger)")
                            timed_out.add(skip_key)

                            results.append({
                                "problem": "2D", "backend": backend,
                                "algorithm": algorithm, "S": S,
                                "L": 0, "Li": Li, "Lo": Lo,
                                "mean_seconds": mean_s, "std_seconds": std_s,
                                "n_reps": 1, "hardware_id": hw,
                            })
                            continue

                        print(f" {mean_s:.4f} +/- {std_s:.4f} s")
                        results.append({
                            "problem": "2D", "backend": backend,
                            "algorithm": algorithm, "S": S,
                            "L": 0, "Li": Li, "Lo": Lo,
                            "mean_seconds": mean_s, "std_seconds": std_s,
                            "n_reps": n_completed, "hardware_id": hw,
                        })

    return results


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def save_results(results, out_dir="results"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    hostname = platform.node() or "unknown"
    filepath = out_path / f"{hostname}.json"
    data = {
        "hardware_id": results[0]["hardware_id"] if results else hardware_id(),
        "hostname": hostname,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "machine_stats": _collect_machine_stats(),
        "results": results,
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {filepath}")
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
        # Build lookup
        lookup = {}
        for r in hw_records:
            key = (r["problem"], r["algorithm"], r["backend"],
                   r.get("S", 0), r.get("L", 0), r.get("Li", 0), r.get("Lo", 0))
            lookup[key] = (r["mean_seconds"], r["std_seconds"])

        safe_hw = hw.replace("/", "-").replace(" ", "_")[:40]

        # --- 1D tables ---
        backends_1d = sorted({r["backend"] for r in hw_records if r["problem"] == "1D"})
        if backends_1d:
            for algo in ALGORITHMS:
                filename = f"{safe_hw}_1D_{algo}.tex"
                filepath = tables_path / filename
                lines = []
                col_spec = "rr" + "r" * len(backends_1d)
                lines.append(r"\begin{tabular}{" + col_spec + "}")
                lines.append(r"\toprule")
                header = r"$S$ & $L$ & " + " & ".join(
                    r"\texttt{" + b.replace("_", r"\_") + "}" for b in backends_1d
                )
                lines.append(header + r" \\")
                lines.append(r"\midrule")

                has_data = False
                for S in PARAM_GRID_1D["S"]:
                    for L in PARAM_GRID_1D["L"]:
                        cells = [str(S), str(L)]
                        for b in backends_1d:
                            key = ("1D", algo, b, S, L, 0, L)
                            if key in lookup:
                                has_data = True
                                cells.append(_fmt_time(*lookup[key]))
                            else:
                                cells.append("---")
                        lines.append(" & ".join(cells) + r" \\")

                lines.append(r"\bottomrule")
                lines.append(r"\end{tabular}")

                if has_data:
                    with open(filepath, "w") as f:
                        f.write("\n".join(lines) + "\n")
                    print(f"Table written: {filepath}")

        # --- 2D tables ---
        backends_2d = sorted({r["backend"] for r in hw_records if r["problem"] == "2D"})
        if backends_2d:
            for algo in ALGORITHMS:
                filename = f"{safe_hw}_2D_{algo}.tex"
                filepath = tables_path / filename
                lines = []
                col_spec = "rrr" + "r" * len(backends_2d)
                lines.append(r"\begin{tabular}{" + col_spec + "}")
                lines.append(r"\toprule")
                header = r"$S$ & $L_{\mathrm{in}}$ & $L_{\mathrm{out}}$ & " + " & ".join(
                    r"\texttt{" + b.replace("_", r"\_") + "}" for b in backends_2d
                )
                lines.append(header + r" \\")
                lines.append(r"\midrule")

                has_data = False
                for S in PARAM_GRID_2D["S"]:
                    for Li in PARAM_GRID_2D["Li"]:
                        for Lo in PARAM_GRID_2D["Lo"]:
                            cells = [str(S), str(Li), str(Lo)]
                            for b in backends_2d:
                                key = ("2D", algo, b, S, 0, Li, Lo)
                                if key in lookup:
                                    has_data = True
                                    cells.append(_fmt_time(*lookup[key]))
                                else:
                                    cells.append("---")
                            lines.append(" & ".join(cells) + r" \\")

                lines.append(r"\bottomrule")
                lines.append(r"\end{tabular}")

                if has_data:
                    with open(filepath, "w") as f:
                        f.write("\n".join(lines) + "\n")
                    print(f"Table written: {filepath}")

    # Generate include files
    for hw in hw_map:
        safe_hw = hw.replace("/", "-").replace(" ", "_")[:40]

    # 1D includes
    _write_include_file(tables_path, hw_map, "1D", "Forward",
                        "1D Forward timings (generators)")
    _write_include_file(tables_path, hw_map, "1D", "Viterbi",
                        "1D Viterbi timings (generators)")
    # 2D includes
    _write_include_file(tables_path, hw_map, "2D", "Forward",
                        "2D Forward timings (transducers)")
    _write_include_file(tables_path, hw_map, "2D", "Viterbi",
                        "2D Viterbi timings (transducers)")


def _write_include_file(tables_path, hw_map, problem, algo, caption_suffix):
    """Write a LaTeX include file that wraps per-host tables."""
    include_lines = []
    for hw in hw_map:
        safe_hw = hw.replace("/", "-").replace(" ", "_")[:40]
        filename = f"{safe_hw}_{problem}_{algo}.tex"
        tex_path = tables_path / filename
        if tex_path.exists():
            include_lines.append(
                r"\begin{table}[H]"
                "\n" r"\centering"
                "\n" r"\caption{" + caption_suffix + r" (seconds, mean $\pm$ std).}"
                "\n" r"\small"
                "\n" r"\resizebox{\textwidth}{!}{\input{tables/" + filename + "}}"
                "\n" r"\end{table}"
                "\n"
            )
    inc_file = tables_path / f"{problem.lower()}_{algo.lower()}_includes.tex"
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
