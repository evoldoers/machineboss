#!/usr/bin/env python3

import sys
import os
import subprocess
import tempfile

if len(sys.argv) < 4:
    print(f"Usage: {sys.argv[0]} <testName> <testNameWidth> <prog> <args...> (<expected>|-fail|-idem)", file=sys.stderr)
    sys.exit(1)

expected = sys.argv[-1]
test = sys.argv[1]
test_width = int(sys.argv[2])
args = sys.argv[3:-1]

indent_test = test.rjust(test_width)

with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.out') as f_out, \
     tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.err') as f_err:
    fname_out = f_out.name
    fname_err = f_err.name

try:
    cmd = args[0] + ' ' + ' '.join("'" + a + "'" for a in args[1:])
    with open(fname_out, 'w') as fo, open(fname_err, 'w') as fe:
        result = subprocess.run(cmd, shell=True, stdout=fo, stderr=fe)
    status = result.returncode

    idem = False
    if expected == '-idem':
        expected = args[-1]
        idem = True

    reset = '\033[0m'
    bold = '\033[1m'
    red = '\033[31m'
    green = '\033[32m'
    magenta = '\033[35m'
    ok = green + '    ok' + reset
    notok = red + 'not ok' + reset

    if expected == '-fail':
        if status:
            print(f"{indent_test} {ok}: `{' '.join(args)}` failed on cue")
        else:
            with open(fname_out) as f:
                stdout_content = f.read()
            with open(fname_err) as f:
                stderr_content = f.read()
            print(f"{bold}{magenta}\n{test}{reset}")
            print(f"Standard output:\n{stdout_content}")
            print(f"Standard error:\n{stderr_content}{reset}")
            print(f"{reset}{indent_test} {notok}: `{' '.join(args)}` succeeded (was expected to fail)", file=sys.stderr)
            sys.exit(1)
    else:
        if status:
            with open(fname_out) as f:
                stdout_content = f.read()
            with open(fname_err) as f:
                stderr_content = f.read()
            print(f"{bold}{magenta}\n{test}{reset}")
            print(f"Standard output:\n{stdout_content}")
            print(f"Standard error:\n{stderr_content}{reset}")
            print(f"{reset}{indent_test} {notok}: `{' '.join(args)}` failed with exit code {status}", file=sys.stderr)
            sys.exit(1)

        if os.path.isfile(expected):
            diff = subprocess.run(['diff', fname_out, expected], capture_output=True, text=True)

            if diff.stdout:
                diff_side = subprocess.run(['diff', '-y', fname_out, expected], capture_output=True, text=True)
                print(f"{bold}{magenta}\n{test}{reset}")
                print(f"`{' '.join(args)}` does not match '{expected}':")
                print(diff_side.stdout)
                print(f"{indent_test} {notok}: `{' '.join(args)}`", file=sys.stderr)
                sys.exit(1)
            else:
                label = "is idempotent" if idem else f"matches '{expected}'"
                print(f"{indent_test} {ok}: `{' '.join(args)}` {label}")
        else:
            with open(fname_out) as f:
                actual = f.read().rstrip('\n')

            if actual == expected:
                print(f"{indent_test} {ok}: `{' '.join(args)}` = '{expected}'")
            else:
                print(f"{bold}{magenta}\n{test}{reset}")
                print(f"`{' '.join(args)}` does not match '{expected}':")
                print(actual)
                print(f"Possibly '{expected}' is a filename? (in which case, file not found)")
                print(f"{indent_test} {notok}: `{' '.join(args)}`", file=sys.stderr)
                sys.exit(1)
finally:
    os.unlink(fname_out)
    os.unlink(fname_err)
