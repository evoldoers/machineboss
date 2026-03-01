#!/usr/bin/env python3

import sys
import re
import subprocess

if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} <maxPrecision> <prog> <args...>", file=sys.stderr)
    sys.exit(1)

precision = int(sys.argv[1])
prog_args = sys.argv[2:]

result = subprocess.run(prog_args, capture_output=True, text=True)

pattern = re.compile(r'\d+\.\d{' + str(precision) + r',}')
output = pattern.sub(lambda m: f'{float(m.group()):.{precision}g}', result.stdout)
sys.stdout.write(output)

sys.stderr.write(result.stderr)
sys.exit(result.returncode)
