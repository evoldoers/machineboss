#!/usr/bin/env python3

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Generate Hamming(7,4) WFST')
    parser.add_argument('--json', '-j', action='store_true', help='Print transducer in JSON format, not GraphViz DOT')
    parser.add_argument('--tex', '-t', action='store_true', help='Add LaTeX transition labels to DOT output')
    args = parser.parse_args()

    trans_map = {}
    state_index = {}
    state_names = []

    def binstr(n, d):
        s = ''
        for _ in range(d):
            s = ('1' if n % 2 else '0') + s
            n //= 2
        return s

    def bit(n, pos):
        return 1 if (n & (1 << (4 - pos))) else 0

    def parity(n, *positions):
        return sum(bit(n, p) for p in positions) % 2

    def add_trans(src, dest, inp, out):
        if args.json:
            trans_map.setdefault(src, []).append((inp, out, dest))
        else:
            if args.tex:
                in_label = f'{inp}_2' if inp is not None else '\\epsilon'
                out_label = f'{out}_2' if out is not None else '\\epsilon'
                print(f'{src} -> {dest} [label=${in_label}/{out_label}$];')
            else:
                in_label = str(inp) if inp is not None else ''
                out_label = str(out) if out is not None else ''
                print(f'{src} -> {dest} [label="{in_label}/{out_label}"];')

    def add_state(state, label=None):
        if label is None:
            label = ''
        if args.json:
            state_index[state] = len(state_names)
            state_names.append(state)
        else:
            print(f'{state} [label="{label}"];')

    # Header
    if not args.json:
        print('digraph G {')
        print('rankdir=LR;')

    # Build transitions
    for n in range(2):
        add_trans('S', binstr(n, 1), n % 2, None)

    for n in range(4):
        add_trans(binstr(n >> 1, 1), binstr(n, 2), n % 2, None)

    for n in range(8):
        add_trans(binstr(n >> 1, 2), binstr(n, 3), n % 2, None)

    for n in range(16):
        add_trans(binstr(n >> 1, 3), 'p1_' + binstr(n, 4), n % 2, parity(n, 1, 2, 4))

    for n in range(16):
        add_trans('p1_' + binstr(n, 4), 'p2_' + binstr(n, 4), None, parity(n, 1, 3, 4))

    for n in range(16):
        add_trans('p2_' + binstr(n, 4), 'd1_' + binstr(n, 3), None, bit(n, 1))

    for n in range(8):
        add_trans('d1_' + binstr(n, 3), 'p3_' + binstr(n, 3), None, parity(n, 2, 3, 4))

    for n in range(8):
        add_trans('p3_' + binstr(n, 3), 'd2_' + binstr(n, 2), None, bit(n, 2))

    for n in range(4):
        add_trans('d2_' + binstr(n, 2), 'd3_' + binstr(n, 1), None, bit(n, 3))

    for n in range(2):
        add_trans('d3_' + binstr(n, 1), 'S', None, bit(n, 4))

    # Register states
    add_state('S', 'S')

    for n in range(2):
        add_state(binstr(n, 1))
        add_state('d3_' + binstr(n, 1))

    for n in range(4):
        add_state(binstr(n, 2))
        add_state('d2_' + binstr(n, 2))

    for n in range(8):
        add_state(binstr(n, 3))
        add_state('d1_' + binstr(n, 3))
        add_state('p3_' + binstr(n, 3))

    for n in range(16):
        add_state('p1_' + binstr(n, 4))
        add_state('p2_' + binstr(n, 4))

    if args.json:
        add_state('E')

    # Footer
    if args.json:
        print('{"state": [')
        for s in range(len(state_names)):
            sid = state_names[s]
            trans_list = trans_map.get(sid, [])
            parts = []
            for inp, out, dest in trans_list:
                t = '{'
                fields = []
                if inp is not None:
                    fields.append(f'"in":"{inp}"')
                if out is not None:
                    fields.append(f'"out":"{out}"')
                fields.append(f'"to":{state_index[dest]}')
                t += ','.join(fields) + '}'
                parts.append(t)
            comma = ',' if s < len(state_names) - 1 else ''
            print(f' {{"n":{s},"id":"{sid}","trans":[{",".join(parts)}]}}{comma}')
        print(']}')
    else:
        print('}')


if __name__ == '__main__':
    main()
