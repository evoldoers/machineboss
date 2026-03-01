#!/usr/bin/env python3

import argparse
import sys
from fractions import Fraction

def main():
    parser = argparse.ArgumentParser(description='Mix-radix encoder generator')
    parser.add_argument('blocklen', type=int, help='Block length')
    parser.add_argument('eofprob', type=str, help='EOF probability')
    parser.add_argument('--maxradix', '-m', type=int, default=4, help='Limit maximum radix (default: 4)')
    parser.add_argument('--wide', '-w', action='store_true', help="Don't merge identical states")
    parser.add_argument('--deep', '-d', action='store_true', help="Don't prune unnecessary states")
    parser.add_argument('--pure', '-p', action='store_true', help="Don't shrink input intervals after encoding each input word")
    parser.add_argument('--keeproots', '-k', action='store_true', help="Don't merge input word states at the root of each output tree")
    parser.add_argument('--rational', action='store_true', help='Use exact rational math (fractions.Fraction)')
    parser.add_argument('--intervals', '-i', action='store_true', help='Show input & output intervals for output states')
    parser.add_argument('--lr', '-l', action='store_true', help='Rank dotfile from left-to-right')
    parser.add_argument('--stats', '-s', action='store_true', help='Collect & print code statistics')
    parser.add_argument('--json', '-j', action='store_true', help='Print transducer in JSON format, not GraphViz DOT format')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print lots of stuff on stderr')
    args = parser.parse_args()

    nomerge = args.wide or args.intervals

    def new_number(val):
        return Fraction(val) if args.rational else float(Fraction(val))

    msglen = args.blocklen
    peof = new_number(args.eofprob)
    pbit = (1 - peof) / 2
    if args.verbose:
        print(f'pbit={pbit} peof={peof}', file=sys.stderr)

    epsilon = 'e'
    div = '/'
    eofsym = '$'
    cprob = {'0': pbit, '1': pbit, eofsym: peof}
    alph = ['0', '1', eofsym]

    def printable(word):
        return word.replace(eofsym, 'x')

    radices = list(range(2, args.maxradix + 1))

    # Generate prefix tree & input words
    state = [{'word': '', 'dest': {}, 'p': new_number('1'), 'start': True}]
    prefix_index = [0]
    word_index = []
    while prefix_index:
        pi = prefix_index.pop(0)
        prefix = state[pi]
        pword = prefix['word']
        for c in alph:
            child_index = len(state)
            child = {'word': pword + c, 'dest': {}, 'p': prefix['p'] * cprob[c]}
            prefix['dest'][f'{c}{div}{epsilon}'] = child_index
            state.append(child)
            if c == eofsym or len(child['word']) >= msglen:
                child['input'] = True
                word_index.append(child_index)
            else:
                child['prefix'] = True
                prefix_index.append(child_index)

    # Sort by probability & find intervals
    sorted_word_index = sorted(word_index, key=lambda i: (-state[i]['p'], state[i]['word']))
    norm = sum(state[i]['p'] for i in sorted_word_index)
    for i in sorted_word_index:
        state[i]['p'] /= norm
    pmin = new_number('0')
    scale = new_number('1')
    all_out_index = list(sorted_word_index)
    final_index = []

    def find_digit(m, d, e, radix):
        ds = [d + (e - d) * k / radix for k in range(radix + 1)]
        digits = [k for k in range(radix) if ds[k] <= m and ds[k + 1] > m]
        if len(digits) != 1:
            print(f'(D,E)=({d},{e}) m={m} radix={radix} ds={ds}', file=sys.stderr)
            raise RuntimeError("Couldn't find subinterval")
        digit = digits[0]
        return digit, ds[digit], ds[digit + 1]

    def generate_tree(root_index):
        output_index = [root_index]
        subtree = []
        final = []
        while output_index:
            oi = output_index.pop(0)
            output = state[oi]
            a, b, m = output['A'], output['B'], output['m']
            for radix in radices:
                digit, d, e = find_digit(m, output['D'], output['E'], radix)
                outsym = f'{digit}_{radix}'
                outseq = output['outseq'] + (' ' if output['outseq'] else '') + outsym
                child_index = len(state)
                child = {'dest': {}, 'A': a, 'B': b, 'D': d, 'E': e, 'm': m, 'outseq': outseq}
                output['dest'][f'{epsilon}{div}{outsym}'] = child_index
                state.append(child)
                if d >= a and e <= b:
                    final.append(child_index)
                else:
                    output_index.append(child_index)
                subtree.append(child_index)
        return subtree, final

    for i in sorted_word_index:
        pmax = pmin + state[i]['p'] * scale
        m = (pmin + pmax) / 2
        state[i]['A'] = pmin
        state[i]['B'] = pmax
        state[i]['m'] = m
        state[i]['D'] = new_number('0')
        state[i]['E'] = new_number('1')
        state[i]['outseq'] = ''
        pmin = pmax
        if args.verbose:
            print(f"P({state[i]['word']})={state[i]['p']} [A,B)=[{state[i]['A']},{state[i]['B']}) m={m}", file=sys.stderr)
        subtree, final = generate_tree(i)
        all_out_index.extend(subtree)
        final_index.extend(final)
        if args.verbose:
            print(f"Created {len(final)} states to encode {state[i]['word']}", file=sys.stderr)
        if not args.pure:
            new_pmax = max(state[f]['E'] for f in final)
            if new_pmax < pmax:
                mul = (1 - new_pmax) / (1 - pmax)
                if args.verbose:
                    print(f'Shrinking B from {pmax} to {new_pmax}, increasing available space by factor of {mul}', file=sys.stderr)
                scale *= mul
                pmin = new_pmax

    # If any nodes have a unique output sequence, remove all their descendants
    n_outseq = {}
    for i in all_out_index:
        seq = state[i].get('outseq', '')
        n_outseq[seq] = n_outseq.get(seq, 0) + 1

    def remove_descendants(idx):
        for label, dest_idx in list(state[idx]['dest'].items()):
            if args.verbose:
                print(f"Removing #{dest_idx} ({state[dest_idx].get('outseq', '')})", file=sys.stderr)
            remove_descendants(dest_idx)
            state[dest_idx]['removed'] = True
        state[idx]['dest'] = {}

    valid_out_index = []
    for i in all_out_index:
        s = state[i]
        if not s.get('removed'):
            if not args.deep and n_outseq.get(s.get('outseq', ''), 0) == 1:
                if args.verbose:
                    print(f"Pruning #{i}: output sequence ({s.get('outseq', '')}) is unique", file=sys.stderr)
                remove_descendants(i)
            valid_out_index.append(i)

    # Statistics
    stats_lines = []
    if args.stats:
        import re
        def get_leaves(idx):
            s = state[idx]
            kids = list(s['dest'].values())
            if kids:
                result = []
                for k in kids:
                    result.extend(get_leaves(k))
                return result
            return [idx]

        radix_count = {}
        for i in sorted_word_index:
            word = state[i]['word']
            outseqs = [state[l].get('outseq', '') for l in get_leaves(i)]
            outseqs = [re.sub(r'\d_', '', s) for s in outseqs]
            outseqs = [s.replace(' ', '') for s in outseqs]
            outseqs.sort()
            if args.verbose:
                print(f'Radices for {word}: {" ".join(outseqs)}', file=sys.stderr)
            if eofsym not in word:
                for seq in outseqs:
                    radix_count[seq] = radix_count.get(seq, 0) + 1

        radix_seqs = sorted(radix_count.keys())
        stats_lines.append('Frequencies of radix sequences for non-EOF codewords:\n')
        for seq in radix_seqs:
            stats_lines.append(f'{seq} {radix_count[seq]}\n')
        stats_lines.append('Mean bits/output symbol for pure-radix sequences:\n')
        for radix in radices:
            s = Fraction(0)
            n = Fraction(0)
            for seq in radix_seqs:
                if re.match(f'^{radix}+$', seq):
                    n += radix_count[seq]
                    s += radix_count[seq] * len(seq)
            if n > 0:
                stats_lines.append(f'{radix} {Fraction(msglen) / (s / n)}\n')

    # Find output tree for each node, merge equivalence sets
    state.append({'end': True, 'dest': {}})
    state[0]['dest'][f'{epsilon}{div}{epsilon}'] = len(state) - 1
    equiv_index = {'()': [0]}
    for output_index in reversed(valid_out_index):
        output = state[output_index]
        dest_labels = sorted(output['dest'].keys())
        dest_indices = [output['dest'][l] for l in dest_labels]
        dest_subtrees = [state[di].get('subtree') for di in dest_indices]
        undef = [k for k, st in enumerate(dest_subtrees) if st is None]
        if undef:
            raise RuntimeError(f'State {output_index} child subtree(s) not defined ({[dest_indices[u] for u in undef]}). Postorder?')
        subtree = '(' + ','.join(dest_subtrees[k] + dest_labels[k] for k in range(len(dest_labels))) + ')'
        if args.keeproots and output.get('input'):
            subtree += ' ' + output['word']
        if nomerge and subtree != '()':
            subtree += '[' + output.get('outseq', '') + ']'
        output['subtree'] = subtree
        equiv_index.setdefault(subtree, []).append(output_index)

    for subtree in equiv_index:
        if len(equiv_index[subtree]) > 1:
            equiv_index[subtree] = sorted(equiv_index[subtree])
            if args.verbose:
                print(f'Merging ({" ".join(str(x) for x in equiv_index[subtree])}) with subtree {subtree}', file=sys.stderr)

    equiv_map = [equiv_index[state[i]['subtree']][0] if 'subtree' in state[i] else i for i in range(len(state))]
    for s in state:
        for label in list(s['dest'].keys()):
            s['dest'][label] = equiv_map[s['dest'][label]]

    # Assign IDs
    n_code_states = 0
    n_states = 0
    n_transitions = 0
    unique_state = []
    for i in equiv_map:
        s = state[i]
        if not s.get('removed'):
            if 'id' not in s:
                n_states += 1
                n_transitions += len(s['dest'])
                if s.get('begin'):
                    s['id'] = 'B'
                elif s.get('start'):
                    s['id'] = 'S'
                elif s.get('end'):
                    s['id'] = 'E'
                elif s.get('prefix'):
                    s['id'] = 'P' + s['word']
                elif s.get('input'):
                    s['id'] = 'W' + printable(s['word'])
                else:
                    n_code_states += 1
                    s['id'] = 'C' + str(n_code_states)
                unique_state.append(s)

    # Print stats
    if args.stats:
        sys.stdout.write(''.join(stats_lines))
        print(f'Number of states: {n_states}')
        print(f'Number of transitions: {n_transitions}')
        return

    # Print JSON
    if args.json:
        for n, s in enumerate(unique_state):
            s['n'] = n
        out_lines = []
        for s in unique_state:
            trans_parts = []
            for label in sorted(s['dest'].keys()):
                parts = label.split(div)
                if len(parts) == 2:
                    inp, outp = parts
                    dest = s['dest'][label]
                    t = ''
                    if inp and inp != epsilon:
                        t += f'"in":"{inp}",'
                    if outp and outp != epsilon:
                        t += f'"out":"{outp}",'
                    t += f'"to":{state[dest]["n"]}'
                    trans_parts.append('{' + t + '}')
            fields = [f'"n":{s["n"]}', f'"id":"{s["id"]}"', '"trans":[' + ','.join(trans_parts) + ']']
            out_lines.append('{' + ','.join(fields) + '}')
        print('{"state": [')
        print(' ' + ',\n '.join(out_lines))
        print(']}')
        return

    # Print DOT format
    print('digraph G {')
    if args.lr:
        print(' rankdir=LR;')
    for s in unique_state:
        if s.get('start'):
            label = 'START'
            shape = 'box'
            style = 'solid'
        elif s.get('prefix'):
            label = f'in: {s["word"][-1]}\\nprefix:{s["word"]}'
            shape = 'circle'
            style = 'solid'
        elif s.get('input'):
            label = f'in: {s["word"][-1]}\\nprefix: {s["word"]}\\nout: '
            for radix in radices:
                import re
                digits = ''.join(m.group(1) for key in s['dest'] for m in [re.match(rf'(\d)_{radix}', key.split(div)[-1])] if m)
                label += digits + '/'
            label = label[:-1]
            shape = 'doublecircle'
            style = 'solid'
        else:
            label = 'out: '
            import re
            for radix in radices:
                digits = ''.join(m.group(1) for key in s['dest'] for m in [re.match(rf'(\d)_{radix}', key.split(div)[-1])] if m)
                label += digits + '/'
            label = label[:-1]
            shape = 'box'
            style = 'solid'
        if args.intervals and 'A' in s:
            a_val = float(s['A']) if args.rational else s['A']
            b_val = float(s['B']) if args.rational else s['B']
            d_val = float(s['D']) if args.rational else s['D']
            e_val = float(s['E']) if args.rational else s['E']
            label += f'\\n[A,B) = [{a_val},{b_val})'
            label += f'\\n[D,E) = [{d_val},{e_val})'
        print(f' {s["id"]} [style={style};shape={shape};label="{label}"];')
    for src in unique_state:
        for label, dest_index in src['dest'].items():
            dest = state[dest_index]
            if label.endswith(epsilon):
                style = 'dotted'
                color = 'black'
                src_attrs = 'dir=both;arrowtail=odot;'
            elif label.endswith('4'):
                style = 'bold'
                color = 'darkslategray'
                src_attrs = ''
            elif label.endswith('3'):
                style = 'solid'
                color = 'darkslategray'
                src_attrs = ''
            elif label.endswith('2'):
                style = 'dashed'
                color = 'darkslategray'
                src_attrs = ''
            else:
                style = 'none'
                color = 'darkslategray'
                src_attrs = ''
            print(f' {src["id"]} -> {dest["id"]} [style={style};color={color};{src_attrs}arrowhead=empty;];')
    print('}')


if __name__ == '__main__':
    main()
