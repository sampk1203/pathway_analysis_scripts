#!/usr/bin/env python3
"""summ.py - summarise 05.log (Li density / MSD near dopant) and 07.log (hop loops) into one HTML file.

Single dopant : python summ.py 05.log 07.log
Compare       : python summ.py --compare folderA folderB folderC      (or: --compare */)
                each folder holds that dopant's own 05.log and 07.log.
Column labels = system name from the [config] line (folder name if two systems share a name).
Output: 05_07_summary.html -> open in browser, Ctrl-A, Ctrl-C, paste into Google Docs (tables survive).
"""
import argparse, glob, html, os, re, sys

SH = r'>?\d+(?:-\d+)?A'
RAT = r'(1 \(ref\)|[\d.]+\s*\+/-\s*(?:nan|[\d.]+)\*?)'
SML = r'(n/a|nan|[\d.]+\s*\(\s*\d+\s*\))'
ROW = re.compile(r'^\s*(\S+)\s+(%s)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s+%s\s+%s\s+%s\s+%s\s+%s\s*$'
                 % (SH, RAT, RAT, RAT, SML, SML))
KEYS = ['hops', 'loop', 'palin', 'nonpal', 'net', 'robust', 'allfar', 'netfar', 'robfar', 'smallnet', 'smallloop']

TH = 'background:#e8e8e8;text-align:center;vertical-align:middle;font-weight:bold'
TD = 'text-align:center;vertical-align:middle'

DENS_DEF = 'Mean Li in shell (Li density relative to uniform ± SE; 1 = uniform)'
MSD_DEF = 'Li MSD over the lag (Å²) ± SE'
RAT_DEF = 'MSD ratio to far shell (>7 Å) ± SE; <1 = slower'


# ---------------------------------------------------------------- helpers
def esc(x): return html.escape(str(x))
def dop_name(d): return 'average' if d == 'any' else d
def shell_name(s): return s.replace('A', ' Å').replace('-', '–')
def pm(a, b): return f'{a} ± {b}'


def fmt(s):
    if s is None: return '–'
    s = re.sub(r'\s*\+/-\s*', ' ± ', s)
    s = re.sub(r'\(\s+', '(', s)
    s = re.sub(r'\s+\)', ')', s)
    return s.replace('(reference)', '(ref)')


def dens(t): return '–' if not t else f'{t[0]} ({t[1]} ± {t[2]}×)'
def msdc(t): return '–' if not t else pm(t[0], t[1])
def ratc(t): return '–' if not t else (t[2] if t[3] is None else pm(t[2], t[3]))


def th(t, c=1, r=1): return f'<th colspan="{c}" rowspan="{r}" style="{TH}">{esc(t)}</th>'
def td(t, r=1): return f'<td rowspan="{r}" style="{TD}">{esc(t)}</td>'


def table(head, body):
    o = ['<table border="1" cellspacing="0" cellpadding="4" '
         'style="border-collapse:collapse;table-layout:fixed;width:100%;font-family:Arial;font-size:8pt">']
    o += ['<tr>' + ''.join(r) + '</tr>' for r in head]
    o += ['<tr>' + ''.join(r) + '</tr>' for r in body]
    o.append('</table><br>')
    return '\n'.join(o)


# ---------------------------------------------------------------- parsers
def kind(text):
    if re.search(r'^dopant\s+shell\s+hops', text, re.M): return '07'
    if 'MSD (A^2)' in text: return '05'
    return None


def parse05(text):
    data, T, dop, tau = {}, None, None, None
    for line in text.splitlines():
        m = re.match(r'=== .*?T = (\d+) K ===', line)
        if m:
            T = int(m[1]); data[T] = {}; dop = tau = None; continue
        if T is None: continue
        m = re.match(r'\s+(\S+): Li per shell', line)
        if m:
            dop = m[1]; data[T][dop] = {'shells': {}, 'msd': {}}; tau = None; continue
        if dop is None: continue
        d = data[T][dop]
        for s, n, r, se in re.findall(r'(%s):\s*(\S+)\s*\((\S+)\s*\+/-\s*(\S+?)x\)' % SH, line):
            d['shells'][s] = (n, r, se)
        m = re.match(r'\s+tau = (\S+) ps', line)
        if m:
            tau = m[1]; d['msd'][tau] = {}; continue
        m = re.match(r'\s+(%s)\s+(\S+)\s*\+/-\s*(\S+)\s+(.*?)\s*$' % SH, line)
        if m and tau is not None:
            r = re.match(r'(\S+)\s*\+/-\s*(\S+)', m[4])
            d['msd'][tau][m[1]] = (m[2], m[3]) + ((r[1], r[2]) if r else ('1 (ref)', None))
    return data


def parse07(text):
    data, meta, T = {}, {}, None
    for line in text.splitlines():
        m = re.match(r'T = (\d+) K\s+maxloop (\d+); small = displacement < (\S+) A over (\S+) ps', line)
        if m:
            T = int(m[1]); data[T] = {}
            meta = {'maxloop': m[2], 'small': m[3], 'win': m[4]}
            continue
        m = ROW.match(line)
        if m and T is not None:
            g = m.groups()
            data[T].setdefault(g[0], {})[g[1]] = dict(zip(KEYS, g[2:]))
    return data, meta


# ---------------------------------------------------------------- definitions (07)
def defs07(meta):
    ml = meta.get('maxloop', 'maxloop'); s = meta.get('small', '?'); w = meta.get('win', '?')
    return [
        ('hops', 'Total Li hops in shell'),
        ('loop', f'% of hops erased as part of a closed loop (Li returns to a site already on its path, up to {ml} steps back; a Li\'s path is cut when it changes shell, so loops never span two shells)'),
        ('palin', '% of hops that exactly retrace an earlier path (e.g. a→b→a)'),
        ('nonpal', '% of hops closing a loop that is not an exact retrace (e.g. a→b→c→a)'),
        ('net', 'Net hops: hops not erased by loop erasure (lasting progress along a path)'),
        ('robust', f'Robust net hops: net hops where Li moved ≥ {s} Å over {w} ps (flicker removed)'),
        ('allfar', 'Hop rate per Li in shell relative to far shell (>7 Å), all hops, ± SE (1 = same as far; <1 = fewer hops)'),
        ('netfar', 'Same rate ratio to far shell, net hops only, ± SE (progress not undone)'),
        ('robfar', 'Same rate ratio to far shell, robust net hops only, ± SE'),
        ('smallnet', f'% of net hops with Li displacement < {s} Å (flicker-like); (n) = net hops tested'),
        ('smallloop', f'% of erased (loop) hops with Li displacement < {s} Å (flicker-like); (n) = erased hops tested'),
    ]


# ---------------------------------------------------------------- per-system section
def system_section(name, d05, d07, meta):
    D = defs07(meta)
    o = [f'<h2>System: {esc(name)}</h2>', '<h3>Li density and MSD around dopant</h3>']
    for T in sorted(d05):
        o.append(f'<h4>T = {T} K</h4>')
        first = next(iter(d05[T].values()))
        shells = list(first['shells'])
        head = [[th('Dopant atom', 1, 2), th(DENS_DEF, len(shells))], [th(shell_name(s)) for s in shells]]
        body = [[td(dop_name(d))] + [td(dens(v['shells'].get(s))) for s in shells] for d, v in d05[T].items()]
        o.append(table(head, body))

        taus = sorted(first['msd'], key=float)
        head = [[th('Dopant atom', 1, 2), th('Shell', 1, 2)] + [th(f'Lag τ = {t} ps', 2) for t in taus],
                [x for _ in taus for x in (th(MSD_DEF), th(RAT_DEF))]]
        body = []
        for d, v in d05[T].items():
            for i, s in enumerate(shells):
                row = [td(dop_name(d), len(shells))] if i == 0 else []
                row.append(td(shell_name(s)))
                for t in taus:
                    c = v['msd'].get(t, {}).get(s)
                    row += [td(msdc(c)), td(ratc(c))]
                body.append(row)
        o.append(table(head, body))

    o.append('<h3>Hop loops around dopant</h3>')
    for T in sorted(d07):
        o.append(f'<h4>T = {T} K</h4>')
        for grp in (D[:6], D[6:]):
            head = [[th('Dopant atom'), th('Shell (Li–dopant distance at hop; >7 Å = far)')] + [th(t) for _, t in grp]]
            body = []
            for d, v in d07[T].items():
                for i, (s, r) in enumerate(v.items()):
                    row = [td(dop_name(d), len(v))] if i == 0 else []
                    row.append(td(shell_name(s)))
                    row += [td(fmt(r[k])) for k, _ in grp]
                    body.append(row)
            o.append(table(head, body))
    return '\n'.join(o)


# ---------------------------------------------------------------- comparison section
def atoms_of(s, src):
    out = []
    for T in s['d'+src].values():
        for a in T:
            if a not in out: out.append(a)
    out.sort(key=lambda a: a != 'any')  # stable: average first
    return out


def shell_union(systems, src):
    out = []
    for s in systems:
        for T in s['d'+src].values():
            for v in T.values():
                for k in (v['shells'] if src == '05' else v):
                    if k not in out: out.append(k)
    return out


def cmp_table(defn, systems, src, getter):
    cols = [(s, a) for s in systems for a in atoms_of(s, src)]
    Ts = sorted(set().union(*[set(s['d'+src]) for s in systems]))
    shells = shell_union(systems, src)
    head = [[th('Temperature (K)', 1, 3), th('Shell', 1, 3), th(defn, len(cols))],
            [th(s['name'], len(atoms_of(s, src))) for s in systems],
            [th(dop_name(a)) for _, a in cols]]
    body = []
    for T in Ts:
        for i, sh in enumerate(shells):
            row = [td(T, len(shells))] if i == 0 else []
            row.append(td(shell_name(sh)))
            row += [td(getter(s, T, a, sh)) for s, a in cols]
            body.append(row)
    return table(head, body)


def compare_section(systems):
    o = ['<h2>Comparison between dopants</h2>',
         '<p>Each table: rows = temperature and shell; columns = dopant system, then dopant atom '
         '(average = nearest dopant to each Li). "–" = not available.</p>']

    def g05(s, T, a):
        return s['d05'].get(T, {}).get(a, {})

    taus = sorted({t for s in systems for T in s['d05'].values() for v in T.values() for t in v['msd']}, key=float)
    o.append('<h3>Li density around dopant</h3>')
    o.append(cmp_table(DENS_DEF, systems, '05', lambda s, T, a, sh: dens(g05(s, T, a).get('shells', {}).get(sh))))
    for t in taus:
        o.append(f'<h3>Li MSD, lag τ = {t} ps</h3>')
        o.append(cmp_table(MSD_DEF + f' (τ = {t} ps)', systems, '05',
                           lambda s, T, a, sh, t=t: msdc(g05(s, T, a).get('msd', {}).get(t, {}).get(sh))))
        o.append(cmp_table(RAT_DEF + f' (τ = {t} ps)', systems, '05',
                           lambda s, T, a, sh, t=t: ratc(g05(s, T, a).get('msd', {}).get(t, {}).get(sh))))

    o.append('<h3>Hop loops around dopant</h3>')
    for k, t in defs07(systems[0]['meta']):
        o.append(cmp_table(t, systems, '07',
                           lambda s, T, a, sh, k=k: fmt(s['d07'].get(T, {}).get(a, {}).get(sh, {}).get(k))))
    return '\n'.join(o)


NOTES0 = ('<h2>Notes</h2><p>± = delete-one-block jackknife SE over 5 time blocks of a single run (differences &lt; 2 SE are noise). '
         '* = fewer than 20 events in that cell (value unreliable). nan = SE undefined (&lt; 5 events). n/a = no testable hops. '
         '1 (ref) = far shell (>7 Å), equal to 1 by definition. Each temperature is an independent run.')


def notes(raw):
    """NOTES0 + what the logs say about the dopant position behind the shells and the 05 ownership rule."""
    sm = sorted({m[1] for _, t5, t7 in raw for t in (t5, t7) for m in [re.search(r'dopant smoothing: ([\d.]+) ps', t)] if m})
    own = sorted({m[1] for _, t5, _ in raw for m in [re.search(r'<Li used per origin> \[(start|resident)\]', t5)] if m})
    s = ' Shell of a Li = its distance to the dopant in each frame; the dopant position is '
    s += ('smoothed over %s ps (centred moving average, removes vibration)' % ' / '.join(sm)) if sm else 'taken per frame (smoothing not stated in the logs: older run?)'
    s += '.' + (' MSD window ownership: %s (start = shell at the window start; resident = Li stayed in the shell the whole window, biased low for inner shells).' % ' / '.join(own) if own else '')
    if len(sm) > 1 or len(own) > 1: s += ' WARNING: systems were run with different settings.'
    return NOTES0 + s + '</p>'


# ---------------------------------------------------------------- main
def load(p):
    try:
        text = open(p, encoding='utf-8', errors='replace').read()
    except OSError as e:
        sys.exit(f'cannot read {p}: {e}')
    k, m = kind(text), re.search(r"system '([^']+)'", text)
    if not k or not m: sys.exit(f'{p}: not a 05/07 log (no table or [config] system line found)')
    return k, m[1], text


def load_pair(p05, p07):
    k5, n5, t5 = load(p05)
    k7, n7, t7 = load(p07)
    if (k5, k7) != ('05', '07'): sys.exit(f'{p05} must be a 05 log and {p07} a 07 log (detected {k5}, {k7})')
    if n5 != n7: sys.exit(f"system name differs: {p05} has '{n5}', {p07} has '{n7}'")
    return n5, t5, t7


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='+', help='single: 05.log 07.log | --compare: folders')
    ap.add_argument('--compare', action='store_true', help='each path is a folder with 05.log and 07.log')
    ap.add_argument('-o', '--out', default='05_07_summary.html')
    a = ap.parse_args()

    if a.compare:
        dirs = []
        for p in a.paths:
            dirs += sorted(glob.glob(p)) or [p]
        dirs = [d for d in dirs if os.path.isdir(d)] or sys.exit('--compare needs folders, each with 05.log and 07.log')
        if len(dirs) < 2: sys.exit('--compare needs >= 2 folders')
        raw = [(d,) + load_pair(os.path.join(d, '05.log'), os.path.join(d, '07.log')) for d in dirs]
        names = [r[1] for r in raw]
        # duplicate system names -> label by folder name instead
        raw = [(n if names.count(n) == 1 else os.path.basename(os.path.normpath(d)), t5, t7) for d, n, t5, t7 in raw]
    else:
        if len(a.paths) != 2: sys.exit('single mode: python summ.py 05.log 07.log  (or use --compare with folders)')
        raw = [load_pair(*sorted(a.paths, key=lambda p: kind(open(p, errors='replace').read()) or ''))]

    systems = []
    for n, t5, t7 in raw:
        d07, meta = parse07(t7)
        s = {'name': n, 'd05': parse05(t5), 'd07': d07, 'meta': meta}
        if not s['d05'] or not d07: sys.exit(f"system '{n}': parsed no data (check log format)")
        systems.append(s)
    if len({tuple(sorted(s['meta'].items())) for s in systems}) > 1:
        print('warning: 07 settings (maxloop/small/window) differ between systems; definitions use the first', file=sys.stderr)

    o = ['<html><head><meta charset="utf-8"><title>05/07 summary</title></head>'
         '<body style="font-family:Arial">', '<h1>05/07 summary</h1>']
    o += [system_section(s['name'], s['d05'], s['d07'], s['meta']) for s in systems]
    if a.compare: o.append(compare_section(systems))
    o += [notes(raw), '</body></html>']
    open(a.out, 'w', encoding='utf-8').write('\n'.join(o))
    print(f'wrote {a.out}  ({len(systems)} system(s))')


if __name__ == '__main__':
    main()
