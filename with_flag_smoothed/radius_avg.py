import sys, numpy as np, pandas as pd

# usage: python radius_avg.py radius_summary.csv [tau_ps ...]
# one value per dopant: valid sigmoid fits only (no '*', A >= 0.1), averaged over the T (and tau) present.
ZEFF = {'Al': 8.85, 'Fe': 6.60, 'Ga': 9.85, 'Mg': 7.85, 'Zn': 8.85}      # Slater Z_eff of the outermost electron of the ion

csv = sys.argv[1]
taus = [float(x) for x in sys.argv[2:]]

d = pd.read_csv(csv)
d = d[d.level == 'element']
cap = d[d.model == 'free'][['name', 'T', 'tau_ps', 'r_max_used']]
s = d[d.model == 'sigmoid'].drop(columns='r_max_used').merge(cap, on=['name', 'T', 'tau_ps'])
s = s[(s.A >= 0.1) & (s.r95 > 0) & (s.r95 <= s.r_max_used)]
if taus:
    keep = np.zeros(len(s), bool)
    for t in taus: keep |= np.isclose(s.tau_ps, t, rtol=0.05)
    s = s[keep]

g = s.groupby('name')
out = pd.DataFrame({'n': g.size(),
                    'r95_mean': g.r95.mean(), 'r95_std': g.r95.std(), 'r95_median': g.r95.median(),
                    'r50_mean': g.r50.mean(), 'r50_std': g.r50.std(), 'r50_median': g.r50.median(),
                    'A_mean': g.A.mean(), 'A_std': g.A.std(), 'B_mean': g.B.mean()})
out['r95_sem'] = out.r95_std / np.sqrt(out.n)
out['r50_sem'] = out.r50_std / np.sqrt(out.n)
print(out.round(2).to_string())

# does anything track Slater Z_eff? (n = number of dopants, so very little statistical power)
from scipy.stats import pearsonr, spearmanr
nm = [k for k in out.index if k in ZEFF]
if len(nm) >= 4:
    z = [ZEFF[k] for k in nm]
    print(f'\nvs Slater Z_eff ({len(nm)} dopants):')
    for c in ('r95_mean', 'r50_mean', 'A_mean'):
        r, p = pearsonr(z, out.loc[nm, c]); rs, ps = spearmanr(z, out.loc[nm, c])
        print(f'  {c:9s} Pearson {r:+.2f} (p={p:.2f})   Spearman {rs:+.2f} (p={ps:.2f})')
