"""
Statistical significance tests: Diebold-Mariano (HLN, HAC) + Holm, block bootstrap, Friedman/Nemenyi CD, Wilcoxon.
Corresponds to Cell 22 of cmsnts_latest.ipynb.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from config import BOOTSTRAP_BLOCK_S, BIN_MS, BOOTSTRAP_REPS, ALPHA, TABLES, PROPOSED
from utils.helpers import checkpoint, savefig

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1, run)
    return adj

def test_preds(study, ds, model):
    if model in CLASSICAL_MODELS:
        return [CLASSICAL_PREDS[(study, ds, model)]['test']]
    return [v['test'] for k, v in DEEP_PREDS.items() if k[0] == study and k[1] == ds and k[2] == model]

def abs_err(study, D, model, h=None):
    ps = test_preds(study, D.name, model)      # error of each seed, averaged over seeds
    e = np.mean([np.abs(p - D.splits['test'].y) for p in ps], 0)
    return e.mean(1) if h is None else e[:, h - 1]

def diebold_mariano(e1, e2, lag, h=HORIZON):
    d = e1 - e2; n = len(d); dc = d - d.mean()
    g = [np.dot(dc[k:], dc[:n - k]) / n for k in range(lag + 1)]
    var = g[0] + 2 * sum((1 - k / (lag + 1)) * g[k] for k in range(1, lag + 1))   # Newey–West (Bartlett)
    stat = d.mean() / np.sqrt(max(var, 1e-12) / n)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)          # Harvey–Leybourne–Newbold correction
    return stat, 2 * stats.t.sf(abs(stat), n - 1)

def block_bootstrap_rel(e_prop, e_base, day, block, reps, rng):
    """moving-block bootstrap (within each test day) of 1 - MAE(prop)/MAE(base)"""
    days = np.unique(day); num = np.zeros(reps); den = np.zeros(reps)
    for k in days:
        a, b = e_prop[day == k], e_base[day == k]; n = len(a); nb = int(np.ceil(n / block))
        ca, cb = np.r_[0, np.cumsum(a)], np.r_[0, np.cumsum(b)]
        starts = rng.integers(0, n - block + 1, size=(reps, nb))
        num += (ca[starts + block] - ca[starts]).sum(1); den += (cb[starts + block] - cb[starts]).sum(1)
    return 1 - num / den

LAG = int(BOOTSTRAP_BLOCK_S * 1000 / BIN_MS)
dm_rows, boot_rows = [], []
rng = np.random.default_rng(2027)
for ds, D in DATA.items():
    if PROPOSED not in set(MAIN[MAIN.dataset == ds].model):
        continue
    e_p = abs_err('main', D, PROPOSED); day = D.splits['test'].day
    rows_ds = []
    for b in ORDER:
        if b == PROPOSED:
            continue
        e_b = abs_err('main', D, b)
        st_, p = diebold_mariano(e_p, e_b, LAG)
        r = dict(dataset=ds, baseline=b, horizon='avg', rel_mae_impr=1 - e_p.mean() / e_b.mean(), dm_stat=st_, p=p)
        for h in REPORT_STEPS:
            eph, ebh = abs_err('main', D, PROPOSED, h), abs_err('main', D, b, h)
            sh, ph = diebold_mariano(eph, ebh, LAG, h=h)
            r[f'dm_{h * BIN_MS}ms'] = sh; r[f'p_{h * BIN_MS}ms'] = ph
        bs = block_bootstrap_rel(e_p, e_b, day, LAG, BOOTSTRAP_REPS, rng)
        r['ci_lo'], r['ci_hi'] = np.percentile(bs, [2.5, 97.5])
        rows_ds.append(r)
    adj = holm([r['p'] for r in rows_ds])
    for r, a in zip(rows_ds, adj):
        r['p_holm'] = a; r['significant'] = bool(a < ALPHA and r['dm_stat'] < 0)
    for h in REPORT_STEPS:
        for r, a in zip(rows_ds, holm([r[f'p_{h * BIN_MS}ms'] for r in rows_ds])):
            r[f'p_holm_{h * BIN_MS}ms'] = a
    dm_rows += rows_ds
DM = pd.DataFrame(dm_rows)
DM.to_csv(TABLES / 'diebold_mariano_bootstrap.csv', index=False)
if len(DM):
    show = DM[['dataset', 'baseline', 'rel_mae_impr', 'ci_lo', 'ci_hi', 'dm_stat', 'p_holm', 'significant']].copy()
    display(show.round(4))
    t4 = show.copy(); t4['Impr. % [95% CI]'] = [f'{100 * a:.1f} [{100 * b:.1f}, {100 * c:.1f}]'
                                                for a, b, c in zip(t4.rel_mae_impr, t4.ci_lo, t4.ci_hi)]
    t4['DM'] = t4.dm_stat.map(lambda v: f'{v:.2f}'); t4['p (Holm)'] = t4.p_holm.map(lambda v: f'{v:.1e}')
    t4 = t4.pivot(index='baseline', columns='dataset', values='Impr. % [95% CI]').reindex([m for m in ORDER if m != PROPOSED])
    t4.index.name = f'{PROPOSED} vs.'
    to_latex(t4, TABLES / 'table4_significance.tex',
             f'Relative MAE improvement of {PROPOSED} (%) with 95% moving-block bootstrap CI ({BOOTSTRAP_BLOCK_S:g} s blocks); '
             'all DM tests: see CSV (HLN-corrected, HAC, Holm-adjusted).', 'tab:sig')

# Friedman + Nemenyi over tasks = dataset x test day x reported horizon (seed-averaged MAE)
task = MAIN[MAIN.h.isin(REPORT_STEPS)].groupby(['dataset', 'day', 'h', 'model'])['mae'].mean().unstack('model')
task = task[[m for m in ORDER if m in task.columns]].dropna(axis=1)
RANKS = task.rank(axis=1)
MEAN_RANK = RANKS.mean().sort_values()
k, N = task.shape[1], task.shape[0]
fried = stats.friedmanchisquare(*[task[c].values for c in task.columns]) if k >= 3 and N >= 2 else None
try:
    q = stats.studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2)
except Exception:
    q = stats.studentized_range.ppf(1 - ALPHA, k, 1e6) / np.sqrt(2)
CD = q * np.sqrt(k * (k + 1) / (6 * N))
print(f'Friedman over {N} tasks, {k} models: chi2={fried.statistic:.1f}, p={fried.pvalue:.2e}; Nemenyi CD={CD:.2f}'
      if fried else 'Friedman test not available')
display(MEAN_RANK.round(2).to_frame('mean rank'))
wil = []
if PROPOSED in task:
    for b in task.columns.drop(PROPOSED):
        d = task[PROPOSED] - task[b]
        p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        wil.append(dict(baseline=b, proposed_wins=int((d < 0).sum()), ties=int((d == 0).sum()), losses=int((d > 0).sum()), p=p))
    WIL = pd.DataFrame(wil); WIL['p_holm'] = holm(WIL.p); display(WIL.round(4))
    WIL.to_csv(TABLES / 'wilcoxon_tasks.csv', index=False)
pd.DataFrame({'mean_rank': MEAN_RANK}).assign(friedman_chi2=fried.statistic if fried else np.nan,
                                              friedman_p=fried.pvalue if fried else np.nan, nemenyi_cd=CD,
                                              tasks=N).to_csv(TABLES / 'friedman_nemenyi.csv')

def cd_diagram(ranks, cd, path):
    """Demsar-style critical-difference diagram (rank 1 = best, on the left)."""
    names = list(ranks.index); vals = ranks.values.astype(float); k = len(vals)
    cliques = []
    for i in range(k):
        j = max(jj for jj in range(i, k) if vals[jj] - vals[i] < cd)
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    half = (k + 1) // 2; rows = max(half, k - half); sp = .55
    y_lab = .45 + .22 * len(cliques)
    pad = .25 * (k - 1) + .5
    height = 1.0 + .16 * (y_lab + rows * sp)
    fig, ax = plt.subplots(figsize=(7.0, height))
    ax.set_xlim(1 - pad, k + pad); ax.set_ylim(y_lab + (rows - .3) * sp, -1.6); ax.axis('off')
    ax.hlines(0, 1, k, color='k', lw=.8)
    for t in range(1, k + 1):
        ax.vlines(t, -.12, 0, color='k', lw=.8); ax.text(t, -.25, str(t), ha='center', va='bottom', fontsize=6)
    ax.hlines(-1.1, 1, 1 + cd, color='C3', lw=1.5); ax.vlines([1, 1 + cd], -1.2, -1.0, color='C3', lw=1)
    ax.text(1 + cd / 2, -1.25, f'CD = {cd:.2f}', ha='center', va='bottom', fontsize=6, color='C3')
    for i, (n, v) in enumerate(zip(names, vals)):
        left = i < half; r = i if left else k - 1 - i; y = y_lab + r * sp
        xe = 1 - .15 if left else k + .15
        ax.plot([v, v, xe], [0, y, y], color='k', lw=.6)
        ax.text(xe - .08 if left else xe + .08, y, f'{n} ({v:.2f})', ha='right' if left else 'left', va='center',
                fontsize=6.5, fontweight='bold' if n == PROPOSED else 'normal')
    for lvl, (a, b) in enumerate(cliques):
        ax.hlines(.25 + .22 * lvl, vals[a] - .05, vals[b] + .05, lw=2.4, color='C0')
    savefig(fig, path)


if fried:
    cd_diagram(MEAN_RANK, CD, 'fig_cd_diagram')
checkpoint('analysis cell 21')
