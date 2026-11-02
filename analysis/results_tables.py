"""
Results and statistical analysis tables (Tables II, III, V, VI, VII).
Corresponds to Cells 21, 24, 25, 26 of cmsnts_latest.ipynb.
"""

import numpy as np
import pandas as pd
from config import REPORT_STEPS, TABLES, PROPOSED, RUN_TRANSFER, DS, DEEP_MODELS
from utils.helpers import checkpoint, stamp, remaining_min
from experiments.metrics import RESULTS, mae_of

def results_df():
    df = pd.DataFrame(RESULTS)
    if 'L' not in df:
        df['L'] = np.nan
    return df.drop_duplicates(['study', 'dataset', 'model', 'seed', 'day', 'h', 'L'], keep='last')

def seed_level(df, metric, hs=None):
    d = df if hs is None else df[df.h.isin(hs)]
    return d.groupby(['dataset', 'model', 'seed', 'h'])[metric].mean().reset_index()

def fmt_pm(m, s, nd=2):
    return f'{m:.{nd}f}' if not np.isfinite(s) or s == 0 else f'{m:.{nd}f} ± {s:.{nd}f}'

def latex_escape(s):
    s = str(s).replace('%', r'\%').replace('_', r'\_').replace('&', r'\&').replace('±', r'$\pm$')
    return s.replace('→', r'$\rightarrow$').replace('α', r'$\alpha$').replace('Δ', r'$\Delta$')

def to_latex(table, path, caption, label, best_lower=None):
    """booktabs table; best_lower: dict column -> numeric Series (lower is better) used for bold/underline."""
    cols = list(table.columns); env = 'table*' if len(cols) > 5 else 'table'
    lines = [r'\begin{' + env + '}[t]', r'\centering', r'\caption{' + latex_escape(caption) + '}',
             r'\label{' + label + '}', r'\scriptsize', r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{l' + 'c' * len(cols) + '}', r'\toprule']
    hdr = [latex_escape(c if not isinstance(c, tuple) else ' '.join(map(str, c))) for c in cols]
    lines.append(latex_escape(table.index.name or '') + ' & ' + ' & '.join(hdr) + r' \\'); lines.append(r'\midrule')
    for idx, row in table.iterrows():
        cells = []
        for c in cols:
            v = latex_escape(row[c])
            if best_lower is not None and c in best_lower:
                s = best_lower[c].dropna().sort_values()
                if len(s) and idx == s.index[0]:
                    v = r'\textbf{' + v + '}'
                elif len(s) > 1 and idx == s.index[1]:
                    v = r'\underline{' + v + '}'
            cells.append(v)
        lines.append(latex_escape(idx) + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{' + env + '}']
    Path(path).write_text('\n'.join(lines), encoding='utf-8')

DF = results_df()
MAIN = DF[DF.study == 'main']
ORDER = [m for m in CLASSICAL_MODELS + DEEP_MODELS + CASCADE_MODELS if m in set(MAIN.model)]
DS = list(DATASETS)

def summary(df, metric, hs):
    s = seed_level(df, metric, hs).groupby(['dataset', 'model', 'seed'])[metric].mean().reset_index()
    return s.groupby(['dataset', 'model'])[metric].agg(['mean', 'std', 'count'])

cols, numeric = {}, {}
for ds in DS:
    for h in REPORT_STEPS:
        s = summary(MAIN[MAIN.dataset == ds], 'mae', [h]).loc[ds]
        c = (ds, f'{h * BIN_MS} ms'); cols[c] = {m: fmt_pm(s.loc[m, 'mean'], s.loc[m, 'std']) for m in s.index}
        numeric[c] = s['mean']
TABLE2 = pd.DataFrame(cols).reindex(ORDER); TABLE2.index.name = 'Model'
TABLE2.insert(0, 'Venue', [MODEL_REF.get(m, '') for m in TABLE2.index])
TABLE2.columns = [c if isinstance(c, str) else f'{c[0]} {c[1]}' for c in TABLE2.columns]
num2 = {f'{c[0]} {c[1]}': v for c, v in numeric.items()}
display(TABLE2)
to_latex(TABLE2, TABLES / 'table2_mae_per_horizon.tex',
         f'Test MAE (Mbit/s) at 10/50/100 ms ahead; mean ± SD over {MAIN[MAIN.model == PROPOSED].seed.nunique()} seeds, '
         f'{len(DATASETS[DS[0]]["test"])} held-out days per dataset.',
         'tab:main', best_lower=num2)

rows, num3 = {}, {}
for ds in DS:
    sub = MAIN[MAIN.dataset == ds]
    for metric in ('mae', 'rmse', 'mase', 'skill'):
        s = summary(sub, metric, None).loc[ds]; c = f'{ds} {metric.upper()}'
        rows[c] = {m: fmt_pm(s.loc[m, 'mean'], s.loc[m, 'std'], 3 if metric in ('mase', 'skill') else 2) for m in s.index}
        num3[c] = s['mean'] * (-1 if metric == 'skill' else 1)
TABLE3 = pd.DataFrame(rows).reindex(ORDER); TABLE3.index.name = 'Model'
display(TABLE3)
to_latex(TABLE3, TABLES / 'table3_horizon_averaged.tex',
         'Horizon-averaged (10-100 ms) test errors; skill = 1 - MAE/MAE(persistence).', 'tab:avg', best_lower=num3)
FULL = MAIN.groupby(['dataset', 'model', 'h'])[['mae', 'rmse', 'mase', 'skill']].mean().reset_index()
FULL.to_csv(TABLES / 'all_metrics_per_horizon.csv', index=False)
MAIN.to_csv(TABLES / 'all_metrics_per_seed_day_horizon.csv', index=False)

AVG_MAE = summary(MAIN, 'mae', None)['mean'].unstack(0)       # model x dataset
BEST_BASELINE = {ds: AVG_MAE[ds].drop([PROPOSED] + CASCADE_MODELS, errors='ignore').idxmin() for ds in DS}
print('Strongest baseline per dataset (horizon-averaged MAE):', BEST_BASELINE)
checkpoint('analysis cell 20')


ABL = DF[DF.study == 'ablation']
if len(ABL):
    FULLNAME = f'{PROPOSED} (full)'
    both = pd.concat([MAIN[MAIN.model == PROPOSED].assign(model=FULLNAME), ABL])
    s = summary(both, 'mae', None)
    full = s.xs(FULLNAME, level='model')['mean']
    rows = {}
    for (ds, m), r in s.iterrows():
        rows.setdefault(m, {})[ds] = f"{fmt_pm(r['mean'], r['std'])} ({100 * (r['mean'] / full[ds] - 1):+.1f}%)"
    TABLE5 = pd.DataFrame(rows).T.reindex([FULLNAME] + list(ABLATION_SPECS)); TABLE5.index.name = 'Variant'
    display(TABLE5)
    to_latex(TABLE5, TABLES / 'table5_ablation.tex',
             f'Ablation: horizon-averaged test MAE (Mbit/s) of the full exit E2, mean ± SD over seeds; change vs full {PROPOSED} in parentheses.',
             'tab:ablation')
    ABL_DELTA = ((s['mean'].unstack(0) / full - 1) * 100).drop(FULLNAME, errors='ignore')
else:
    ABL_DELTA = pd.DataFrame(); print('No ablation runs finished.')
checkpoint('analysis cell 23')


KEY_MODELS = [m for m in ['Persistence', 'EWMA', 'Ridge-AR', 'HistGB', 'DLinear', 'PatchTST', 'TimesNet', 'iTransformer',
                          'TimeMixer', 'TimeXer', PROPOSED, 'CAF-E1', 'CAF-adaptive'] if m in ORDER]

def all_preds(ds, model, split):
    if model in CLASSICAL_MODELS:
        return [CLASSICAL_PREDS[('main', ds, model)][split]]
    return [v[split] for k, v in DEEP_PREDS.items() if k[0] == 'main' and k[1] == ds and k[2] == model]

ops = []
for ds, D in DATA.items():
    yv, yt = D.splits['val'].y, D.splits['test'].y
    thr = D.p95
    tr_rates = np.concatenate([to_channels(TRACE[d], BIN_MS)[:, 0] for d in DATASETS[ds]['train']])
    lo, hi = np.percentile(tr_rates, [33.3, 66.7])
    for m in KEY_MODELS:
        for pv, pt in zip(all_preds(ds, m, 'val'), all_preds(ds, m, 'test')):
            r = {'dataset': ds, 'model': m}
            for h in REPORT_STEPS:
                y, p = yt[:, h - 1], pt[:, h - 1]
                tp = np.sum((p > thr) & (y > thr)); fp = np.sum((p > thr) & (y <= thr)); fn = np.sum((p <= thr) & (y > thr))
                r[f'burst_F1_{h * BIN_MS}ms'] = 2 * tp / max(2 * tp + fp + fn, 1)
            res_v = yv - pv; n = len(res_v)
            q = np.quantile(res_v, min(1, np.ceil((n + 1) * (1 - ALPHA)) / n), axis=0)       # split-conformal, per horizon
            upper = pt + q
            r['coverage'] = float((yt <= upper).mean())
            r['headroom_mbps'] = float(q.mean())
            r['overprov_pct'] = float(100 * np.maximum(upper - yt, 0).mean() / yt.mean())
            e = np.abs(pt - yt)
            for nm, msk in (('low', yt < lo), ('mid', (yt >= lo) & (yt < hi)), ('high', yt >= hi)):
                r[f'mae_{nm}_load'] = float(e[msk].mean()) if msk.any() else np.nan
            ops.append(r)
OPS = pd.DataFrame(ops).groupby(['dataset', 'model']).mean().reset_index()
OPS.to_csv(TABLES / 'operational_metrics.csv', index=False)
display(OPS.round(3))
t6 = OPS.pivot(index='model', columns='dataset', values=f'burst_F1_{REPORT_STEPS[1] * BIN_MS}ms').reindex(KEY_MODELS).round(3)
t6b = OPS.pivot(index='model', columns='dataset', values='overprov_pct').reindex(KEY_MODELS).round(2)
TABLE6 = pd.concat({f'Burst F1 @{REPORT_STEPS[1] * BIN_MS} ms': t6, 'Over-prov. % (95% conformal)': t6b}, axis=1)
TABLE6.columns = [f'{a} {b}' for a, b in TABLE6.columns]; TABLE6.index.name = 'Model'
to_latex(TABLE6.astype(str), TABLES / 'table6_operational.tex',
         f'Burst detection (rate > train p95) and over-provisioning of a split-conformal {int(100 * (1 - ALPHA))}% upper bound.',
         'tab:ops', best_lower={c: (-TABLE6[c] if 'F1' in c else TABLE6[c]) for c in TABLE6.columns})
checkpoint('analysis cell 24')


TRANSFER = []
TRANSFER_MODELS = [m for m in ['DLinear', 'PatchTST', 'iTransformer', 'TimeXer', PROPOSED] if m in DEEP_MODELS]
if RUN_TRANSFER and remaining_min() > 8:
    for src in DS:
        for tgt in DS:
            if src == tgt:
                continue
            Dt = DATA[tgt]; pers = abs_err('main', Dt, 'Persistence').mean()
            for m in TRANSFER_MODELS:
                seeds = [k[3] for k in DEEP_PREDS if k[0] == 'main' and k[1] == src and k[2] == m][:3]
                for seed in seeds:
                    model = load_model('main', DATA[src], m, MODEL_SPECS[m], seed)
                    pt = predict(model, Dt, 'test', MODEL_SPECS[m][1])
                    ind = DEEP_PREDS.get(('main', tgt, m, seed))
                    TRANSFER.append(dict(source=src, target=tgt, model=m, seed=seed, mae=mae_of(Dt, pt, 'test'),
                                         in_domain_mae=mae_of(Dt, ind['test'], 'test') if ind else np.nan,
                                         skill=1 - np.abs(pt - Dt.splits['test'].y).mean() / pers))
    TR = pd.DataFrame(TRANSFER)
    if len(TR):
        TR['ratio_vs_in_domain'] = TR.mae / TR.in_domain_mae
        TRs = TR.groupby(['source', 'target', 'model'])[['mae', 'ratio_vs_in_domain', 'skill']].mean().reset_index()
        TRs.to_csv(TABLES / 'transfer.csv', index=False)
        TABLE7 = TRs.assign(pair=TRs.source.str.replace('MAWI-', '') + '→' + TRs.target.str.replace('MAWI-', '')) \
                    .pivot(index='model', columns='pair', values='skill').reindex(TRANSFER_MODELS).round(3)
        TABLE7.index.name = 'Model'; display(TABLE7)
        to_latex(TABLE7.astype(str), TABLES / 'table7_transfer.tex',
                 'Zero-shot cross-dataset transfer: skill vs persistence on the target test days (higher is better).',
                 'tab:transfer', best_lower={c: -TABLE7[c] for c in TABLE7.columns})
checkpoint('analysis cell 25')
