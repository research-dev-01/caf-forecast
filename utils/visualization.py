"""
Visualization suite: publication figures, learning curves, diagnostic panels, and XAI plots.
Corresponds to Cells 23, 31, 32, 33, 34, 38, 39 of cmsnts_latest.ipynb.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
import torch
import torch.nn as nn

from config import (
    BIN_MS, FIGS, TABLES, DS, PROPOSED, LOOKBACK, HORIZON, DEVICE
)
from utils.helpers import checkpoint, stamp, savefig
from experiments.metrics import mae_of
from experiments.engine import load_model

fig, axes = plt.subplots(1, len(DS), figsize=(7.2, 2.4), layout='constrained', sharey=False)
axes = np.atleast_1d(axes)
for ax, (ds, D) in zip(axes, DATA.items()):
    pers = mae_of(D, CLASSICAL_PREDS[('main', ds, 'Persistence')]['val'], 'val')
    for j, name in enumerate(DEEP_MODELS):
        curves = [np.array([h['val_mae'] for h in v['history']]) for k, v in HISTORY.items()
                  if k[0] == 'main' and k[1] == ds and k[2] == name]
        if not curves:
            continue
        T = max(len(c) for c in curves)
        M = np.array([np.r_[c, np.full(T - len(c), c[-1])] for c in curves]) / pers   # stopped runs hold their last value
        mu, sd = M.mean(0), M.std(0)
        ax.plot(mu, lw=1.6 if name == PROPOSED else .8, color='k' if name == PROPOSED else f'C{j % 10}', label=name)
        ax.fill_between(range(T), mu - sd, mu + sd, alpha=.15, color='k' if name == PROPOSED else f'C{j % 10}')
    ax.axhline(1, ls='--', lw=.6, color='grey'); ax.set_title(ds); ax.set_xlabel('epoch')
    ax.set_ylim(top=1.3)
axes[0].set_ylabel('val MAE / persistence'); axes[-1].legend(ncol=2, fontsize=5)
savefig(fig, 'fig_learning_curves')
CONV = pd.DataFrame([dict(dataset=k[1], model=k[2], seed=k[3], best_epoch=v['best_epoch'], epochs=v['epochs_run'],
                          sec_per_epoch=v['sec_per_epoch'], train_s=v['train_seconds'], status=v['status'])
                     for k, v in HISTORY.items() if k[0] == 'main'])
conv_t = CONV.groupby('model').agg(best_epoch=('best_epoch', 'mean'), best_epoch_sd=('best_epoch', 'std'),
                                   epochs=('epochs', 'mean'), sec_per_epoch=('sec_per_epoch', 'mean'),
                                   train_s=('train_s', 'mean'), runs=('seed', 'count'),
                                   failures=('status', lambda s: int((s != 'ok').sum()))).reindex(DEEP_MODELS)
display(conv_t.round(2)); conv_t.to_csv(TABLES / 'convergence.csv')

# seed-to-seed variability of the test MAE (coefficient of variation across seeds)
var_t = summary(MAIN[MAIN.model.isin(DEEP_MODELS)], 'mae', None)
var_t = (var_t['std'] / var_t['mean'] * 100).unstack(0).reindex(DEEP_MODELS)
print('Seed-to-seed CV of test MAE (%):'); display(var_t.round(2))
checkpoint('analysis cell 22')


fig, axes = plt.subplots(1, len(DS), figsize=(7.2, 2.0), layout='constrained')
axes = np.atleast_1d(axes); ANCHOR = {}
for ax, (ds, D) in zip(axes, DATA.items()):
    ws = []
    for seed in [k[3] for k in DEEP_PREDS if k[0] == 'main' and k[1] == ds and k[2] == PROPOSED]:
        with torch.no_grad():
            ws.append(load_model('main', D, PROPOSED, MODEL_SPECS[PROPOSED], seed).anchor_weights().cpu().numpy())
    if not ws:
        continue
    W = np.mean(ws, 0); ANCHOR[ds] = W
    labels = ['last'] + [f'MA{s}' for s in CAF.SCALES if s < D.L] + ['mean']
    im = ax.imshow(W.T, aspect='auto', cmap='viridis', vmin=0, vmax=max(.5, W.max()))
    ax.set_yticks(range(len(labels)), labels, fontsize=5); ax.set_xticks(range(0, HORIZON, 3), [f'{(h + 1) * BIN_MS}' for h in range(0, HORIZON, 3)])
    ax.set_xlabel('horizon (ms)'); ax.set_title(ds); ax.grid(False)
fig.colorbar(im, ax=axes, shrink=.8, label='anchor weight')
savefig(fig, 'fig_caf_anchor_weights')

def occlusion(model_name, D, seeds, chans_total=len(CHANNELS), lag_block=8):
    spec = MODEL_SPECS[model_name]; out_c, out_t = [], []
    for seed in seeds:
        m = load_model('main', D, model_name, spec, seed); chans = spec[1]
        base = mae_of(D, predict(m, D, 'test', chans), 'test')
        cl = chans if chans is not None else list(range(chans_total))
        row = {}
        for j, c in enumerate(cl):
            def tf(x, j=j):
                x = x.clone(); x[..., j] = 0; return x            # z = 0 is the training mean of the channel
            row[CHANNELS[c]] = 100 * (mae_of(D, predict(m, D, 'test', chans, transform=tf), 'test') / base - 1)
        out_c.append(row); rowt = {}
        for b0 in range(0, D.L, lag_block):
            def tf(x, b0=b0):
                x = x.clone(); x[:, b0:b0 + lag_block] = 0; return x
            rowt[f'{(D.L - b0) * D.bin_ms}-{(D.L - b0 - lag_block) * D.bin_ms} ms'] = \
                100 * (mae_of(D, predict(m, D, 'test', chans, transform=tf), 'test') / base - 1)
        out_t.append(rowt)
    return pd.DataFrame(out_c).mean(), pd.DataFrame(out_t).mean()

OCC_C, OCC_T = {}, {}
if remaining_min() > 6:
    for m in [x for x in ('iTransformer', 'TimeXer', PROPOSED) if x in DEEP_MODELS]:
        for ds, D in DATA.items():
            seeds = sorted(k[3] for k in DEEP_PREDS if k[0] == 'main' and k[1] == ds and k[2] == m)[:2]
            if seeds:
                OCC_C[(m, ds)], OCC_T[(m, ds)] = occlusion(m, D, seeds)
    occ_c = pd.DataFrame(OCC_C); occ_t = pd.DataFrame(OCC_T)
    occ_c.to_csv(TABLES / 'occlusion_channels_dMAE_pct.csv'); occ_t.to_csv(TABLES / 'occlusion_lags_dMAE_pct.csv')
    display(occ_c.round(2))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3), layout='constrained')
    mg = occ_c[[c for c in occ_c.columns if c[0] == PROPOSED]]; mg.columns = [c[1] for c in mg.columns]
    mg.plot.barh(ax=axes[0], width=.8); axes[0].set_xlabel('ΔMAE (%) when channel is occluded'); axes[0].invert_yaxis()
    mt = occ_t[[c for c in occ_t.columns if c[0] == PROPOSED]]; mt.columns = [c[1] for c in mt.columns]
    mt.plot(ax=axes[1], marker='o', ms=3); axes[1].set_xticks(range(len(mt)), mt.index, rotation=60, fontsize=5)
    axes[1].set_ylabel('ΔMAE (%)'); axes[1].set_xlabel('occluded lag block (age)')
    savefig(fig, 'fig_occlusion_caf')
checkpoint('analysis cell 29')


RES = DF[DF.study == 'res50']
if len(RES):
    s = summary(RES, 'skill', None)['mean'].unstack(0)
    s50 = s.reindex([m for m in CLASSICAL_MODELS + RES_MODELS if m in s.index])
    s10 = (summary(MAIN, 'skill', None)['mean'].unstack(0)).reindex(s50.index)
    TABLE11 = pd.concat({'10 ms': s10, '50 ms': s50}, axis=1).round(3)
    TABLE11.columns = [f'{b} {a}' for a, b in TABLE11.columns]; TABLE11.index.name = 'Model'
    display(TABLE11)
    to_latex(TABLE11.astype(str), TABLES / 'table11_resolution.tex',
             'Skill vs persistence (horizon-averaged) at 10 ms and 50 ms aggregation (horizons 10-100 ms and 50-500 ms).',
             'tab:res', best_lower={c: -TABLE11[c] for c in TABLE11.columns})
LBK = DF[DF.study == 'lookback']
if len(LBK):
    lb = LBK.groupby(['dataset', 'model', 'L', 'seed']).mae.mean().groupby(['dataset', 'model', 'L']).mean()
    ref = MAIN[MAIN.model.isin(LB_MODELS)].groupby(['dataset', 'model', 'seed']).mae.mean().groupby(['dataset', 'model']).mean()
    lbt = pd.concat([lb.unstack('L'), ref.rename(LOOKBACK)], axis=1)
    lbt.columns = [int(c) for c in lbt.columns]
    lbt = lbt[sorted(lbt.columns)]; lbt.columns = [f'L={c} ({c * BIN_MS} ms)' for c in lbt.columns]
    display(lbt.round(3)); lbt.to_csv(TABLES / 'lookback_sensitivity.csv')
    fig, axes = plt.subplots(1, len(DS), figsize=(7.2, 2.0), layout='constrained'); axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, DS):
        if ds in lbt.index.get_level_values(0):
            t = lbt.loc[ds]
            for m in t.index:
                ax.plot([int(c.split('=')[1].split()[0]) for c in t.columns], t.loc[m].values, marker='o', ms=3,
                        label=m, color='k' if m == PROPOSED else None)
        ax.set_xscale('log', base=2); ax.set_title(ds); ax.set_xlabel('look-back L (bins)')
    axes[0].set_ylabel('test MAE (Mbit/s)'); axes[-1].legend()
    savefig(fig, 'fig_lookback')
checkpoint('analysis cell 30')


fig, axes = plt.subplots(1, len(DS), figsize=(7.2, 2.3), layout='constrained'); axes = np.atleast_1d(axes)
for ax, ds in zip(axes, DS):
    f = FULL[FULL.dataset == ds]
    for j, m in enumerate(KEY_MODELS):
        g = f[f.model == m]
        ax.plot(g.h * BIN_MS, g.skill, marker='o', ms=2, lw=1.6 if m == PROPOSED else .8,
                color='k' if m == PROPOSED else f'C{j % 10}', label=m)
    ax.set_title(ds); ax.set_xlabel('horizon (ms)')
axes[0].set_ylabel('skill vs persistence'); axes[-1].legend(fontsize=5, ncol=2)
savefig(fig, 'fig_skill_vs_horizon')

fig, axes = plt.subplots(len(DS), 1, figsize=(7.2, 1.6 * len(DS)), layout='constrained'); axes = np.atleast_1d(axes)
hh = REPORT_STEPS[1]
for ax, (ds, D) in zip(axes, DATA.items()):
    S = D.splits['test']; n0 = int(np.argmax(S.day == 0)) + 1000; sl = slice(n0, n0 + 300)
    tt = np.arange(300) * BIN_MS / 1000
    ax.plot(tt, S.y[sl, hh - 1], color='#999', lw=.8, label='actual')
    for j, m in enumerate([x for x in ('EWMA', 'PatchTST', 'TimeXer', 'CAF-adaptive', PROPOSED) if x in ORDER]):
        p = np.mean(all_preds(ds, m, 'test'), 0)
        ax.plot(tt, p[sl, hh - 1], lw=1.2 if m == PROPOSED else .7, color='k' if m == PROPOSED else f'C{j}', label=m)
    ax.set_ylabel('Mbit/s'); ax.set_title(f'{ds}, {S.days[0]}, {hh * BIN_MS} ms ahead', fontsize=7)
axes[-1].set_xlabel('time (s)'); axes[0].legend(ncol=5, fontsize=5)
savefig(fig, 'fig_example_forecasts')
checkpoint('analysis cell 31')


from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
COL2 = 7.16
PAL = {'CAF-X': '#D55E00', 'CAF-E1': '#E69F00', PROPOSED: '#CC79A7', 'TimeXer': '#0072B2', 'TimeMixer': '#56B4E9',
       'TimesNet': '#009E73', 'TSMixer': '#7570B3', 'iTransformer': '#999933', 'PatchTST': '#882255', 'EWMA': '#000000',
       'Persistence': '#888888', 'DLinear': '#A6761D', 'N-HiTS': '#66A61E'}
FIG_LOG = []

def pc(m):
    return PAL.get(m, '#AAAAAA')

def run_figure(name, fn):
    t0 = time.time()
    try:
        fn(name); FIG_LOG.append(dict(figure=name, status='ok', s=round(time.time() - t0, 1)))
    except Exception as e:
        plt.close('all'); FIG_LOG.append(dict(figure=name, status=f'{type(e).__name__}: {str(e)[:150]}', s=0))
        print(f'[{name}] skipped: {type(e).__name__}: {e}')

def head_keys(ds):
    t = CAFX_HEAD.get(ds); return t, sorted(k for k in CAFX if k[0] == ds and k[1] == t)

def cafx_test(ds):
    t, ks = head_keys(ds); return np.mean([CAFX[k]['test'] for k in ks], 0) if ks else None

def model_test(ds, m):
    if m == 'CAF-X':
        return cafx_test(ds)
    ps = all_preds(ds, m, 'test')
    return np.mean(ps, 0) if ps else None

def seed_maes(ds, m):
    y = DATA[ds].splits['test'].y
    if m == 'CAF-X':
        t, ks = head_keys(ds); return np.array([np.abs(CAFX[k]['test'] - y).mean() for k in ks])
    return np.array([np.abs(p - y).mean() for p in all_preds(ds, m, 'test')])

def compute_of(ds, m):
    """FLOPs, expected CPU ms, params, energy J per 1M forecasts (CAF-X: cheap path + escalation-rate x target)"""
    if m == 'CAF-X':
        r = CAFX_MAIN[(CAFX_MAIN.dataset == ds) & CAFX_MAIN.headline].iloc[0]; t = r.target; q = r.escalation_rate
        par = sum(EFF.loc[x, 'params'] for x in _parts(t)) + EFF.loc['CAF-E1', 'params'] + 32 * (CAF(len(CHANNELS), LOOKBACK, HORIZON).nrf + 2)
        en = EFF.loc['CAF-E1', 'J_per_M_forecasts'] + q * sum(EFF.loc[x, 'J_per_M_forecasts'] for x in _parts(t))
        return dict(MFLOPs=r.MFLOPs, cpu_ms=r.cpu_ms_expected, params=par, J_per_M=en)
    if m in EFF.index:
        e = EFF.loc[m]; return dict(MFLOPs=e.MFLOPs, cpu_ms=e.cpu1t_ms_b1, params=e.params, J_per_M=e.J_per_M_forecasts)
    return dict(MFLOPs=CLASSICAL_FLOPS.get(m, np.nan) / 1e6, cpu_ms=np.nan, params=0, J_per_M=np.nan)

SB_MODELS = [m for m in ['CAF-X', 'TimeXer', 'TimeMixer', 'TimesNet', 'TSMixer', 'iTransformer', 'PatchTST', 'N-HiTS',
                         'DLinear', PROPOSED, 'CAF-E1', 'EWMA'] if m == 'CAF-X' or m in ORDER]

# ---- multi-metric scoreboard (Table XIII): accuracy (MAE, RMSE, p99 |err|, burst-onset MAE) + compute ----
def _scoreboard():
    rows = []
    for ds in DS:
        D = DATA[ds]; S = D.splits['test']; y = S.y
        c = np.r_[0, np.cumsum(S.R, dtype=np.float64)]; ma8 = (c[S.t] - c[S.t - 8]) / 8
        burst = (y[:, 4] - ma8) > np.quantile(y[:, 4] - ma8, .99)
        for m in SB_MODELS:
            p = model_test(ds, m)
            if p is None:
                continue
            e = np.abs(p - y); sm = seed_maes(ds, m)
            rows.append(dict(dataset=ds, model=m, MAE=e.mean(), MAE_seed_sd=sm.std(ddof=1) if len(sm) > 1 else 0,
                             RMSE=np.sqrt(((p - y) ** 2).mean()), p99_abs_err=np.percentile(e.mean(1), 99),
                             burst_MAE=e[burst].mean(), **compute_of(ds, m)))
    return pd.DataFrame(rows)

SCORE = _scoreboard(); SCORE.to_csv(TABLES / 'table13_multimetric_scoreboard.csv', index=False)
LOWER = ['MAE', 'RMSE', 'p99_abs_err', 'burst_MAE', 'MFLOPs', 'cpu_ms', 'params', 'J_per_M']
SCORE_RANK = SCORE.groupby('dataset')[LOWER].rank(method='min')
SCORE_RANK[['dataset', 'model']] = SCORE[['dataset', 'model']]
WINS = SCORE_RANK.assign(**{c: SCORE_RANK[c] == 1 for c in LOWER}).groupby('model')[LOWER].sum()
display(SCORE.round(3)); display(WINS)
t13 = SCORE.pivot(index='model', columns='dataset', values='MAE').reindex(SB_MODELS).round(2)
cmp_ = SCORE.groupby('model')[['MFLOPs', 'cpu_ms', 'params', 'J_per_M']].mean().reindex(SB_MODELS)
t13 = t13.join(cmp_.round(3)); t13.index.name = 'Model'
to_latex(t13.astype(str), TABLES / 'table13_multimetric_scoreboard.tex',
         'Multi-metric scoreboard: test MAE per dataset (seed-ensemble forecasts) and per-forecast compute '
         '(MFLOPs, expected 1-core CPU ms, parameters, T4 energy in J per 10^6 forecasts). CAF-X compute includes the '
         'always-on cheap path and the measured escalation rate.', 'tab:score',
         best_lower={c: pd.to_numeric(t13[c], errors='coerce') for c in t13.columns})

def figS1(name):           # ---- scoreboard heatmap: rank of every model on every metric and dataset ----
    cols = [(ds, m) for ds in DS for m in ('MAE', 'RMSE', 'p99_abs_err', 'burst_MAE')] + \
           [('compute', m) for m in ('MFLOPs', 'cpu_ms', 'params', 'J_per_M')]
    M = pd.DataFrame(index=SB_MODELS, columns=pd.MultiIndex.from_tuples(cols), dtype=float)
    for (g, met) in cols:
        if g == 'compute':
            v = SCORE.groupby('model')[met].mean()
        else:
            v = SCORE[SCORE.dataset == g].set_index('model')[met]
        M[(g, met)] = v.reindex(SB_MODELS).rank(method='min')
    fig, ax = plt.subplots(figsize=(COL2, 3.2), layout='constrained')
    im = ax.imshow(M.values, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=len(SB_MODELS))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=5, fontweight='bold' if v == 1 else 'normal')
    ax.set_xticks(range(len(cols)), [m.replace('_abs_err', '|e|').replace('_', ' ') for _, m in cols], rotation=60, fontsize=5)
    ax.set_yticks(range(len(SB_MODELS)), SB_MODELS, fontsize=5.5); ax.grid(False)
    for k, g in enumerate(DS + ['compute']):
        ax.text(4 * k + 1.5, -1.0, g, ha='center', fontsize=6, fontweight='bold')
    ax.set_ylim(len(SB_MODELS) - .5, -1.6)
    fig.colorbar(im, ax=ax, shrink=.8, label='rank (1 = best)')
    savefig(fig, name)

def figS2(name):           # ---- radar: normalised multi-metric profile of CAF-X vs strongest SOTA models ----
    mets = ['MAE', 'RMSE', 'p99_abs_err', 'burst_MAE', 'MFLOPs', 'cpu_ms', 'J_per_M']
    avg = SCORE.groupby('model')[mets].mean()
    rel = SCORE.copy()
    for met in ['MAE', 'RMSE', 'p99_abs_err', 'burst_MAE']:
        rel[met] = SCORE[met] / SCORE.groupby('dataset')[met].transform('min')
    rel = rel.groupby('model')[mets].mean()
    for met in ['MFLOPs', 'cpu_ms', 'J_per_M']:
        rel[met] = np.log10(avg[met].clip(lower=1e-6) / avg[met].clip(lower=1e-6).min()) + 1
    score = 1 / rel                                                           # 1 = best on that axis
    models = [m for m in ['CAF-X', 'TimeXer', 'TimeMixer', 'TimesNet', 'TSMixer'] if m in score.index]
    ang = np.linspace(0, 2 * np.pi, len(mets), endpoint=False).tolist(); ang += ang[:1]
    fig = plt.figure(figsize=(3.5, 3.3)); ax = fig.add_subplot(111, polar=True)
    for m in models:
        v = score.loc[m, mets].tolist(); v += v[:1]
        ax.plot(ang, v, color=pc(m), lw=1.6 if m == 'CAF-X' else .9, label=m); ax.fill(ang, v, color=pc(m), alpha=.08)
    ax.set_xticks(ang[:-1], ['MAE', 'RMSE', 'p99 |e|', 'burst MAE', 'FLOPs', 'CPU ms', 'energy'], fontsize=6)
    ax.set_yticklabels([]); ax.legend(fontsize=5.5, loc='upper right', bbox_to_anchor=(1.3, 1.12))
    ax.set_title('accuracy (relative) and compute (log-relative); outer = better', fontsize=6.5, pad=14)
    savefig(fig, name)

def figS3(name):           # ---- budget curves: learned vs volatility vs random vs oracle router ----
    fig, axes = plt.subplots(1, len(DS), figsize=(COL2, 2.3), layout='constrained')
    for ax, ds in zip(np.atleast_1d(axes), DS):
        t = CAFX_HEAD.get(ds); g = CAFX_SWEEP[(CAFX_SWEEP.dataset == ds) & (CAFX_SWEEP.target == t)]
        tgt_mae = np.mean([CAFX[k]['target_test_mae'] for k in head_keys(ds)[1]])
        for rn, ls in (('learned', '-'), ('volatility', '--'), ('random', ':'), ('oracle (test labels)', '-.')):
            gg = g[g.router == rn]
            if not len(gg):
                continue
            mode = gg.groupby('mode').val_mae.mean().idxmin(); c = gg[gg['mode'] == mode].groupby('budget').test_mae.mean()
            ax.plot(100 * c.index, 100 * (c.values / tgt_mae - 1), ls, marker='o', ms=2, lw=1.2 if rn == 'learned' else .8,
                    color={'learned': pc('CAF-X'), 'volatility': '#0072B2', 'random': '#888888'}.get(rn, '#009E73'),
                    label=rn)
        ax.axhline(0, color='k', lw=.6); ax.set_title(f'{ds}\nwraps {t}', fontsize=6.3)
        ax.set_xlabel('windows escalated (%)')
    np.atleast_1d(axes)[0].set_ylabel(f'MAE vs always-on target (%)'); np.atleast_1d(axes)[-1].legend(fontsize=5)
    savefig(fig, name)

def _router_xai(ds):
    """permutation importance of router inputs on the validation day: drop in Spearman rank correlation between router
    score and the realised escalation gain when one feature is shuffled (3 repeats, first 2 seeds)"""
    D = DATA[ds]; t, ks = head_keys(ds); out = []
    C = len(CHANNELS)
    names = [f'E1−E0 h{h + 1}' for h in range(HORIZON)] + ['σ8/σ', 'σ32/σ', '|r−MA8|/σ', '|Δr|/σ'] + \
            [f'hdr:{c}' for c in CHANNELS[1:]]
    for k in ks[:2]:
        seed = k[2]; key = f'cafx_{slug(ds)}_{slug(t)}_s{seed}'
        ck = torch.load(CKPT / f'{key}__router.pt', map_location=DEVICE, weights_only=False)
        c = _caf_cheap(ds, seed); deep = _deep_preds(ds, t, seed); y = D.splits['val'].y
        g = (np.abs(c['val']['e1'] - y) - np.abs(deep['val'] - y)).mean(1)
        X = ((c['val']['f'] - ck['mu']) / ck['sd']).astype(np.float32)
        net = nn.Sequential(nn.Linear(X.shape[1], 32), nn.GELU(), nn.Linear(32, 1)).to(DEVICE); net.load_state_dict(ck['router'])
        net.eval(); rng = np.random.default_rng(seed)
        sc_ = lambda Z: net(torch.tensor(Z, device=DEVICE)).squeeze(-1).detach().cpu().numpy()
        base = stats.spearmanr(sc_(X), g).correlation; imp = []
        for j in range(X.shape[1]):
            d = []
            for _ in range(3):
                Z = X.copy(); Z[:, j] = Z[rng.permutation(len(Z)), j]; d.append(base - stats.spearmanr(sc_(Z), g).correlation)
            imp.append(np.mean(d))
        out.append(imp)
    return pd.Series(np.mean(out, 0), index=names[:len(out[0])]), float(base)

ROUTER_XAI = {}
for ds in DS:
    try:
        if CAFX_HEAD.get(ds):
            ROUTER_XAI[ds] = _router_xai(ds)
    except Exception as e:
        print(f'router XAI {ds} skipped: {e}')
if ROUTER_XAI:
    pd.DataFrame({ds: v[0] for ds, v in ROUTER_XAI.items()}).to_csv(TABLES / 'xai_router_permutation_importance.csv')

def figS4(name):           # ---- XAI 1: what makes the router escalate? ----
    fig, axes = plt.subplots(1, len(ROUTER_XAI), figsize=(COL2, 2.6), layout='constrained', sharey=True)
    for ax, (ds, (imp, rho)) in zip(np.atleast_1d(axes), ROUTER_XAI.items()):
        top = imp.sort_values(ascending=True).tail(10)
        ax.barh(range(len(top)), top.values, color=[pc('CAF-X') if 'σ' in n or 'Δ' in n or 'MA8' in n else
                                                    '#0072B2' if n.startswith('E1') else '#009E73' for n in top.index])
        ax.set_yticks(range(len(top)), top.index, fontsize=5.3); ax.set_xlabel('Δ Spearman ρ when shuffled')
        ax.set_title(f'{ds} (router ρ={rho:.2f})', fontsize=6.3)
    fig.legend(handles=[Patch(color=pc('CAF-X'), label='volatility / change'), Patch(color='#0072B2', label='E1 residual'),
                        Patch(color='#009E73', label='packet-header channels')], loc='outside upper center', ncol=3,
               frameon=False, fontsize=5.5)
    savefig(fig, name)

def figS5(name):           # ---- XAI 2: escalation anatomy by load and volatility decile ----
    fig, axes = plt.subplots(2, len(DS), figsize=(COL2, 3.6), layout='constrained')
    for j, ds in enumerate(DS):
        D = DATA[ds]; S = D.splits['test']; y = S.y; t, ks = head_keys(ds)
        esc = np.mean([np.load(PREDS / f'cafx_{slug(ds)}_{slug(t)}_s{k[2]}.npz')['esc_test'] for k in ks], 0)
        c = _caf_cheap(ds, ks[0][2])['test']; vol = c['f'][:, HORIZON]
        deep = np.mean([_deep_preds(ds, t, k[2])['test'] for k in ks], 0)
        gain = (np.abs(c['e1'] - y) - np.abs(deep - y)).mean(1)
        for i, (v, lab) in enumerate(((S.last, 'current load decile'), (vol, '80 ms volatility decile'))):
            q = np.digitize(v, np.quantile(v, np.linspace(.1, .9, 9))); ax = axes[i, j]
            ax.bar(range(10), [100 * esc[q == d].mean() for d in range(10)], color=pc('CAF-X'), alpha=.75, label='escalated %')
            a2 = ax.twinx(); a2.plot(range(10), [gain[q == d].mean() for d in range(10)], 'o-', ms=2.5, color='#0072B2',
                                     label='deep gain (Mbit/s)')
            a2.axhline(0, color='#0072B2', lw=.4, ls=':'); a2.tick_params(labelsize=5)
            ax.set_xlabel(lab, fontsize=5.8); ax.set_xticks(range(10), range(1, 11), fontsize=5)
            if j == 0:
                ax.set_ylabel('escalated windows (%)')
            if i == 0:
                ax.set_title(f'{ds} → {t}', fontsize=6.3)
    savefig(fig, name)

def figS6(name):           # ---- case study: burst onset, escalation decisions, forecasts ----
    fig, axes = plt.subplots(len(DS), 1, figsize=(COL2, 1.8 * len(DS)), layout='constrained')
    h = 5
    for ax, ds in zip(np.atleast_1d(axes), DS):
        D = DATA[ds]; S = D.splits['test']; t, ks = head_keys(ds)
        c = np.r_[0, np.cumsum(S.R, dtype=np.float64)]; ma8 = (c[S.t] - c[S.t - 8]) / 8
        sc = S.y[:, h - 1] - ma8; i0 = None
        for i in np.argsort(-sc):
            if 200 <= i < S.n - 200 and S.day[i - 200] == S.day[i + 200]:
                i0 = int(i); break
        w = np.arange(i0 - 150, i0 + 151); tt = (S.t[w] + h - 1 - S.t[i0]) * BIN_MS
        esc = np.load(PREDS / f'cafx_{slug(ds)}_{slug(t)}_s{ks[0][2]}.npz')['esc_test'][w]
        ax.plot((S.t[w] - S.t[i0]) * BIN_MS, S.R[S.t[w]], color='#BBBBBB', lw=1.6, label='measured')
        for m in ['CAF-E1', BEST_BASELINE[ds], 'CAF-X']:
            p = model_test(ds, m)
            if p is not None:
                ax.plot(tt, p[w, h - 1], color=pc(m), lw=1.4 if m == 'CAF-X' else .8, label=m)
        yl = ax.get_ylim()
        ax.fill_between(tt, yl[0], yl[1], where=esc, color=pc('CAF-X'), alpha=.10, lw=0, label='escalated (seed 1)')
        ax.set_ylim(yl); ax.axvline(0, color='k', lw=.5, ls=':')
        ax.set_title(f'{ds}: largest burst onset, {h * BIN_MS} ms-ahead forecasts; CAF-X escalates to {t} only in shaded windows',
                     fontsize=6.2)
        ax.set_ylabel('Mbit/s'); ax.legend(fontsize=4.8, ncol=5, loc='upper left')
    np.atleast_1d(axes)[-1].set_xlabel('time relative to burst onset (ms)')
    savefig(fig, name)

def figS7(name):           # ---- per-horizon MAE with 95% CI over seeds ----
    fig, axes = plt.subplots(1, len(DS), figsize=(COL2, 2.3), layout='constrained')
    hx = np.arange(1, HORIZON + 1) * BIN_MS
    for ax, ds in zip(np.atleast_1d(axes), DS):
        y = DATA[ds].splits['test'].y; ref = None
        for m in ['CAF-X', 'TimeXer', 'TimeMixer', 'TimesNet', 'CAF-E1', 'EWMA']:
            if m == 'CAF-X':
                t, ks = head_keys(ds); arr = np.array([np.abs(CAFX[k]['test'] - y).mean(0) for k in ks])
            elif m in ORDER:
                arr = np.array([np.abs(p - y).mean(0) for p in all_preds(ds, m, 'test')])
            else:
                continue
            if ref is None:
                ref = arr.mean(0)
            mu = 100 * (arr / ref - 1); ci = stats.t.ppf(.975, max(len(arr) - 1, 1)) * mu.std(0, ddof=1) / np.sqrt(len(arr)) \
                if len(arr) > 1 else np.zeros(HORIZON)
            ax.plot(hx, mu.mean(0), color=pc(m), lw=1.4 if m == 'CAF-X' else .8, marker='o' if m == 'CAF-X' else None, ms=2, label=m)
            ax.fill_between(hx, mu.mean(0) - ci, mu.mean(0) + ci, color=pc(m), alpha=.15, lw=0)
        ax.axhline(0, color='k', lw=.5); ax.set_title(ds); ax.set_xlabel('horizon (ms)')
    np.atleast_1d(axes)[0].set_ylabel('MAE relative to CAF-X (%)'); np.atleast_1d(axes)[-1].legend(fontsize=5)
    savefig(fig, name)

def figS8(name):           # ---- paired seeds: CAF-X vs the model it wraps (every seed) ----
    fig, axes = plt.subplots(1, len(DS), figsize=(COL2, 2.2), layout='constrained')
    for ax, ds in zip(np.atleast_1d(axes), DS):
        y = DATA[ds].splits['test'].y
        for t in CAFX_TARGETS[ds]:
            ks = sorted(k for k in CAFX if k[0] == ds and k[1] == t)
            a = [CAFX[k]['target_test_mae'] for k in ks]; b = [np.abs(CAFX[k]['test'] - y).mean() for k in ks]
            ax.scatter(a, b, s=12, label=t[:22], alpha=.85)
        lim = ax.get_xlim(); ax.plot(lim, lim, color='k', lw=.6); ax.set_xlim(lim)
        ax.set_xlabel('wrapped model alone: MAE'); ax.set_title(ds, fontsize=6.5)
    np.atleast_1d(axes)[0].set_ylabel('CAF-X: MAE (below diagonal = better)'); np.atleast_1d(axes)[-1].legend(fontsize=4.5)
    savefig(fig, name)

def figS9(name):           # ---- deployability: CPU latency vs links per core and energy ----
    sub = SCORE.groupby('model')[['MAE', 'cpu_ms', 'J_per_M', 'MFLOPs']].mean()
    sub['rel'] = (SCORE.assign(r=SCORE.MAE / SCORE.groupby('dataset').MAE.transform('min')).groupby('model').r.mean() - 1) * 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 2.5), layout='constrained')
    for m, r in sub.iterrows():
        a1.scatter(r.cpu_ms, r.rel, s=25, color=pc(m), edgecolor='k', lw=.3); a1.annotate(m, (r.cpu_ms, r.rel), fontsize=5,
                                                                                         xytext=(3, 2), textcoords='offset points')
        a2.scatter(r.J_per_M, r.rel, s=25, color=pc(m), edgecolor='k', lw=.3); a2.annotate(m, (r.J_per_M, r.rel), fontsize=5,
                                                                                          xytext=(3, 2), textcoords='offset points')
    a1.axvline(BIN_MS, color='#B22222', lw=.7); a1.text(BIN_MS, a1.get_ylim()[1], ' 10 ms deadline', color='#B22222', fontsize=5, va='top')
    for a, xl in ((a1, '1-core CPU ms per forecast (expected)'), (a2, 'T4 energy (J per 10⁶ forecasts)')):
        a.set_xscale('log'); a.set_xlabel(xl)
    a1.set_ylabel('mean MAE above best (%)')
    savefig(fig, name)

for nm, fn in (('figS1_scoreboard_ranks', figS1), ('figS2_multimetric_radar', figS2), ('figS3_router_budget_curves', figS3),
               ('figS4_xai_router_importance', figS4), ('figS5_xai_escalation_anatomy', figS5),
               ('figS6_case_burst_escalation', figS6), ('figS7_horizon_profile', figS7), ('figS8_paired_seeds', figS8),
               ('figS9_deployability', figS9)):
    run_figure(nm, fn)
FIG_LOG = pd.DataFrame(FIG_LOG); FIG_LOG.to_csv(TABLES / 'figure_log.csv', index=False); display(FIG_LOG)
checkpoint('CAF-X figures', force=True)


from scipy.stats import spearmanr
TH_SEED = {ds: sorted(k[3] for k in DEEP_PREDS if k[0] == 'main' and k[1] == ds and k[2] == PROPOSED)[0] for ds in DS}
H_LOG = []

def _hrun(name, fn):
    try:
        fn(name); H_LOG.append((name, 'ok'))
    except Exception as e:
        plt.close('all'); H_LOG.append((name, f'{type(e).__name__}: {str(e)[:150]}')); print(f'[{name}] skipped: {e}')

def _caf(ds, seed=None):
    return load_model('main', DATA[ds], PROPOSED, MODEL_SPECS[PROPOSED], seed or TH_SEED[ds]).float().eval()

def _xw(ds, chans, n=1024):
    S = DATA[ds].splits['test']; idx = torch.tensor(np.linspace(0, S.n - 1, min(n, S.n)).astype(int), device=DEVICE)
    x, _ = S.batch(idx, chans); return x.float()

class _Exit(nn.Module):
    def __init__(self, m, k):
        super().__init__(); self.m, self.k = m, k
    def forward(self, x):
        return self.m.forward_exits(x)[self.k]

def _cfg_esc(ds, tgt, seed, budget, mode):
    r = next(r for r in CAFX[(ds, tgt, seed)]['rows'] if r['router'] == 'learned' and r['budget'] == budget and r['mode'] == mode)
    return np.load(PREDS / f'cafx_{slug(ds)}_{slug(tgt)}_s{seed}.npz')['score_test'] > r['tau']

# ---------- Theorem 1 ----------
def _t1():
    rows = []
    for ds in DS:
        D = DATA[ds]; mu0, sd0 = float(D.mu[0]), float(D.sd[0])
        nets = [(f'CAF-E{k}', _Exit(_caf(ds), k), MODEL_SPECS[PROPOSED][1]) for k in range(3)]
        for m in [x for x in DEEP_MODELS if x != PROPOSED]:
            s = sorted(k[3] for k in DEEP_PREDS if k[0] == 'main' and k[1] == ds and k[2] == m)[0]
            nets.append((m, load_model('main', D, m, MODEL_SPECS[m], s).float().eval(), MODEL_SPECS[m][1]))
        for name, net, chans in nets:
            x = _xw(ds, chans)
            with torch.no_grad():
                y1 = net(x).float() * sd0 + mu0
                for a in (0.25, 0.5, 2.0, 4.0):
                    xt = x.clone(); xt[..., 0] = a * x[..., 0] + (a - 1) * mu0 / sd0
                    ya = net(xt).float() * sd0 + mu0
                    rows.append(dict(dataset=ds, model=name, a=a, rel_err=float((ya - a * y1).abs().mean() / (a * y1.abs().mean()))))
    return pd.DataFrame(rows)
TH1 = cache_obj('theory_t1_exits', _t1); TH1S = TH1.groupby('model').rel_err.max().sort_values()
TH1S.to_csv(TABLES / 'theorem1_exit_equivariance.csv'); display(TH1S.round(6).to_frame('max rel. error'))

# ---------- Proposition 2: routing invariance (CAF-X headline router, first seed) ----------
def _p2():
    rows = []
    for ds in DS:
        t, ks = head_keys(ds)
        if not ks:
            continue
        seed = ks[0][2]; D = DATA[ds]; key = f'cafx_{slug(ds)}_{slug(t)}_s{seed}'
        ck = torch.load(CKPT / f'{key}__router.pt', map_location=DEVICE, weights_only=False)
        net = nn.Sequential(nn.Linear(len(ck['mu']), 32), nn.GELU(), nn.Linear(32, 1)).to(DEVICE); net.load_state_dict(ck['router']); net.eval()
        m = _caf(ds, seed)
        with torch.no_grad():
            m.rf_mu.zero_(); m.rf_sd.fill_(1.0)
        mu, sd = torch.tensor(ck['mu'], device=DEVICE), torch.tensor(ck['sd'], device=DEVICE)
        x = _xw(ds, MODEL_SPECS[PROPOSED][1], 8192); mu0, sd0 = float(D.mu[0]), float(D.sd[0])

        def esc(xx):
            with torch.no_grad():
                r, ma, e0, e1 = m._cheap(xx); f = m.router_features(xx, r, e0, e1).float()
                return (net((f - mu) / sd).squeeze(-1) > ck['tau']).cpu().numpy()
        e_ref = esc(x)
        for a in (0.25, 0.5, 2.0, 4.0):
            xt = x.clone(); xt[..., 0] = a * x[..., 0] + (a - 1) * mu0 / sd0
            rows.append(dict(dataset=ds, a=a, agreement=float((esc(xt) == e_ref).mean()), rate_ref=float(e_ref.mean())))
    return pd.DataFrame(rows)
TH2 = cache_obj('theory_p2_router_invariance', _p2); display(TH2.round(4)); TH2.to_csv(TABLES / 'prop2_router_invariance.csv', index=False)

# ---------- Theorem 3 / 4 / Prop. 5 data from the CAF-X sweep ----------
SW = CAFX_SWEEP.copy()
def _head_sweep(ds):
    t = CAFX_HEAD.get(ds); g = SW[(SW.dataset == ds) & (SW.target == t)]
    mode = g[g.router == 'learned'].groupby('mode').val_mae.mean().idxmin()
    return t, g[g['mode'] == mode]

def _blend_days():
    rows = []
    for ds in DS:
        if ds not in EFF_SEL:
            continue
        s = EFF_SEL[ds]; a = float(s['mode'].split('=')[1]); y = DATA[ds].splits['test'].y; day = DATA[ds].splits['test'].day
        for seed in SEEDS[:3]:
            e1 = _caf_cheap(ds, seed)['test']['e1']; dp = _deep_preds(ds, s['target'], seed)['test']
            for a_ in (0.25, 0.5, 0.75):
                bl = a_ * dp + (1 - a_) * e1
                for d in np.unique(day):
                    k = day == d
                    rows.append(dict(dataset=ds, seed=seed, a=a_, day=d, blend=np.abs(bl[k] - y[k]).mean(),
                                     bound=a_ * np.abs(dp[k] - y[k]).mean() + (1 - a_) * np.abs(e1[k] - y[k]).mean(),
                                     best_end=min(np.abs(dp[k] - y[k]).mean(), np.abs(e1[k] - y[k]).mean())))
    return pd.DataFrame(rows)
BL = cache_obj('theory_p5_blend', _blend_days); BL.to_csv(TABLES / 'prop5_blend_bound.csv', index=False)

ABLF = []
_a = DF[DF.study == 'ablation'].groupby(['dataset', 'model', 'seed']).mae.mean()
_f = MAIN[MAIN.model == PROPOSED].groupby(['dataset', 'seed']).mae.mean()
for (ds, m, s), v in _a.items():
    if (ds, s) in _f:
        ABLF.append(dict(dataset=ds, variant=m, seed=s, delta_pct=100 * (v / _f[(ds, s)] - 1)))
ABLF = pd.DataFrame(ABLF); ABLF.to_csv(TABLES / 'ablation_seed_level.csv', index=False)

# ======================= figures =======================
def figH01(name):
    fig, ax = plt.subplots(figsize=(COL2, 2.4), layout='constrained')
    v = TH1S.clip(lower=1e-7)
    ax.bar(range(len(v)), v.values, color=[pc('CAF-X') if m.startswith('CAF') else pc(m) for m in v.index])
    ax.set_yscale('log'); ax.set_xticks(range(len(v)), v.index, rotation=40, fontsize=5.5); ax.axhline(1e-3, color='#B22222', ls=':', lw=.7)
    ax.set_ylabel('max relative error of f(a·r)=a·f(r)\n(a ∈ {¼,½,2,4}, 3 datasets)')
    ax.set_title('Theorem 1: every CAF exit is load-scale equivariant by construction', fontsize=6.5)
    savefig(fig, name)

def figH02(name):
    fig, ax = plt.subplots(figsize=(3.5, 2.3), layout='constrained')
    for ds, g in TH2.groupby('dataset'):
        ax.plot(g.a, 100 * g.agreement, '-o', ms=3, label=f'{ds} (escalation {100 * g.rate_ref.iloc[0]:.0f}%)')
    ax.set_xscale('log', base=2); ax.set_ylim(95, 100.2); ax.set_xlabel('load scale a'); ax.set_ylabel('identical escalation decisions (%)')
    ax.legend(fontsize=5); ax.set_title('Proposition 2: routing is load-invariant', fontsize=6.5)
    savefig(fig, name)

def figH03(name):
    fig, axes = plt.subplots(1, len(DS), figsize=(COL2, 2.3), layout='constrained')
    for ax, ds in zip(np.atleast_1d(axes), DS):
        t, g = _head_sweep(ds); g = g[g.router == 'learned'].groupby('budget')[['val_rate', 'test_rate']].agg(['mean', 'min', 'max'])
        b = g.index.values; nv = DATA[ds].splits['val'].n / LAG; nt = DATA[ds].splits['test'].n / LAG
        eps = np.sqrt(np.log(2 / .05) / (2 * nv)) + np.sqrt(np.log(2 / .05) / (2 * nt))
        ax.fill_between(b, np.clip(b - eps, 0, 1), np.clip(b + eps, 0, 1), color='#009E73', alpha=.15, label='DKW 95% band (n_eff)')
        ax.plot(b, b, 'k--', lw=.6); ax.errorbar(b, g[('test_rate', 'mean')], yerr=[g[('test_rate', 'mean')] - g[('test_rate', 'min')],
                                                                                 g[('test_rate', 'max')] - g[('test_rate', 'mean')]],
                                                 fmt='o', ms=3, color=pc('CAF-X'), capsize=1.5, label='realised test rate (seeds)')
        ax.set_title(f'{ds}', fontsize=6.5); ax.set_xlabel('calibrated budget b')
    np.atleast_1d(axes)[0].set_ylabel('test escalation rate'); np.atleast_1d(axes)[-1].legend(fontsize=5)
    fig.suptitle('Theorem 3: validation-calibrated budgets transfer to unseen test days', fontsize=6.5)
    savefig(fig, name)

def figH04(name):
    fig, axes = plt.subplots(1, len(DS), figsize=(COL2, 2.3), layout='constrained', sharey=True)
    for ax, ds in zip(np.atleast_1d(axes), DS):
        t, g = _head_sweep(ds); e1 = np.mean([CAFX[k]['e1_test_mae'] for k in CAFX if k[0] == ds and k[1] == t])
        c = g.groupby(['router', 'budget']).test_mae.mean().unstack(0)
        orc = c['oracle (test labels)']
        for rn, col in (('learned', pc('CAF-X')), ('volatility', '#0072B2'), ('random', '#888888')):
            if rn in c:
                ax.plot(100 * c.index, 100 * (e1 - c[rn]) / (e1 - orc).replace(0, np.nan), '-o', ms=2.5, color=col, label=rn)
        ax.axhline(0, color='k', lw=.5); ax.set_title(f'{ds} → {t}', fontsize=6); ax.set_xlabel('budget (%)')
    np.atleast_1d(axes)[0].set_ylabel('oracle headroom captured (%)'); np.atleast_1d(axes)[-1].legend(fontsize=5)
    fig.suptitle('Theorem 4: fraction of the optimal (oracle) gain captured at each budget', fontsize=6.5)
    savefig(fig, name)

def figH05(name):
    fig, ax = plt.subplots(figsize=(3.5, 2.6), layout='constrained')
    for (ds, a), g in BL.groupby(['dataset', 'a']):
        ax.scatter(g.bound, g.blend, s=8, alpha=.7, label=f'{ds}, a={a}' if a == .5 else None)
    lim = [BL[['bound', 'blend']].min().min(), BL[['bound', 'blend']].max().max()]
    ax.plot(lim, lim, 'k--', lw=.6); ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('convex bound a·MAE(D)+(1−a)·MAE(E1)'); ax.set_ylabel('MAE of blend')
    frac = (BL.blend < BL.best_end).mean()
    ax.set_title(f'Prop. 5: blend ≤ bound on {100 * (BL.blend <= BL.bound + 1e-9).mean():.0f}% of day×seed×a; '
                 f'better than both ends on {100 * frac:.0f}%', fontsize=5.5)
    ax.legend(fontsize=4.8)
    savefig(fig, name)

def figH06(name):          # glass-box waterfall of a CAF-X decision
    ds = DS[-1]; D = DATA[ds]; S = D.splits['test']; m = _caf(ds); mu0, sd0 = float(D.mu[0]), float(D.sd[0])
    t, ks = head_keys(ds); z = np.load(PREDS / f'cafx_{slug(ds)}_{slug(t)}_s{ks[0][2]}.npz'); sc_t = z['score_test']
    tau = CAFX[ks[0]]['op']['tau']; i0 = int(np.argmax(sc_t)); x, _ = S.batch(torch.tensor([i0], device=DEVICE), MODEL_SPECS[PROPOSED][1])
    with torch.no_grad():
        r, ma, e0, e1 = m._cheap(x.float()); cand = m.anchors(r, ma)[0].cpu().numpy() * sd0 + mu0; W = m.anchor_weights().cpu().numpy()
    e0v, e1v = e0[0].cpu().numpy() * sd0 + mu0, e1[0].cpu().numpy() * sd0 + mu0
    final = z['test'][i0]; h = 4
    labels = ['last'] + [f'MA{s}' for s in m.scales] + ['mean']
    parts = list(W[h] * cand) + [e1v[h] - e0v[h], final[h] - e1v[h]]
    names = [f'{l}\nw={W[h][i]:.2f}' for i, l in enumerate(labels)] + ['E1 linear\ncorrection', f'escalated\n(score {sc_t[i0]:.2f}>τ {tau:.2f})']
    cum = np.r_[0, np.cumsum(parts)]
    fig, ax = plt.subplots(figsize=(COL2, 2.4), layout='constrained')
    for k, v in enumerate(parts):
        ax.bar(k, v, bottom=cum[k], color='#0072B2' if k < len(labels) else ('#009E73' if v >= 0 else '#D55E00'), width=.7)
    ax.bar(len(parts), final[h], color=pc('CAF-X'), width=.7); ax.axhline(S.y[i0, h], color='k', ls='--', lw=.8)
    ax.set_xticks(range(len(parts) + 1), names + ['CAF-X\nforecast'], fontsize=4.8); ax.set_ylabel('Mbit/s')
    ax.set_title(f'Glass-box CAF-X forecast ({ds}, {(h + 1) * BIN_MS} ms ahead, highest-score window; dashed = actual)', fontsize=6.5)
    savefig(fig, name)

def figH07(name):          # E1 linear map: fully interpretable weights
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.4), layout='constrained', gridspec_kw=dict(width_ratios=[2.2, 1]))
    Ws = []
    for ds in DS:
        for s in sorted(k[3] for k in DEEP_PREDS if k[0] == 'main' and k[1] == ds and k[2] == PROPOSED):
            Ws.append(_caf(ds, s).lin.weight.detach().cpu().numpy())
    W = np.mean(Ws, 0); L = LOOKBACK
    lim = np.abs(W[:, :L]).max()
    im = axes[0].imshow(W[:, :L], aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim)
    axes[0].set_xticks([0, L // 2, L - 1], [f'{-L * BIN_MS}', f'{-L // 2 * BIN_MS}', f'{-BIN_MS}'])
    axes[0].set_yticks(range(0, HORIZON, 3), [f'{(h + 1) * BIN_MS}' for h in range(0, HORIZON, 3)])
    axes[0].set_xlabel('lag (ms) of (r−ℓ)/σ'); axes[0].set_ylabel('horizon (ms)'); axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], shrink=.8)
    hw = np.abs(W[:, L:]).mean(0); axes[1].barh(range(len(hw)), hw, color='#009E73')
    axes[1].set_yticks(range(len(hw)), CHANNELS[1:], fontsize=5); axes[1].set_xlabel('mean |weight| (header channels)')
    fig.suptitle('Glass box: the cheap exit E1 is a single linear map (mean over datasets × seeds)', fontsize=6.5)
    savefig(fig, name)

def figH08(name):          # router calibration
    fig, axes = plt.subplots(1, len(DS), figsize=(COL2, 2.2), layout='constrained')
    for ax, ds in zip(np.atleast_1d(axes), DS):
        t, ks = head_keys(ds); y = DATA[ds].splits['test'].y; xs, ys = [], []
        for k in ks[:3]:
            z = np.load(PREDS / f'cafx_{slug(ds)}_{slug(t)}_s{k[2]}.npz'); c = _caf_cheap(ds, k[2])['test']
            gain = (np.abs(c['e1'] - y) - np.abs(_deep_preds(ds, t, k[2])['test'] - y)).mean(1)
            q = np.digitize(z['score_test'], np.quantile(z['score_test'], np.linspace(.1, .9, 9)))
            xs.append([z['score_test'][q == d].mean() for d in range(10)]); ys.append([gain[q == d].mean() for d in range(10)])
            rho = spearmanr(z['score_test'], gain).correlation
        ax.plot(range(1, 11), np.mean(ys, 0), '-o', ms=3, color=pc('CAF-X')); ax.axhline(0, color='k', lw=.5)
        ax.set_xlabel('router score decile'); ax.set_title(f'{ds} (ρ={rho:.2f})', fontsize=6.3)
    np.atleast_1d(axes)[0].set_ylabel('realised gain of escalating (Mbit/s)')
    fig.suptitle('Router calibration: higher score → larger realised gain', fontsize=6.5)
    savefig(fig, name)

def figH09(name):          # escalation timeline with cumulative compute
    ds = DS[0]; S = DATA[ds].splits['test']
    if ds in EFF_SEL:
        s = EFF_SEL[ds]; pred, _ = _cfg_preds(ds, s['target'], SEEDS[0], s['budget'], s['mode'])
        esc = _cfg_esc(ds, s['target'], SEEDS[0], s['budget'], s['mode']); tgt = s['target']
    else:
        return
    w = np.arange(S.n // 3, S.n // 3 + 3000); tt = (S.t[w] - S.t[w[0]]) * BIN_MS / 1000
    fig, axes = plt.subplots(3, 1, figsize=(COL2, 3.0), layout='constrained', sharex=True, gridspec_kw=dict(height_ratios=[2, .4, 1]))
    axes[0].plot(tt, S.R[S.t[w]], color='#BBBBBB', lw=.6); axes[0].plot(tt, pred[w, 4], color=pc('CAF-X'), lw=.6)
    axes[0].set_ylabel('Mbit/s')
    axes[1].eventplot(tt[esc[w]], colors=pc('CAF-X'), lineoffsets=.5, linelengths=.9); axes[1].set_yticks([]); axes[1].set_ylabel('esc.', fontsize=5)
    cum_x = np.cumsum(CAF_FLOPS['cheap'] + esc[w] * FLOPS[tgt]) / 1e9
    for m, col in ((BEST_BASELINE[ds], pc(BEST_BASELINE[ds])), (tgt, '#888888')):
        if m in FLOPS:
            axes[2].plot(tt, np.arange(1, len(w) + 1) * FLOPS[m] / 1e9, color=col, lw=.8, label=f'always {m}')
    axes[2].plot(tt, cum_x, color=pc('CAF-X'), lw=1.2, label='CAF-X-eff'); axes[2].set_yscale('log')
    axes[2].set_ylabel('cumulative GFLOPs'); axes[2].set_xlabel('time (s)'); axes[2].legend(fontsize=5)
    fig.suptitle(f'Case study ({ds}): 30 s of forecasts, escalations to {tgt}, and cumulative compute', fontsize=6.5)
    savefig(fig, name)

def figH10(name):          # ablation forest
    variants = list(ABLF.groupby('variant').delta_pct.mean().sort_values().index)
    fig, ax = plt.subplots(figsize=(COL2, 2.4), layout='constrained')
    for j, ds in enumerate(DS):
        for i, v in enumerate(variants):
            d = ABLF[(ABLF.dataset == ds) & (ABLF.variant == v)].delta_pct.values
            if len(d) > 1:
                ci = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                ax.errorbar(d.mean(), i + (j - 1) * .22, xerr=ci, fmt='o', ms=3, capsize=1.5, color=f'C{j}', label=ds if i == 0 else None)
    ax.axvline(0, color='k', lw=.6); ax.set_xscale('symlog', linthresh=1)
    ax.set_yticks(range(len(variants)), [v.replace('CAF ', '') for v in variants], fontsize=5.5)
    ax.set_xlabel('MAE change vs full CAF (%), mean ± 95% CI over seeds'); ax.legend(fontsize=5.5)
    savefig(fig, name)

def figH11(name):          # exit-wise error profile
    fig, axes = plt.subplots(1, len(DS), figsize=(COL2, 2.3), layout='constrained')
    for ax, ds in zip(np.atleast_1d(axes), DS):
        D = DATA[ds]; y = D.splits['test'].y; hx = np.arange(1, HORIZON + 1) * BIN_MS
        m = _caf(ds); outs = {'E0': [], 'E1': [], 'E2': []}; S = D.splits['test']
        with torch.no_grad():
            for i in range(0, S.n, 8192):
                x, _ = S.batch(torch.arange(i, min(i + 8192, S.n), device=DEVICE), MODEL_SPECS[PROPOSED][1])
                for k, e in zip(outs, m.forward_exits(x.float())):
                    outs[k].append(e.float().cpu().numpy())
        ref = None
        for k, v in outs.items():
            mae = np.abs(D.to_mbps(np.concatenate(v)) - y).mean(0); ref = mae if ref is None else ref
            ax.plot(hx, mae, '-o', ms=2, label=f'CAF-{k}')
        if ds in EFF_SEL:
            s = EFF_SEL[ds]; p, _ = _cfg_preds(ds, s['target'], TH_SEED[ds], s['budget'], s['mode'])
            ax.plot(hx, np.abs(p - y).mean(0), '-*', ms=4, color=pc('CAF-X'), label='CAF-X-eff')
        b = BEST_BASELINE[ds]; ax.plot(hx, np.abs(np.mean(all_preds(ds, b, 'test'), 0) - y).mean(0), '--', color=pc(b), label=b)
        ax.set_title(ds); ax.set_xlabel('horizon (ms)')
    np.atleast_1d(axes)[0].set_ylabel('test MAE (Mbit/s)'); np.atleast_1d(axes)[-1].legend(fontsize=4.8)
    savefig(fig, name)

def figH12(name):          # per-day accuracy-compute arrows
    fig, ax = plt.subplots(figsize=(3.5, 2.6), layout='constrained')
    for ds in DS:
        if ds not in EFF_SEL:
            continue
        s = EFF_SEL[ds]; D = DATA[ds]; y = D.splits['test'].y; day = D.splits['test'].day; b = BEST_BASELINE[ds]
        pb = np.mean(all_preds(ds, b, 'test'), 0); outs = [_cfg_preds(ds, s['target'], sd, s['budget'], s['mode']) for sd in SEEDS[:3]]
        px = np.mean([o[0] for o in outs], 0)
        escs = [_cfg_esc(ds, s['target'], sd, s['budget'], s['mode']) for sd in SEEDS[:3]]
        for d in np.unique(day):
            k = day == d; ex = np.mean([cafx_flops(s['target'], float(e[k].mean())) for e in escs]) / 1e6
            mb, mx = np.abs(pb[k] - y[k]).mean(), np.abs(px[k] - y[k]).mean()
            ax.annotate('', xy=(ex, 100 * (mx / mb - 1)), xytext=(FLOPS[b] / 1e6, 0), arrowprops=dict(arrowstyle='->', lw=.7))
            ax.scatter(ex, 100 * (mx / mb - 1), s=12, label=ds if d == 0 else None)
    ax.set_xscale('log'); ax.axhline(0, color='k', lw=.5); ax.set_xlabel('MFLOPs per forecast'); ax.set_ylabel('MAE vs best single model (%)')
    ax.set_title('Per test day: best single model → CAF-X-eff', fontsize=6.5); ax.legend(fontsize=5)
    savefig(fig, name)

for nm, fn in (('figH01_theorem1_exit_equivariance', figH01), ('figH02_prop2_router_invariance', figH02),
               ('figH03_theorem3_budget_concentration', figH03), ('figH04_theorem4_oracle_capture', figH04),
               ('figH05_prop5_blend_bound', figH05), ('figH06_glassbox_waterfall', figH06), ('figH07_glassbox_E1_weights', figH07),
               ('figH08_router_calibration', figH08), ('figH09_case_timeline', figH09), ('figH10_ablation_forest', figH10),
               ('figH11_exit_profiles', figH11), ('figH12_per_day_arrows', figH12)):
    _hrun(nm, fn)
display(pd.DataFrame(H_LOG, columns=['figure', 'status']))
checkpoint('theory + glass-box', force=True)


# Run after the main training, cascade, statistics, and efficiency cells.
# Uses existing checkpoints and predictions; does not retrain or alter results.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from scipy import stats as scipy_stats

assert "DATA" in globals() and "DEEP_PREDS" in globals() and "CASCADE" in globals(), \
    "Run the training and cascade cells first."

XAI_DS = "MAWI-2025" if "MAWI-2025" in DATA else next(iter(DATA))
XAI_D = DATA[XAI_DS]
XAI_SEEDS = sorted(
    k[3] for k in CASCADE
    if k[0] == "main" and k[1] == XAI_DS and k[2] == PROPOSED
)
assert XAI_SEEDS, f"No fitted CAF router/checkpoint for {XAI_DS}"
XAI_SEED = XAI_SEEDS[0]
XAI_KEY = ("main", XAI_DS, PROPOSED, XAI_SEED)
XAI_S = XAI_D.splits["test"]
XAI_Y = XAI_S.y

caf, router_ck = load_caf("main", XAI_D, PROPOSED, MODEL_SPECS[PROPOSED], XAI_SEED)
caf.float().eval()
for p in caf.parameters():
    p.requires_grad_(False)

P_E2 = DEEP_PREDS[XAI_KEY]["test"]
P_E1 = DEEP_PREDS[("main", XAI_DS, "CAF-E1", XAI_SEED)]["test"]
P_AD = DEEP_PREDS[("main", XAI_DS, "CAF-adaptive", XAI_SEED)]["test"]
P_PERSIST = CLASSICAL_PREDS[("main", XAI_DS, "Persistence")]["test"]
TAU = float(router_ck["tau"])
H_SHOW = [h for h in (1, 5, 10) if h <= XAI_D.H]
C_CAF = "#12263a"
C_DEEP = "#147d86"
C_AD = "#ed8654"
C_GOOD = "#24764a"
C_BAD = "#bd4c51"

plt.rcParams.update({
    "figure.dpi": 125,
    "savefig.dpi": 250,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})

def panel(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", pad=10)
    ax.grid(alpha=.14, lw=.55)

def finish(fig, title):
    fig.suptitle(title, x=.035, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .945))
    plt.show()

def signed_heat(ax, values, xlabels, ylabels, label, cmap="RdBu_r"):
    v = np.asarray(values, float)
    lim = max(float(np.nanpercentile(np.abs(v), 98)), 1e-8)
    im = ax.imshow(v, cmap=cmap, aspect="auto", origin="lower",
                   norm=mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim))
    ax.set_xticks(np.arange(len(xlabels)), xlabels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(ylabels)), ylabels)
    ax.grid(False)
    ax.figure.colorbar(im, ax=ax, fraction=.042, pad=.025, label=label)
    return im

def empirical_survival(x):
    x = np.sort(np.asarray(x, float))
    return x, 1 - np.arange(1, len(x) + 1) / len(x)

# Fixed, evenly spaced diagnostic windows. Attribution is descriptive:
# the all-zero z-score baseline represents training-channel means, which
# need not correspond to a physically possible joint traffic window.
n_attr = min(12, XAI_S.n)
attr_idx = np.linspace(0, XAI_S.n - 1, n_attr, dtype=np.int64)
x_attr, _ = XAI_S.batch(torch.as_tensor(attr_idx, device=DEVICE))
x_attr = x_attr.float().detach()
base_attr = torch.zeros_like(x_attr)

def integrated_gradients_horizon(h, steps=20):
    """Integrated gradients for CAF E2 at horizon h; float32 and no cuDNN RNN backward."""
    acc = torch.zeros_like(x_attr)
    # End-point trapezoidal rule. Do not use inference_mode here.
    with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
        for j in range(steps + 1):
            alpha = j / steps
            xi = (base_attr + alpha * (x_attr - base_attr)).detach().requires_grad_(True)
            output = caf(xi)[:, h - 1].sum()
            grad = torch.autograd.grad(output, xi, retain_graph=False)[0]
            acc += grad.detach() * (.5 if j in (0, steps) else 1.)
    return ((x_attr - base_attr) * acc / steps).detach().cpu().numpy()

ATTR = {h: integrated_gradients_horizon(h) for h in H_SHOW}
W = caf.anchor_weights().detach().cpu().numpy()

# ── FIGURE 1: Forecast attribution and architecture ───────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
a, b, c, d = axes.flat

anchor_labels = ["last"] + [f"MA {s}" for s in caf.scales] + ["window mean"]
im = a.imshow(W.T, origin="lower", aspect="auto", cmap="magma")
a.set_yticks(range(len(anchor_labels)), anchor_labels)
a.set_xticks(range(XAI_D.H), (np.arange(XAI_D.H) + 1) * BIN_MS)
a.set_xlabel("Forecast horizon (ms)")
a.figure.colorbar(im, ax=a, fraction=.042, pad=.025, label="Convex anchor weight")
panel(a, "A", "Learned E0 multiscale anchor")

channel_attr = np.stack(
    [np.mean(np.abs(ATTR[h]), axis=(0, 1)) for h in H_SHOW], axis=1
)
im = b.imshow(channel_attr, origin="lower", aspect="auto", cmap="viridis")
b.set_yticks(range(len(CHANNELS)), CHANNELS)
b.set_xticks(range(len(H_SHOW)), [f"{h * BIN_MS} ms" for h in H_SHOW])
b.figure.colorbar(im, ax=b, fraction=.042, pad=.025,
                  label="Mean |integrated attribution| (z-score output)")
panel(b, "B", "Channel influence across forecast horizons")

lag_attr = np.stack(
    [np.mean(np.abs(ATTR[h]), axis=(0, 2)) for h in H_SHOW]
)
im = c.imshow(lag_attr, origin="lower", aspect="auto", cmap="cividis",
              extent=[-XAI_D.L * BIN_MS, 0, len(H_SHOW) - .5, -.5])
c.set_yticks(range(len(H_SHOW)), [f"{h * BIN_MS} ms" for h in H_SHOW])
c.set_xlabel("Input age relative to forecast origin (ms)")
c.figure.colorbar(im, ax=c, fraction=.042, pad=.025,
                  label="Mean |integrated attribution|")
panel(c, "C", "Which parts of the causal window matter?")

h_mid = min(H_SHOW, key=lambda h: abs(h - 5))
attr_cl = np.mean(ATTR[h_mid], axis=0).T
signed_heat(
    d, attr_cl,
    [str(v) for v in range(-XAI_D.L * BIN_MS, 0, BIN_MS)],
    CHANNELS, "Signed attribution"
)
d.set_xticks(np.arange(0, XAI_D.L, 8),
             [f"{(i - XAI_D.L) * BIN_MS}" for i in range(0, XAI_D.L, 8)])
d.set_xlabel("Input age (ms)")
panel(d, "D", f"Signed channel × lag attribution at {h_mid * BIN_MS} ms")
finish(fig, f"CAF forecast explanations  |  {XAI_DS}  |  seed {XAI_SEED}")

# Router diagnostics use a bounded, chronological test subsample. The cached
# predictions remain aligned to their original test-window indices.
route_idx = np.unique(np.linspace(
    0, XAI_S.n - 1, min(12000, XAI_S.n), dtype=np.int64
))
scores = []
with torch.inference_mode():
    for start in range(0, len(route_idx), 2048):
        ii = torch.as_tensor(route_idx[start:start + 2048], device=DEVICE)
        xb, _ = XAI_S.batch(ii)
        _, score = caf.forward_cheap(xb.float())
        scores.append(score.float().cpu().numpy())
score = np.concatenate(scores)
selected = score > TAU
sample_gain = (
    np.abs(P_E1[route_idx] - XAI_Y[route_idx])
    - np.abs(P_E2[route_idx] - XAI_Y[route_idx])
).mean(axis=1)
sample_scale = np.maximum(
    XAI_S.R[XAI_S.t[route_idx] - 1], 1.0
)
# The fitted router predicts gain divided by the input window's local scale.
# For comparisons below, realized gain is kept in Mbit/s; rankings are the focus.

def decile_curve(score_values, gains, n_bins=10):
    edges = np.quantile(score_values, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    group = np.clip(np.searchsorted(edges[1:-1], score_values, side="right"),
                    0, len(edges) - 2)
    rows = []
    for q in range(len(edges) - 1):
        mask = group == q
        if mask.any():
            rows.append((q + 1, score_values[mask].mean(),
                         gains[mask].mean(), mask.sum()))
    return np.asarray(rows)

dc = decile_curve(score, sample_gain)

# ── FIGURE 2: Router explanation and compute allocation ──────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
a, b, c, d = axes.flat

hb = a.hexbin(score, sample_gain, gridsize=46, mincnt=2,
              bins="log", cmap="viridis")
a.axhline(0, color=C_BAD, ls="--", lw=1, label="No benefit from E2")
a.axvline(TAU, color=C_AD, ls=":", lw=1.4, label="Validation-set threshold")
a.set_xlabel("Learned router score")
a.set_ylabel("Realized E1 − E2 absolute-error gain (Mbit/s)")
a.legend(loc="upper left")
a.figure.colorbar(hb, ax=a, fraction=.042, pad=.025, label="log window count")
panel(a, "E", "Does the router identify useful escalation?")

b.plot(dc[:, 0], dc[:, 2], marker="o", color=C_DEEP, lw=2)
b.axhline(0, color=C_BAD, ls="--", lw=1)
b.set_xticks(dc[:, 0].astype(int))
b.set_xlabel("Router-score decile (10 = highest priority)")
b.set_ylabel("Mean realized E1 − E2 gain (Mbit/s)")
panel(b, "F", "Observed value of compute by router decile")

route_main = CASC[(CASC.study == "main") & (CASC.dataset == XAI_DS)]
route_mean = route_main.groupby(["router", "budget"], as_index=False)[
    ["test_mae", "test_rate"]
].mean()
route_style = {
    "learned": (C_DEEP, "-"),
    "volatility": (C_AD, "--"),
    "random": ("#88939d", ":"),
    "oracle (test labels)": (C_GOOD, "-."),
}
for rn, (color, style) in route_style.items():
    sub = route_mean[route_mean.router == rn].sort_values("test_rate")
    if len(sub):
        c.plot(100 * sub.test_rate, sub.test_mae, marker="o", ms=3,
               color=color, ls=style, lw=1.7, label=rn)
op_rate = 100 * selected.mean()
c.axvline(op_rate, color=C_CAF, ls=":", lw=1,
          label=f"Selected seed: {op_rate:.1f}% test escalation")
c.set_xlabel("Realized test windows escalated (%)")
c.set_ylabel("Test MAE (Mbit/s)")
c.legend()
panel(c, "G", "Router ablation at matched validation budgets")

for mask, label, color in [
    (selected, "Escalated", C_AD),
    (~selected, "Cheap exit", C_DEEP),
]:
    if mask.sum() > 2:
        vals = np.sort(score[mask])
        d.plot(vals, np.arange(1, len(vals) + 1) / len(vals),
               label=f"{label} (n={mask.sum():,})", color=color, lw=2)
d.axvline(TAU, color=C_CAF, ls="--", lw=1, label="Calibrated threshold")
d.set_xlabel("Router score")
d.set_ylabel("Within-group empirical CDF")
d.legend()
panel(d, "H", "Score distribution behind the compute decision")
finish(fig, f"Adaptive-compute explanations  |  {XAI_DS}")

# ── FIGURE 3: Accuracy, systems, and operational behavior ────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
a, b, c, d = axes.flat

for ds, color in zip(DS, ["#147d86", "#ed8654", "#875ba6"]):
    sub = POINTS[(POINTS.dataset == ds) & np.isfinite(POINTS.flops)
                 & (POINTS.flops > 0)].sort_values("flops")
    a.scatter(sub.flops, sub.rel_mae, color=color, s=24, alpha=.45)
    frontier = sub[sub.model.isin(PARETO[ds])].sort_values("flops")
    a.plot(frontier.flops, frontier.rel_mae, color=color, marker="o",
           ms=4, lw=1.7, label=ds)
caf_point = POINTS[(POINTS.dataset == XAI_DS) & (POINTS.model == "CAF")]
if len(caf_point):
    a.annotate("CAF E2", (caf_point.flops.iloc[0], caf_point.rel_mae.iloc[0]),
               xytext=(5, 8), textcoords="offset points", fontsize=8)
a.set_xscale("log")
a.set_xlabel("Counted FLOPs per forecast (log scale)")
a.set_ylabel("MAE / persistence MAE")
a.legend()
panel(a, "I", "Accuracy–compute Pareto frontiers")

for name, color, lw in [
    ("Persistence", "#88939d", 1.3),
    ("CAF-E1", C_AD, 1.7),
    ("CAF-adaptive", C_DEEP, 2.1),
    ("CAF", C_CAF, 2.1),
]:
    sub = FULL[(FULL.dataset == XAI_DS) & (FULL.model == name)].sort_values("h")
    if len(sub):
        b.plot(sub.h * BIN_MS, sub.skill, label=name, color=color, lw=lw,
               marker="o", ms=3)
b.axhline(0, color="#88939d", ls="--", lw=.8)
b.set_xlabel("Forecast horizon (ms)")
b.set_ylabel("Skill relative to persistence")
b.legend()
panel(b, "J", "Forecast skill as the horizon increases")

h = h_mid - 1
for name, pred, color in [
    ("Persistence", P_PERSIST, "#88939d"),
    ("CAF E1", P_E1, C_AD),
    ("CAF adaptive", P_AD, C_DEEP),
    ("CAF E2", P_E2, C_CAF),
]:
    err = np.abs(pred[:, h] - XAI_Y[:, h])
    xx, yy = empirical_survival(err)
    c.plot(xx, np.maximum(yy, 1 / len(yy)), color=color, lw=1.7, label=name)
c.set_yscale("log")
c.set_xlim(0, np.percentile(np.abs(P_PERSIST[:, h] - XAI_Y[:, h]), 99.9))
c.set_xlabel(f"Absolute error at {(h + 1) * BIN_MS} ms (Mbit/s)")
c.set_ylabel("P(error exceeds threshold), log scale")
c.legend()
panel(c, "K", "Tail reliability of forecast error")

thresholds = np.quantile(XAI_Y[:, h], np.linspace(.75, .995, 30))
for name, pred, color in [
    ("Persistence", P_PERSIST, "#88939d"),
    ("CAF adaptive", P_AD, C_DEEP),
    ("CAF E2", P_E2, C_CAF),
]:
    precision, recall = [], []
    for threshold in thresholds:
        actual = XAI_Y[:, h] > threshold
        called = pred[:, h] > threshold
        tp = np.sum(actual & called)
        precision.append(tp / max(int(called.sum()), 1))
        recall.append(tp / max(int(actual.sum()), 1))
    d.plot(recall, precision, color=color, lw=1.7, marker=".",
           ms=3, label=name)
d.set_xlim(0, 1.02)
d.set_ylim(0, 1.02)
d.set_xlabel("Recall for rate exceeding threshold")
d.set_ylabel("Precision")
d.legend()
panel(d, "L", "Burst detection across load thresholds")
finish(fig, "Systems and operational evidence  |  held-out test days")

# ── FIGURE 4: Event-level forecast and failure analysis ──────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
a, b, c, d = axes.flat

# Pick a high-volatility segment wholly inside the first test day.
first_day = np.flatnonzero(XAI_S.day == 0)
segment_len = min(260, max(len(first_day) // 4, 20))
segment_len = min(segment_len, len(first_day))
rate_first = XAI_Y[first_day, 0]
probe = np.linspace(0, max(len(first_day) - segment_len, 0),
                    min(250, max(len(first_day) - segment_len + 1, 1)),
                    dtype=np.int64)
best_start = int(max(
    probe, key=lambda j: np.std(rate_first[j:j + segment_len])
))
seg = first_day[best_start:best_start + segment_len]
tt = np.arange(len(seg)) * BIN_MS / 1000

a.plot(tt, XAI_Y[seg, h], color="#929aa3", lw=1.5, label="Observed")
a.plot(tt, P_E1[seg, h], color=C_AD, lw=1, alpha=.85, label="E1")
a.plot(tt, P_AD[seg, h], color=C_DEEP, lw=1.5, label="Adaptive")
a.plot(tt, P_E2[seg, h], color=C_CAF, lw=1, alpha=.85, label="E2")
a.set_xlabel("Time within selected high-volatility segment (s)")
a.set_ylabel("Wire rate (Mbit/s)")
a.legend(ncol=4)
panel(a, "M", f"Forecast trace at {(h + 1) * BIN_MS} ms")

# Validation-only split-conformal upper offset; plot test outcomes as a diagnostic.
p_val = DEEP_PREDS[("main", XAI_DS, "CAF-adaptive", XAI_SEED)]["val"]
r_val = XAI_D.splits["val"].y[:, h] - p_val[:, h]
n_val = len(r_val)
q_level = min(1., np.ceil((n_val + 1) * (1 - ALPHA)) / n_val)
q_upper = np.quantile(r_val, q_level)
upper = P_AD[seg, h] + q_upper
b.plot(tt, XAI_Y[seg, h], color=C_CAF, lw=1.4, label="Observed")
b.plot(tt, P_AD[seg, h], color=C_DEEP, lw=1.3, label="Adaptive point forecast")
b.plot(tt, upper, color=C_AD, lw=1.2, ls="--",
       label=f"Validation-calibrated upper ({1 - ALPHA:.0%})")
b.fill_between(tt, P_AD[seg, h], upper, color=C_AD, alpha=.14)
miss = XAI_Y[seg, h] > upper
if miss.any():
    b.scatter(tt[miss], XAI_Y[seg, h][miss], color=C_BAD,
              marker="x", s=32, label="Upper-bound exceedance", zorder=5)
b.set_xlabel("Time within segment (s)")
b.set_ylabel("Wire rate (Mbit/s)")
b.legend(ncol=2)
panel(b, "N", "Capacity headroom and exceedances")

density_idx = np.unique(np.linspace(
    0, XAI_S.n - 1, min(35000, XAI_S.n), dtype=np.int64
))
hb = c.hexbin(
    XAI_Y[density_idx, h],
    P_AD[density_idx, h] - XAI_Y[density_idx, h],
    gridsize=54, mincnt=2, bins="log", cmap="viridis"
)
c.axhline(0, color=C_BAD, ls="--", lw=1)
c.set_xlabel("Observed wire rate (Mbit/s)")
c.set_ylabel("Adaptive forecast − observed (Mbit/s)")
c.figure.colorbar(hb, ax=c, fraction=.042, pad=.025, label="log window count")
panel(c, "O", "Load-dependent residual structure")

day_gain = []
for day_i, day in enumerate(XAI_S.days):
    mask = XAI_S.day == day_i
    for horizon in range(XAI_D.H):
        e_ref = np.abs(P_PERSIST[mask, horizon] - XAI_Y[mask, horizon]).mean()
        e_ad = np.abs(P_AD[mask, horizon] - XAI_Y[mask, horizon]).mean()
        day_gain.append((day_i, horizon, 100 * (1 - e_ad / e_ref)))
gain_matrix = np.full((len(XAI_S.days), XAI_D.H), np.nan)
for day_i, horizon, improvement in day_gain:
    gain_matrix[day_i, horizon] = improvement
signed_heat(
    d, gain_matrix,
    [str((i + 1) * BIN_MS) for i in range(XAI_D.H)],
    XAI_S.days,
    "MAE improvement vs persistence (%)"
)
d.set_xlabel("Forecast horizon (ms)")
for i in range(len(XAI_S.days)):
    for j in range(XAI_D.H):
        d.text(j, i, f"{gain_matrix[i, j]:+.1f}",
               ha="center", va="center", fontsize=6,
               color="white" if abs(gain_matrix[i, j]) >
               .55 * max(np.nanmax(np.abs(gain_matrix)), 1e-8) else "black")
panel(d, "P", "Held-out day × horizon robustness")
finish(fig, f"Forecast events and failure modes  |  {XAI_DS}")

print(
    f"Displayed 16 panels. Detailed explanations: {XAI_DS}, seed {XAI_SEED}; "
    f"IG: {n_attr} fixed test windows × {len(H_SHOW)} horizons × 20 steps. "
    "Attributions use a training-mean (zero z-score) reference and are descriptive, "
    "not causal feature effects. The oracle routing curve uses test labels and is "
    "an upper bound. Conformal coverage can be affected by temporal dependence."
)