"""
Accuracy-compute trade-off, Pareto front, and CAF-X model-agnostic serving.
Corresponds to Cells 29, 30, 35 of cmsnts_latest.ipynb.
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from config import (
    DEVICE, PROPOSED, TABLES, FIGS, BUDGET_GRID, ROUTER_EPOCHS, DS
)
from utils.helpers import checkpoint, stamp, savefig
from experiments.metrics import mae_of
from experiments.engine import load_model

def direct_mae(ds, model):
    D = DATA[ds]; ps = all_preds(ds, model, 'test')
    return float(np.mean([np.abs(p - D.splits['test'].y).mean() for p in ps])) if ps else np.nan

PERS = {ds: direct_mae(ds, 'Persistence') for ds in DS}
CASC = pd.DataFrame(CASCADE_ROWS)
POINTS = []                                                   # (dataset, model, flops, rel MAE)
for ds in DS:
    for m in ORDER:
        fl = FLOPS.get(m, CLASSICAL_FLOPS.get(m, np.nan))
        POINTS.append(dict(dataset=ds, model=m, flops=fl, rel_mae=direct_mae(ds, m) / PERS[ds]))
POINTS = pd.DataFrame(POINTS)

def pareto_set(df):
    d = df.dropna(subset=['flops', 'rel_mae']).sort_values(['flops', 'rel_mae']); best, keep = np.inf, []
    for _, r in d.iterrows():
        if r.rel_mae < best - 1e-12:
            keep.append(r.model); best = r.rel_mae
    return keep

PARETO = {ds: pareto_set(POINTS[POINTS.dataset == ds]) for ds in DS}
print('Pareto-optimal models (FLOPs vs MAE):'); [print(f'  {ds}: {v}') for ds, v in PARETO.items()]

if len(CASC):
    cm = CASC[(CASC.study == 'main')]
    curve = cm.groupby(['dataset', 'router', 'budget'])[['test_mae', 'test_rate', 'val_mae']].mean().reset_index()
    curve['flops'] = curve.test_rate.map(cascade_flops)
    curve['rel_mae'] = [r.test_mae / PERS[r.dataset] for r in curve.itertuples()]
    curve.to_csv(TABLES / 'cascade_budget_curves.csv', index=False)
    # Table IX: learned-router budget sweep relative to always running E2
    lr_c = curve[curve.router == 'learned'].copy()
    full = lr_c[lr_c.budget == 1.0].set_index('dataset').test_mae
    lr_c['dMAE_vs_full_pct'] = [100 * (r.test_mae / full[r.dataset] - 1) for r in lr_c.itertuples()]
    lr_c['compute_saving_x'] = CAF_FLOPS['full'] / lr_c.flops
    TABLE_BUDGET = lr_c.pivot(index='budget', columns='dataset', values='dMAE_vs_full_pct').round(2)
    TABLE_BUDGET.columns = [f'{c} ΔMAE%' for c in TABLE_BUDGET.columns]
    TABLE_BUDGET['FLOP saving (x)'] = lr_c.groupby('budget').compute_saving_x.mean().round(1)
    TABLE_BUDGET.index = [f'{100 * b:.0f}%' for b in TABLE_BUDGET.index]; TABLE_BUDGET.index.name = 'Escalation budget'
    display(TABLE_BUDGET)
    to_latex(TABLE_BUDGET.astype(str), TABLES / 'table9_budget_sweep.tex',
             'CAF budget sweep (learned router, threshold calibrated on validation): test MAE change vs always running '
             'the deep exit, and average FLOP saving. Mean over seeds.', 'tab:budget')
    # Table X: router ablation at fixed budgets. Primary number: test-MAE change vs the cheap exit E1 (always defined);
    # "gap recovered" (share of the E1->E2 improvement) only where E2 is better than E1 by >0.2 % (else the ratio is
    # numerically meaningless and reported as n/a).
    e1m = curve[(curve.router == 'learned') & (curve.budget == 0)].set_index('dataset').test_mae
    rows = {}
    for b in (.1, .2, .3):
        for rn in ('learned', 'volatility', 'random', 'oracle (test labels)'):
            c = curve[(curve.router == rn) & (curve.budget == b)].set_index('dataset').test_mae
            cell = {}
            for ds in c.index:
                gap = e1m[ds] - full[ds]
                rec = f'{100 * (e1m[ds] - c[ds]) / gap:.0f}%' if gap > 2e-3 * e1m[ds] else 'n/a'
                cell[ds] = f'{100 * (c[ds] / e1m[ds] - 1):+.2f} ({rec})'
            rows[(rn, f'{100 * b:.0f}%')] = cell
    TABLE_ROUTER = pd.DataFrame(rows).T; TABLE_ROUTER.index = [f'{a} @ {b}' for a, b in TABLE_ROUTER.index]
    TABLE_ROUTER.index.name = 'Router @ budget'; display(TABLE_ROUTER)
    to_latex(TABLE_ROUTER, TABLES / 'table10_router_ablation.tex',
             'Router ablation: test-MAE change (%) vs the cheap exit E1 when only a budget fraction of windows is '
             'escalated (negative = better); in brackets the share of the E1->E2 gap recovered (n/a if E2 does not '
             'improve on E1 by >0.2%). Oracle uses test labels (upper bound, not deployable).', 'tab:router')
    # deep-supervision ablation of the cascade (if the ablated backbone was trained)
    ca = CASC[(CASC.study == 'ablation') & (CASC.router == 'learned')]
    if len(ca):
        cmp_ = pd.concat([cm[cm.router == 'learned'].assign(variant='CAF (deep supervision)'),
                          ca.assign(variant='CAF w/o deep supervision')])
        ds_t = cmp_.groupby(['variant', 'dataset', 'budget']).test_mae.mean().unstack('budget')[[0, .2, 1.0]].round(3)
        ds_t.columns = [f'MAE @ {100 * c:.0f}%' for c in ds_t.columns]; display(ds_t)
        ds_t.to_csv(TABLES / 'cascade_deep_supervision_ablation.csv')
    # accuracy at a FLOP budget: best fixed model vs CAF cascade
    acc_rows = []
    for ds in DS:
        P = POINTS[POINTS.dataset == ds]; cc = lr_c[lr_c.dataset == ds]
        for B in (1e3, 1e4, 1e5, 1e6, 1e7):
            fx = P[P.flops <= B].dropna(subset=['rel_mae']).sort_values('rel_mae').head(1)
            cs = cc[cc.flops <= B].sort_values('rel_mae').head(1)
            acc_rows.append(dict(dataset=ds, flop_budget=f'{B:.0e}',
                                 best_fixed=f'{fx.model.iloc[0]} ({fx.rel_mae.iloc[0]:.3f})' if len(fx) else '--',
                                 caf_cascade=f'{cs.rel_mae.iloc[0]:.3f} @ {100 * cs.budget.iloc[0]:.0f}%' if len(cs) else '--'))
    ACC_AT_BUDGET = pd.DataFrame(acc_rows); display(ACC_AT_BUDGET)
    ACC_AT_BUDGET.to_csv(TABLES / 'accuracy_at_flop_budget.csv', index=False)

    fig, axes = plt.subplots(1, len(DS), figsize=(7.2, 2.6), layout='constrained'); axes = np.atleast_1d(axes)
    sty = {'learned': ('k', '-'), 'volatility': ('C1', '--'), 'random': ('C7', ':'), 'oracle (test labels)': ('C2', '-.')}
    for ax, ds in zip(axes, DS):
        P = POINTS[(POINTS.dataset == ds) & (POINTS.model.isin(DEEP_MODELS + ['CAF-E1']))]
        for r in P.itertuples():
            ax.scatter(r.flops, r.rel_mae, s=14, color='C3' if r.model.startswith('CAF') else 'C0',
                       marker='s' if r.model.startswith('CAF') else 'o', zorder=3)
            ax.annotate(r.model, (r.flops, r.rel_mae), fontsize=4.5, xytext=(2, 2), textcoords='offset points')
        for rn, (col, ls) in sty.items():
            c = curve[(curve.dataset == ds) & (curve.router == rn)].sort_values('flops')
            ax.plot(c.flops, c.rel_mae, color=col, ls=ls, lw=1, marker='.', ms=3, label=f'CAF cascade ({rn})')
        if 'CAF-adaptive' in set(POINTS.model):
            pa = POINTS[(POINTS.dataset == ds) & (POINTS.model == 'CAF-adaptive')].dropna(subset=['rel_mae'])
            if len(pa) and ds in OP_RATE:
                ax.scatter(cascade_flops(OP_RATE[ds]), pa.rel_mae.iloc[0], marker='*', s=70, color='k', zorder=4,
                           label='CAF-adaptive (val-calibrated)')
        for m in ('EWMA', 'Ridge-AR', 'HistGB'):
            v = POINTS[(POINTS.dataset == ds) & (POINTS.model == m)].rel_mae
            if len(v):
                ax.axhline(v.iloc[0], lw=.5, ls=':', color='grey')
                ax.text(.01, v.iloc[0], m, fontsize=4.5, color='grey', va='bottom', transform=ax.get_yaxis_transform())
        ax.set_xscale('log'); ax.set_title(ds); ax.set_xlabel('FLOPs per forecast')
    axes[0].set_ylabel('MAE / MAE(persistence)'); axes[-1].legend(fontsize=4.5, loc='upper right')
    savefig(fig, 'fig_pareto_flops')

fig, ax = plt.subplots(figsize=(3.5, 2.5), layout='constrained')
avg_rel = POINTS.groupby('model').rel_mae.mean()
for m in EFF.index:
    if m in avg_rel:
        ax.scatter(EFF.loc[m, 'cpu1t_ms_b1'], avg_rel[m], s=14, color='k' if m.startswith('CAF') else 'C0',
                   marker='*' if m == 'CAF-adaptive' else 'o', zorder=3)
        ax.annotate(m, (EFF.loc[m, 'cpu1t_ms_b1'], avg_rel[m]), fontsize=5, xytext=(2, 2), textcoords='offset points')
ax.axvline(BIN_MS, ls='--', lw=.6, color='C3'); ax.text(BIN_MS, ax.get_ylim()[1], ' 10 ms cadence', fontsize=5, color='C3', va='top')
ax.set_xscale('log'); ax.set_xlabel('CPU single-core latency per window (ms)'); ax.set_ylabel('MAE / MAE(persistence), avg')
savefig(fig, 'fig_pareto_cpu_latency')
checkpoint('analysis cell 28')


#       CAF's 0.8k-parameter scale-equivariant exit E1 answers EVERY window at ~2 kFLOPs; a scale-free router (fit on
#       TRAINING days, budget calibrated on the VALIDATION day) escalates only predicted-hard windows to ANY deep
#       forecaster of the zoo. Escalation target, budget and escalation output (deep / mean(E1, deep)) are chosen on the
#       validation day only; test days are evaluated once. This turns CAF into a serving layer that wraps SOTA models.
CAFX_TOP_K = 3                                    # validation-best deep baselines wrapped per dataset
CAFX_ALPHAS = (1.0, 0.75, 0.5, 0.25)              # escalated output = a*deep + (1-a)*E1 (a chosen on validation)
CAFX = {}                                         # (ds, target, seed) -> dict(val/test preds, op point, sweep rows)
CAFX_ROWS = []
_DEEP_BASE = [m for m in DEEP_MODELS if m != PROPOSED]

def _seed_val_mae(ds, m):
    y = DATA[ds].splits['val'].y
    v = [np.abs(p['val'] - y).mean() for k, p in DEEP_PREDS.items() if k[0] == 'main' and k[1] == ds and k[2] == m]
    return float(np.mean(v)) if v else np.inf

CAFX_TARGETS = {ds: sorted(_DEEP_BASE, key=lambda m: _seed_val_mae(ds, m))[:CAFX_TOP_K] for ds in DS}
for ds in DS:                                     # + the validation-top-2 deep ensemble as a heavier escalation target
    CAFX_TARGETS[ds].append(' + '.join(CAFX_TARGETS[ds][:2]))
for ts in CAFX_TARGETS.values():
    for t in ts:
        if ' + ' in t:
            FLOPS[t] = sum(FLOPS[q] for q in t.split(' + '))
print('Escalation targets (validation-selected, test never used):', CAFX_TARGETS)

def _caf_cheap(ds, seed):
    """E1 forecasts (Mbit/s), raw router features and local scale for train/val/test, cached on disk"""
    D = DATA[ds]; f = PREDS / f'{run_key("main", D, PROPOSED, seed)}__cafx_cheap.npz'
    if f.exists():
        z = np.load(f); return {s: {k: z[f'{s}_{k}'] for k in ('e1', 'f', 'sc')} for s in ('train', 'val', 'test')}
    m = load_model('main', D, PROPOSED, MODEL_SPECS[PROPOSED], seed)
    with torch.no_grad():
        m.rf_mu.zero_(); m.rf_sd.fill_(1.0)
    out = {}
    for s in ('train', 'val', 'test'):
        o = caf_outputs(m, D, s, MODEL_SPECS[PROPOSED][1])
        out[s] = dict(e1=D.to_mbps(o['e1'].cpu().numpy()).astype(np.float32), f=o['f'].cpu().numpy().astype(np.float32),
                      sc=(o['sc'].cpu().numpy() * float(D.sd[0])).astype(np.float32))          # local scale in Mbit/s
    np.savez_compressed(f, **{f'{s}_{k}': v for s, d in out.items() for k, v in d.items()})
    del m; gc.collect(); return out

def _deep_train_preds(ds, m, seed):
    D = DATA[ds]; f = PREDS / f'{run_key("main", D, m, seed)}__train.npz'
    if f.exists():
        return np.load(f)['train']
    model = load_model('main', D, m, MODEL_SPECS[m], seed); p = predict(model, D, 'train', MODEL_SPECS[m][1])
    np.savez_compressed(f, train=p); del model; gc.collect(); return p

def _parts(tgt):
    return tgt.split(' + ')

def _deep_preds(ds, tgt, seed):
    ps = [{'train': _deep_train_preds(ds, q, seed), 'val': DEEP_PREDS[('main', ds, q, seed)]['val'],
           'test': DEEP_PREDS[('main', ds, q, seed)]['test']} for q in _parts(tgt)]
    return {s: np.mean([p[s] for p in ps], 0).astype(np.float32) for s in ('train', 'val', 'test')}

def _fit_router(Xtr, gtr, Xva, gva, seed):
    set_seed(seed)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 32), nn.GELU(), nn.Linear(32, 1)).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    X, g = torch.tensor(Xtr, device=DEVICE), torch.tensor(gtr, device=DEVICE)
    Xv, gv = torch.tensor(Xva, device=DEVICE), torch.tensor(gva, device=DEVICE)
    gen = torch.Generator().manual_seed(seed); best, state, bad = np.inf, None, 0
    for ep in range(ROUTER_EPOCHS):
        net.train(); perm = torch.randperm(len(X), generator=gen).to(DEVICE)
        for i in range(0, len(X), 2048):
            ii = perm[i:i + 2048]
            loss = F.huber_loss(net(X[ii]).squeeze(-1), g[ii], delta=1.0)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = F.huber_loss(net(Xv).squeeze(-1), gv, delta=1.0).item()
        if vl < best - 1e-7:
            best, bad, state = vl, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                break
    net.load_state_dict(state); net.eval()
    return net

def _cafx_one(ds, tgt, seed):
    D = DATA[ds]; key = f'cafx_{slug(ds)}_{slug(tgt)}_s{seed}'; fj, fp = HIST / f'{key}.json', PREDS / f'{key}.npz'
    if fj.exists() and fp.exists():
        z = np.load(fp); return dict(json.loads(fj.read_text()), val=z['val'], test=z['test'])
    c = _caf_cheap(ds, seed); deep = _deep_preds(ds, tgt, seed)
    y = {s: D.splits[s].y for s in ('train', 'val', 'test')}
    gain = {s: (np.abs(c[s]['e1'] - y[s]) - np.abs(deep[s] - y[s])).mean(1) / c[s]['sc'] for s in ('train', 'val')}
    mu, sd = c['train']['f'].mean(0), c['train']['f'].std(0) + 1e-6
    X = {s: ((c[s]['f'] - mu) / sd).astype(np.float32) for s in ('train', 'val', 'test')}
    net = _fit_router(X['train'], gain['train'].astype(np.float32), X['val'], gain['val'].astype(np.float32), seed)
    with torch.no_grad():
        score = {s: net(torch.tensor(X[s], device=DEVICE)).squeeze(-1).cpu().numpy() for s in ('val', 'test')}
    rng = np.random.default_rng(seed); H = D.H
    routers = {'learned': score, 'volatility': {s: c[s]['f'][:, H] for s in ('val', 'test')},
               'random': {s: rng.random(len(y[s])) for s in ('val', 'test')},
               'oracle (test labels)': {s: (np.abs(c[s]['e1'] - y[s]) - np.abs(deep[s] - y[s])).mean(1) for s in ('val', 'test')}}
    rows, cand = [], {}
    for rn, sc_ in routers.items():
        for q in BUDGET_GRID:
            tau = budget_threshold(sc_['val'], q)
            esc = {s: (sc_[s] > tau) for s in ('val', 'test')}
            for a in CAFX_ALPHAS:
                mode = f'a={a:g}'
                out = {s: np.where(esc[s][:, None], a * deep[s] + (1 - a) * c[s]['e1'], c[s]['e1']) for s in ('val', 'test')}
                r = dict(router=rn, budget=q, mode=mode, tau=tau, val_rate=float(esc['val'].mean()),
                         test_rate=float(esc['test'].mean()), val_mae=float(np.abs(out['val'] - y['val']).mean()),
                         test_mae=float(np.abs(out['test'] - y['test']).mean()))
                rows.append(r)
                if rn == 'learned':
                    cand[(q, mode)] = (r, out, esc['test'])
    tgt_val = float(np.abs(deep['val'] - y['val']).mean())
    ok = [k for k, (r, _, _) in cand.items() if r['val_mae'] <= tgt_val]
    k_op = min(ok, key=lambda k: (k[0], cand[k][0]['val_mae'])) if ok else min(cand, key=lambda k: cand[k][0]['val_mae'])
    k_best = min(cand, key=lambda k: cand[k][0]['val_mae'])
    r_op, out_op, esc_op = cand[k_op]; r_best, out_best, _ = cand[k_best]
    torch.save({'router': net.state_dict(), 'mu': mu, 'sd': sd, 'tau': r_op['tau']}, CKPT / f'{key}__router.pt')
    info = dict(rows=rows, op=r_op, best=r_best, target_val_mae=tgt_val,
                target_test_mae=float(np.abs(deep['test'] - y['test']).mean()),
                e1_test_mae=float(np.abs(c['test']['e1'] - y['test']).mean()))
    np.savez_compressed(fp, val=out_op['val'].astype(np.float32), test=out_op['test'].astype(np.float32),
                        best_val=out_best['val'].astype(np.float32), best_test=out_best['test'].astype(np.float32),
                        esc_test=esc_op, score_test=score['test'].astype(np.float32))
    fj.write_text(json.dumps(info, default=float))
    return dict(info, val=out_op['val'], test=out_op['test'])

stamp('CAF-X: wrapping validation-selected deep forecasters ...')
for ds in DS:
    for tgt in CAFX_TARGETS[ds]:
        for seed in SEEDS:
            if any(('main', ds, q, seed) not in DEEP_PREDS for q in _parts(tgt) + [PROPOSED]):
                continue
            if remaining_min() < ANALYSIS_RESERVE_MIN and not (HIST / f'cafx_{slug(ds)}_{slug(tgt)}_s{seed}.json').exists():
                print(f'CAF-X {ds}/{tgt}/s{seed} skipped by the time guard'); continue
            try:
                CAFX[(ds, tgt, seed)] = _cafx_one(ds, tgt, seed)
                CAFX_ROWS += [dict(r, dataset=ds, target=tgt, seed=seed) for r in CAFX[(ds, tgt, seed)]['rows']]
            except Exception as e:
                print(f'CAF-X {ds}/{tgt}/s{seed} failed: {type(e).__name__}: {e}')
CAFX_SWEEP = pd.DataFrame(CAFX_ROWS)
CAFX_SWEEP.to_csv(TABLES / 'cafx_budget_sweep.csv', index=False)

def cafx_flops(tgt, rate):
    return CAF_FLOPS['cheap'] + rate * FLOPS[tgt]

def _cpu_ms(m):
    return float(sum(EFF.loc[q, 'cpu1t_ms_b1'] if q in EFF.index else np.nan for q in _parts(m)))

# headline variant per dataset = lowest mean VALIDATION MAE at its operating point (pre-registered, test never used)
CAFX_HEAD = {}
for ds in DS:
    vm = {t: np.mean([CAFX[k]['op']['val_mae'] for k in CAFX if k[0] == ds and k[1] == t]) for t in CAFX_TARGETS[ds]
          if any(k[0] == ds and k[1] == t for k in CAFX)}
    if vm:
        CAFX_HEAD[ds] = min(vm, key=vm.get)
print('CAF-X headline (validation-selected):', CAFX_HEAD)

# ---- headline: per dataset, the validation-best target wrapped by CAF-X, vs every model (DM + Holm, bootstrap) ----
CAFX_MAIN, cafx_dm = [], []
rng_x = np.random.default_rng(11)
for ds in DS:
    D = DATA[ds]; yt = D.splits['test'].y; day = D.splits['test'].day
    for rank, tgt in enumerate(CAFX_TARGETS[ds]):
        ks = [k for k in CAFX if k[0] == ds and k[1] == tgt]
        if not ks:
            continue
        maes = [np.abs(CAFX[k]['test'] - yt).mean() for k in ks]
        rate = float(np.mean([CAFX[k]['op']['test_rate'] for k in ks]))
        fl = cafx_flops(tgt, rate); best_b = BEST_BASELINE[ds]
        row = dict(dataset=ds, target=tgt, target_val_rank=rank + 1, headline=CAFX_HEAD.get(ds) == tgt, seeds=len(ks), cafx_mae=np.mean(maes),
                   cafx_sd=np.std(maes, ddof=1) if len(maes) > 1 else 0.0,
                   target_mae=float(np.mean([CAFX[k]['target_test_mae'] for k in ks])),
                   e1_mae=float(np.mean([CAFX[k]['e1_test_mae'] for k in ks])),
                   budget=float(np.mean([CAFX[k]['op']['budget'] for k in ks])), escalation_rate=rate,
                   mode=pd.Series([CAFX[k]['op']['mode'] for k in ks]).mode().iloc[0],
                   MFLOPs=fl / 1e6, flop_reduction_vs_target_x=FLOPS[tgt] / fl,
                   strongest_test_baseline=best_b, strongest_mae=float(AVG_MAE.loc[best_b, ds]),
                   flop_reduction_vs_strongest_x=FLOPS.get(best_b, np.nan) / fl,
                   cpu_ms_expected=_cpu_ms('CAF-E1') + rate * _cpu_ms(tgt))
        row['vs_target_pct'] = 100 * (row['cafx_mae'] / row['target_mae'] - 1)
        row['vs_strongest_pct'] = 100 * (row['cafx_mae'] / row['strongest_mae'] - 1)
        CAFX_MAIN.append(row)
        if CAFX_HEAD.get(ds) == tgt:                      # significance for the pre-registered (validation-selected) variant
            e_p = np.mean([np.abs(CAFX[k]['test'] - yt) for k in ks], 0).mean(1); rr = []
            for b in [m for m in ORDER if m not in (PROPOSED,)] + [q for q in _parts(tgt) if q not in ORDER]:
                if b in CASCADE_MODELS and b != 'CAF-adaptive':
                    continue
                e_b = abs_err('main', D, b); st_, p = diebold_mariano(e_p, e_b, LAG)
                bs = block_bootstrap_rel(e_p, e_b, day, LAG, BOOTSTRAP_REPS, rng_x)
                rr.append(dict(dataset=ds, target=tgt, baseline=b, rel_mae_change_pct=100 * (e_p.mean() / e_b.mean() - 1),
                               ci_lo_pct=-100 * np.percentile(bs, 97.5), ci_hi_pct=-100 * np.percentile(bs, 2.5),
                               dm_stat=st_, p=p))
            if ' + ' in tgt:                              # the wrapped ensemble itself (always-on), same seeds
                e_b = np.mean([np.abs(_deep_preds(ds, tgt, k[2])['test'] - yt) for k in ks], 0).mean(1)
                st_, p = diebold_mariano(e_p, e_b, LAG); bs = block_bootstrap_rel(e_p, e_b, day, LAG, BOOTSTRAP_REPS, rng_x)
                rr.append(dict(dataset=ds, target=tgt, baseline=tgt, rel_mae_change_pct=100 * (e_p.mean() / e_b.mean() - 1),
                               ci_lo_pct=-100 * np.percentile(bs, 97.5), ci_hi_pct=-100 * np.percentile(bs, 2.5),
                               dm_stat=st_, p=p))
            for r, a in zip(rr, holm([r['p'] for r in rr])):
                r['p_holm'] = a; r['better'] = bool(a < ALPHA and r['dm_stat'] < 0); r['worse'] = bool(a < ALPHA and r['dm_stat'] > 0)
            cafx_dm += rr
CAFX_MAIN = pd.DataFrame(CAFX_MAIN); CAFX_DM = pd.DataFrame(cafx_dm)
CAFX_MAIN.to_csv(TABLES / 'table12_cafx.csv', index=False); CAFX_DM.to_csv(TABLES / 'cafx_dm_bootstrap.csv', index=False)
if len(CAFX_MAIN):
    display(CAFX_MAIN.round(3))
    t12 = CAFX_MAIN.copy()
    t12['CAF-X MAE'] = [fmt_pm(a, b) for a, b in zip(t12.cafx_mae, t12.cafx_sd)]
    t12['vs target'] = t12.vs_target_pct.map(lambda v: f'{v:+.2f}%')
    t12['vs strongest'] = t12.vs_strongest_pct.map(lambda v: f'{v:+.2f}%')
    t12['escal.'] = t12.escalation_rate.map(lambda v: f'{100 * v:.0f}%')
    t12['FLOP saving'] = t12.flop_reduction_vs_target_x.map(lambda v: f'{v:.1f}x')
    t12['CPU ms'] = t12.cpu_ms_expected.map(lambda v: f'{v:.2f}')
    t12 = t12.set_index(['dataset', 'target'])[['CAF-X MAE', 'vs target', 'vs strongest', 'escal.', 'FLOP saving', 'CPU ms']]
    t12.index = [f'{a} / {b}' for a, b in t12.index]; t12.index.name = 'Dataset / wrapped model'
    to_latex(t12, TABLES / 'table12_cafx.tex',
             f'CAF-X wraps the {CAFX_TOP_K} validation-best deep forecasters: test MAE (mean ± SD over seeds), change vs the '
             'wrapped model and vs the strongest (test-selected) baseline, escalation rate, FLOP saving and expected '
             'CPU latency per forecast. Target, budget and escalation mode are chosen on the validation day only.', 'tab:cafx')
if len(CAFX_DM):
    display(CAFX_DM[['dataset', 'baseline', 'rel_mae_change_pct', 'ci_lo_pct', 'ci_hi_pct', 'p_holm', 'better', 'worse']].round(4))

# ---- figure: accuracy-compute curves of CAF-X (budget sweep) over every model's operating point ----
if len(CAFX_SWEEP):
    fig, axes = plt.subplots(1, len(DS), figsize=(7.16, 2.5), layout='constrained')
    for ax, ds in zip(np.atleast_1d(axes), DS):
        pts = POINTS[POINTS.dataset == ds].dropna(subset=['flops'])
        ax.scatter(np.maximum(pts.flops, 1), pts.rel_mae, s=10, color='#999999', zorder=2)
        for _, r in pts.iterrows():
            ax.annotate(r.model, (max(r.flops, 1), r.rel_mae), fontsize=4.2, xytext=(2, 2), textcoords='offset points',
                        color='#555555')
        for tgt, col in zip(CAFX_TARGETS[ds], ('#D55E00', '#0072B2', '#009E73', '#CC79A7')):
            g = CAFX_SWEEP[(CAFX_SWEEP.dataset == ds) & (CAFX_SWEEP.target == tgt) & (CAFX_SWEEP.router == 'learned')]
            if not len(g):
                continue
            best_mode = g.groupby('mode').val_mae.mean().idxmin(); g = g[g['mode'] == best_mode]
            c = g.groupby('budget')[['test_rate', 'test_mae']].mean().reset_index()
            ax.plot([cafx_flops(tgt, r) for r in c.test_rate], c.test_mae / PERS[ds], '-o', ms=2, lw=1, color=col,
                    label=f'CAF-X[{tgt}] ({best_mode})')
            op = CAFX_MAIN[(CAFX_MAIN.dataset == ds) & (CAFX_MAIN.target == tgt)] if len(CAFX_MAIN) else CAFX_MAIN
            if len(op):
                ax.scatter(op.MFLOPs * 1e6, op.cafx_mae / PERS[ds], marker='*', s=70, color=col, edgecolor='k', lw=.4, zorder=5)
        ax.set_xscale('log'); ax.set_xlabel('FLOPs per forecast'); ax.set_title(ds)
        lo = min(pts.rel_mae.min(), CAFX_SWEEP[CAFX_SWEEP.dataset == ds].test_mae.min() / PERS[ds])
        ax.set_ylim(lo - .005, lo + .06); ax.legend(fontsize=4.6, loc='upper right')
    np.atleast_1d(axes)[0].set_ylabel('MAE / MAE(persistence)')
    fig.suptitle('CAF-X accuracy–compute curves (★ = validation-calibrated operating point); grey = all evaluated models',
                 fontsize=6.5)
    savefig(fig, 'fig_cafx_pareto')
checkpoint('CAF-X', force=True)
stamp('CAF-X done.')


#   (1) Fair ensemble baselines: every always-on 2-model ensemble of the candidate pool is evaluated as a competitor, so
#       ensembling alone cannot be mistaken for a CAF-X gain.
#   (2) CAF-X-eff (pre-registered compute cap): per dataset, the configuration (target, budget, blend a) with the lowest
#       VALIDATION MAE subject to expected FLOPs (validation escalation rate) <= the FLOPs of the cheapest of the
#       validation-top-3 single deep models. Test days are evaluated once for the selected configuration.
import itertools
EFF_POOL_K = 4                                     # cheapest-to-wrap pool: validation-top-K deep models with <= 2 MFLOPs
EFF_POOL_MAX_FLOPS = 2.0e6

def _val_rank(ds):
    return sorted(_DEEP_BASE, key=lambda m: _seed_val_mae(ds, m))

EFF_POOL = {ds: [m for m in _val_rank(ds) if FLOPS[m] <= EFF_POOL_MAX_FLOPS][:EFF_POOL_K] for ds in DS}
EFF_TARGETS = {ds: EFF_POOL[ds] + [' + '.join(p) for p in itertools.combinations(EFF_POOL[ds], 2)] for ds in DS}
FLOP_CAP = {ds: min(FLOPS[m] for m in _val_rank(ds)[:3]) for ds in DS}
for ts in EFF_TARGETS.values():
    for t in ts:
        FLOPS.setdefault(t, sum(FLOPS[q] for q in _parts(t)))
print('Compute-constrained pool (validation-ranked, <= 2 MFLOPs):', EFF_POOL)
print('Pre-registered FLOP cap = cheapest of the validation-top-3 single models:',
      {ds: f'{v / 1e6:.3f} MFLOPs' for ds, v in FLOP_CAP.items()})

stamp('CAF-X-eff: router fits for the compute-constrained pool ...')
for ds in DS:
    for tgt in EFF_TARGETS[ds]:
        for seed in SEEDS:
            if (ds, tgt, seed) in CAFX or any(('main', ds, q, seed) not in DEEP_PREDS for q in _parts(tgt) + [PROPOSED]):
                continue
            if remaining_min() < 12 and not (HIST / f'cafx_{slug(ds)}_{slug(tgt)}_s{seed}.json').exists():
                print(f'CAF-X-eff {ds}/{tgt}/s{seed} skipped by the time guard'); continue
            try:
                CAFX[(ds, tgt, seed)] = _cafx_one(ds, tgt, seed)
            except Exception as e:
                print(f'CAF-X-eff {ds}/{tgt}/s{seed} failed: {type(e).__name__}: {e}')
    checkpoint(f'CAF-X-eff {ds}')

def _cfg_preds(ds, tgt, seed, budget, mode):
    """test forecasts of one CAF-X configuration, rebuilt from cached E1, deep forecasts, router scores and tau"""
    r = next(r for r in CAFX[(ds, tgt, seed)]['rows'] if r['router'] == 'learned' and r['budget'] == budget and r['mode'] == mode)
    z = np.load(PREDS / f'cafx_{slug(ds)}_{slug(tgt)}_s{seed}.npz')
    e1 = _caf_cheap(ds, seed)['test']['e1']; deep = _deep_preds(ds, tgt, seed)['test']; a = float(mode.split('=')[1])
    esc = z['score_test'] > r['tau']
    return np.where(esc[:, None], a * deep + (1 - a) * e1, e1), float(esc.mean())

EFF_SEL, EFF_ROWS = {}, []
for ds in DS:
    cand = []
    for tgt in EFF_TARGETS[ds] + CAFX_TARGETS[ds]:
        ks = [k for k in CAFX if k[0] == ds and k[1] == tgt]
        if len(ks) < len(SEEDS):
            continue
        rows = pd.DataFrame([dict(r, seed=k[2]) for k in ks for r in CAFX[k]['rows'] if r['router'] == 'learned'])
        g = rows.groupby(['budget', 'mode'])[['val_mae', 'val_rate']].mean().reset_index()
        for r in g.itertuples():
            fl = cafx_flops(tgt, r.val_rate)
            cand.append(dict(dataset=ds, target=tgt, budget=r.budget, mode=r.mode, val_mae=r.val_mae, val_flops=fl,
                             feasible=fl <= FLOP_CAP[ds]))
    cand = pd.DataFrame(cand); EFF_ROWS.append(cand)
    feas = cand[cand.feasible]
    if len(feas):
        EFF_SEL[ds] = feas.loc[feas.val_mae.idxmin()].to_dict()
EFF_CAND = pd.concat(EFF_ROWS) if EFF_ROWS else pd.DataFrame()
EFF_CAND.to_csv(TABLES / 'cafx_eff_candidates_validation.csv', index=False)
print('CAF-X-eff selections (validation only):'); display(pd.DataFrame(EFF_SEL).T)

def _pair_err(ds, pair):
    """seed-mean absolute error (n, H) of an always-on ensemble"""
    y = DATA[ds].splits['test'].y; a, b = _parts(pair)
    return np.mean([np.abs(.5 * (DEEP_PREDS[('main', ds, a, s)]['test'] + DEEP_PREDS[('main', ds, b, s)]['test']) - y)
                    for s in SEEDS if ('main', ds, a, s) in DEEP_PREDS and ('main', ds, b, s) in DEEP_PREDS], 0)

def _ens_cost(pair):
    ps = _parts(pair)
    return dict(MFLOPs=sum(FLOPS[q] for q in ps) / 1e6, cpu_ms=sum(EFF.loc[q, 'cpu1t_ms_b1'] for q in ps),
                J_per_M=sum(EFF.loc[q, 'J_per_M_forecasts'] for q in ps))

ENS_PAIRS = {ds: sorted({' + '.join(p) for p in itertools.combinations(_val_rank(ds)[:5], 2)} |
                        {t for t in EFF_TARGETS[ds] + CAFX_TARGETS[ds] if ' + ' in t}) for ds in DS}
FAIR, FAIR_DM = [], []
rng_f = np.random.default_rng(13)
for ds in DS:
    D = DATA[ds]; y = D.splits['test'].y; day = D.splits['test'].day
    comp = {}
    for m in [m for m in ORDER if m not in ('CAF-E1',)]:
        if m in CLASSICAL_MODELS and m != 'EWMA':
            continue
        comp[m] = (np.mean([np.abs(p - y) for p in all_preds(ds, m, 'test')], 0), compute_of(ds, m))
    for pr in ENS_PAIRS[ds]:
        comp[f'Ens({pr})'] = (_pair_err(ds, pr), _ens_cost(pr))
    if ds in EFF_SEL:
        s = EFF_SEL[ds]; outs = [_cfg_preds(ds, s['target'], sd, s['budget'], s['mode']) for sd in SEEDS]
        rate = float(np.mean([o[1] for o in outs])); fl = cafx_flops(s['target'], rate)
        cpu = EFF.loc['CAF-E1', 'cpu1t_ms_b1'] + rate * sum(EFF.loc[q, 'cpu1t_ms_b1'] for q in _parts(s['target']))
        en = EFF.loc['CAF-E1', 'J_per_M_forecasts'] + rate * sum(EFF.loc[q, 'J_per_M_forecasts'] for q in _parts(s['target']))
        comp['CAF-X-eff'] = (np.mean([np.abs(o[0] - y) for o in outs], 0), dict(MFLOPs=fl / 1e6, cpu_ms=cpu, J_per_M=en))
    t, ks = head_keys(ds)
    if ks:
        comp['CAF-X-acc'] = (np.mean([np.abs(CAFX[k]['test'] - y) for k in ks], 0), compute_of(ds, 'CAF-X'))
    for m, (e, c) in comp.items():
        FAIR.append(dict(dataset=ds, model=m, MAE=float(e.mean()), p99=float(np.percentile(e.mean(1), 99)),
                         MFLOPs=c['MFLOPs'], cpu_ms=c['cpu_ms'], J_per_M=c['J_per_M'],
                         kind='CAF-X' if m.startswith('CAF-X') else 'ensemble' if m.startswith('Ens(') else 'single'))
    for prop in ('CAF-X-eff', 'CAF-X-acc'):
        if prop not in comp:
            continue
        e_p = comp[prop][0].mean(1); rr = []
        for b, (e_b, _) in comp.items():
            if b.startswith('CAF-X'):
                continue
            e_b = e_b.mean(1); st_, p = diebold_mariano(e_p, e_b, LAG)
            bs = block_bootstrap_rel(e_p, e_b, day, LAG, BOOTSTRAP_REPS, rng_f)
            rr.append(dict(dataset=ds, proposed=prop, baseline=b, mae_change_pct=100 * (e_p.mean() / e_b.mean() - 1),
                           ci_lo_pct=-100 * np.percentile(bs, 97.5), ci_hi_pct=-100 * np.percentile(bs, 2.5), p=p, dm=st_))
        for r, a in zip(rr, holm([r['p'] for r in rr])):
            r['p_holm'] = a; r['better'] = bool(a < ALPHA and r['dm'] < 0); r['worse'] = bool(a < ALPHA and r['dm'] > 0)
        FAIR_DM += rr
FAIR = pd.DataFrame(FAIR); FAIR_DM = pd.DataFrame(FAIR_DM)
_bsm = FAIR[FAIR.kind == 'single'].groupby('dataset').MAE.min()
FAIR['rel_to_best_single_pct'] = 100 * (FAIR.MAE / FAIR.dataset.map(_bsm) - 1)
FAIR.to_csv(TABLES / 'table14_fair_ensembles_and_cafx_eff.csv', index=False)
FAIR_DM.to_csv(TABLES / 'table14_fair_dm.csv', index=False)

def _pareto(g):
    g = g.sort_values(['MFLOPs', 'MAE']); best, keep = np.inf, []
    for r in g.itertuples():
        if r.MAE < best - 1e-9:
            keep.append(r.model); best = r.MAE
    return keep
FAIR_PARETO = {ds: _pareto(FAIR[(FAIR.dataset == ds) & FAIR.MFLOPs.notna()]) for ds in DS}
for ds in DS:
    g = FAIR[FAIR.dataset == ds].sort_values('MAE')
    print(f'\n{ds}: Pareto front (MFLOPs vs MAE, incl. all ensembles): {FAIR_PARETO[ds]}')
    display(g.head(12).round(3))
if len(FAIR_DM):
    display(FAIR_DM[FAIR_DM.proposed == 'CAF-X-eff'].sort_values(['dataset', 'mae_change_pct']).groupby('dataset').head(6).round(4))

t14 = []
for ds in DS:
    g = FAIR[FAIR.dataset == ds].set_index('model'); bs = g[g.kind == 'single'].MAE.idxmin()
    be = g[g.kind == 'ensemble'].MAE.idxmin() if (g.kind == 'ensemble').any() else None
    for m in [x for x in ('CAF-X-eff', bs, 'CAF-X-acc', be) if x is not None and x in g.index]:
        d = FAIR_DM[(FAIR_DM.dataset == ds) & (FAIR_DM.proposed == 'CAF-X-eff') & (FAIR_DM.baseline == m)] if len(FAIR_DM) else []
        t14.append({'Dataset': ds, 'Model': m, 'MAE': f'{g.loc[m, "MAE"]:.2f}', 'MFLOPs': f'{g.loc[m, "MFLOPs"]:.3f}',
                    'CPU ms': f'{g.loc[m, "cpu_ms"]:.2f}', 'J/1M': f'{g.loc[m, "J_per_M"]:.0f}',
                    'CAF-X-eff vs. (%) [95% CI]': (f'{d.mae_change_pct.iloc[0]:+.2f} [{d.ci_lo_pct.iloc[0]:+.2f}, '
                                                   f'{d.ci_hi_pct.iloc[0]:+.2f}]{"*" if d.better.iloc[0] or d.worse.iloc[0] else ""}')
                    if len(d) else '--'})
T14 = pd.DataFrame(t14).set_index(['Dataset', 'Model'])
if len(T14):
    display(T14)
    t = T14.copy(); t.index = [f'{a} / {b}' for a, b in t.index]; t.index.name = 'Dataset / model'
    to_latex(t, TABLES / 'table14_cafx_eff.tex',
             'Compute-constrained CAF-X (CAF-X-eff, FLOP cap = cheapest validation-top-3 single model, configuration chosen '
             'on validation) vs. the best single model, the accuracy-mode CAF-X and the best always-on 2-model ensemble. '
             '* = DM (HLN, HAC) significant after Holm correction over all competitors incl. all ensembles.', 'tab:cafxeff')

FIG_LOG2 = []
def figS10(name):
    fig, axes = plt.subplots(1, len(DS), figsize=(COL2, 2.6), layout='constrained')
    for ax, ds in zip(np.atleast_1d(axes), DS):
        g = FAIR[(FAIR.dataset == ds) & FAIR.MFLOPs.notna()]; best = g[g.kind == 'single'].MAE.min()
        for kind, mk, col in (('single', 'o', '#999999'), ('ensemble', 's', '#56B4E9'), ('CAF-X', '*', '#D55E00')):
            h = g[g.kind == kind]
            ax.scatter(np.maximum(h.MFLOPs, 1e-4), 100 * (h.MAE / best - 1), marker=mk, s=60 if kind == 'CAF-X' else 14,
                       color=col, edgecolor='k' if kind == 'CAF-X' else 'none', lw=.4, label=kind, zorder=5 if kind == 'CAF-X' else 2)
        for r in g.itertuples():
            if r.kind != 'ensemble' or r.model in FAIR_PARETO[ds]:
                ax.annotate(r.model.replace('Ens(', '').replace(')', ''), (max(r.MFLOPs, 1e-4), 100 * (r.MAE / best - 1)),
                            fontsize=3.8, xytext=(2, 2), textcoords='offset points')
        pf = g.set_index('model').loc[FAIR_PARETO[ds]]
        ax.step(np.maximum(pf.MFLOPs, 1e-4), 100 * (pf.MAE / best - 1), where='post', color='k', lw=.6, ls='--')
        ax.axvline(FLOP_CAP[ds] / 1e6, color='#B22222', lw=.6, ls=':'); ax.axhline(0, color='#888888', lw=.4)
        ax.set_xscale('log'); ax.set_ylim(-.6, 2.0); ax.set_title(ds); ax.set_xlabel('MFLOPs per forecast')
    np.atleast_1d(axes)[0].set_ylabel('MAE vs best single model (%)'); np.atleast_1d(axes)[-1].legend(fontsize=5)
    fig.suptitle('Fair Pareto view incl. every always-on 2-model ensemble; red dotted = pre-registered FLOP cap', fontsize=6.5)
    savefig(fig, name)
try:
    figS10('figS10_fair_pareto_ensembles'); FIG_LOG2.append(('figS10_fair_pareto_ensembles', 'ok'))
except Exception as e:
    plt.close('all'); FIG_LOG2.append(('figS10_fair_pareto_ensembles', f'{type(e).__name__}: {e}')); print('figS10 skipped:', e)
checkpoint('CAF-X-eff', force=True)
stamp('CAF-X-eff and fair ensemble comparison done.')
