"""
Evidence-based claims check (accuracy + systems), run manifest, and export.
Corresponds to Cell 41 of cmsnts_latest.ipynb.
"""

import os
import json
import numpy as np
import pandas as pd
from config import (
    PROPOSED, DS, RUN, TABLES, NONINF_MARGIN, CONFIG, RUN_MINUTES, TIME_BUDGET_HOURS,
    CAPTURE_SECONDS, SEEDS, LOOKBACK, HORIZON, BIN_MS, ROOT
)
from utils.helpers import checkpoint, stamp

claims = []
COMPETITORS = [m for m in ORDER if m != PROPOSED and m not in CASCADE_MODELS]
for ds in DS:
    if PROPOSED in AVG_MAE.index and ds in AVG_MAE:
        col = AVG_MAE[ds].drop(CASCADE_MODELS, errors='ignore').sort_values(); best = col.index[0]
        claims.append(dict(claim=f'{ds}: {PROPOSED} has the lowest horizon-averaged test MAE', holds=best == PROPOSED,
                           evidence=f'best={best} ({col.iloc[0]:.3f}); {PROPOSED}={col[PROPOSED]:.3f}; strongest baseline '
                                    f'{BEST_BASELINE[ds]}={AVG_MAE.loc[BEST_BASELINE[ds], ds]:.3f} Mbit/s'))
        d = DM[(DM.dataset == ds) & DM.baseline.isin(COMPETITORS)] if len(DM) else pd.DataFrame()
        if len(d):
            claims.append(dict(claim=f'{ds}: {PROPOSED} significantly better than every baseline (DM, Holm, α={ALPHA})',
                               holds=bool(d.significant.all()),
                               evidence=f'{int(d.significant.sum())}/{len(d)} significant; not: '
                                        f'{", ".join(d.loc[~d.significant, "baseline"]) or "-"}'))
R_FAIR = MEAN_RANK.drop(CASCADE_MODELS, errors='ignore') if len(MEAN_RANK) else MEAN_RANK
if len(R_FAIR):
    claims.append(dict(claim='Best mean rank across tasks (Friedman/Nemenyi)', holds=R_FAIR.index[0] == PROPOSED,
                       evidence=(f'ranks: {", ".join(f"{m} {v:.2f}" for m, v in R_FAIR.head(4).items())}; CD={CD:.2f}; '
                                 f'Friedman p={fried.pvalue:.1e}') if fried else 'n/a'))
    if PROPOSED in R_FAIR and len(R_FAIR) > 1:
        second = R_FAIR.drop(PROPOSED).index[0]
        claims.append(dict(claim='Rank difference to best competitor exceeds Nemenyi CD',
                           holds=bool(R_FAIR[second] - R_FAIR.get(PROPOSED, np.inf) > CD),
                           evidence=f'{second}: {R_FAIR[second]:.2f} vs {PROPOSED} {R_FAIR.get(PROPOSED, np.nan):.2f}'))

# --- systems claims: non-inferiority of the cascade, compute saving, Pareto optimality, router quality ---
NONINF = []
rng_ni = np.random.default_rng(7)
for ds, D in DATA.items():
    if not test_preds('main', ds, 'CAF-adaptive') or not test_preds('main', ds, PROPOSED):
        continue
    e_ad, e_full = abs_err('main', D, 'CAF-adaptive'), abs_err('main', D, PROPOSED)
    bs = block_bootstrap_rel(e_ad, e_full, D.splits['test'].day, LAG, BOOTSTRAP_REPS, rng_ni)
    lo, hi = np.percentile(bs, [2.5, 97.5]); rate = OP_RATE.get(ds, np.nan)
    b = BEST_BASELINE[ds]; e_b = abs_err('main', D, b); st_, p = diebold_mariano(e_ad, e_b, LAG)
    bs_b = block_bootstrap_rel(e_ad, e_b, D.splits['test'].day, LAG, BOOTSTRAP_REPS, rng_ni); lo_b = np.percentile(bs_b, 2.5)
    fl_b = FLOPS.get(b, CLASSICAL_FLOPS.get(b, np.nan))
    NONINF.append(dict(dataset=ds, rel_change_pct=-100 * (1 - e_ad.mean() / e_full.mean()), ci_lo_pct=-100 * hi,
                       ci_hi_pct=-100 * lo, escalation_rate=rate, flops_pct_of_full=100 * cascade_flops(rate) / CAF_FLOPS['full'],
                       vs_best_baseline=b, impr_vs_best_pct=100 * (1 - e_ad.mean() / e_b.mean()), dm_p=p,
                       better_than_best=bool(st_ < 0 and p < ALPHA), vs_best_ci_hi_pct=-100 * lo_b,
                       flop_reduction_vs_best_x=fl_b / max(cascade_flops(rate), 1.0)))
NONINF = pd.DataFrame(NONINF)
if len(NONINF):
    display(NONINF.round(3)); NONINF.to_csv(TABLES / 'cascade_noninferiority.csv', index=False)
    for r in NONINF.itertuples():
        claims.append(dict(claim=f'{r.dataset}: CAF-adaptive non-inferior to full CAF (margin {100 * NONINF_MARGIN:.0f}%, '
                                 f'95% block-bootstrap CI)', holds=bool(r.ci_hi_pct < 100 * NONINF_MARGIN),
                           evidence=f'MAE change {r.rel_change_pct:+.2f}% [{r.ci_lo_pct:+.2f}, {r.ci_hi_pct:+.2f}] at '
                                    f'{100 * r.escalation_rate:.0f}% escalation = {r.flops_pct_of_full:.0f}% of full FLOPs'))
        claims.append(dict(claim=f'{r.dataset}: CAF-adaptive significantly better than strongest baseline ({r.vs_best_baseline})',
                           holds=r.better_than_best, evidence=f'{r.impr_vs_best_pct:+.1f}% MAE, DM p={r.dm_p:.1e}'))
        claims.append(dict(claim=f'{r.dataset}: CAF-adaptive non-inferior to the strongest baseline ({r.vs_best_baseline}, '
                                 f'margin {100 * NONINF_MARGIN:.0f}%, 95% block-bootstrap CI)',
                           holds=bool(r.vs_best_ci_hi_pct < 100 * NONINF_MARGIN),
                           evidence=f'MAE change {-r.impr_vs_best_pct:+.2f}% (CI upper {r.vs_best_ci_hi_pct:+.2f}%) at '
                                    f'{r.flop_reduction_vs_best_x:,.0f}x fewer FLOPs'))
for ds, front in PARETO.items():
    claims.append(dict(claim=f'{ds}: CAF-adaptive lies on the FLOPs-MAE Pareto front of all evaluated models',
                       holds='CAF-adaptive' in front, evidence='front: ' + ' -> '.join(front)))

# --- compute headroom: what the deep SOTA buys over the best O(1) smoother, and how much of it CAF-adaptive recovers ---
HEADROOM = []
for ds in DS:
    P = POINTS[POINTS.dataset == ds].set_index('model')
    mae = {m: direct_mae(ds, m) for m in P.index}
    cheap = [m for m in ('Persistence', 'MovingAvg', 'EWMA') if np.isfinite(mae.get(m, np.nan))]
    deep = [m for m in COMPETITORS if m in MODEL_SPECS and np.isfinite(mae.get(m, np.nan))]
    if not cheap or not deep or not np.isfinite(mae.get('CAF-adaptive', np.nan)):
        continue
    c = min(cheap, key=mae.get); s = min(deep, key=mae.get); a = 'CAF-adaptive'
    gap = mae[c] - mae[s]; fl_a = cascade_flops(OP_RATE.get(ds, 0.0))
    HEADROOM.append(dict(dataset=ds, best_smoother=c, smoother_mae=mae[c], best_deep=s, deep_mae=mae[s],
                         deep_gain_over_smoother_pct=100 * gap / mae[c], deep_MFLOPs=FLOPS[s] / 1e6,
                         caf_adaptive_mae=mae[a], caf_adaptive_MFLOPs=fl_a / 1e6,
                         headroom_recovered_pct=100 * (mae[c] - mae[a]) / gap if gap > 0 else np.nan,
                         flop_reduction_vs_deep_x=FLOPS[s] / max(fl_a, 1.0)))
HEADROOM = pd.DataFrame(HEADROOM)
if len(HEADROOM):
    print('Compute headroom (Table XI): deep-SOTA gain over the best O(1) smoother and the share CAF-adaptive recovers')
    display(HEADROOM.round(3)); HEADROOM.to_csv(TABLES / 'table11_compute_headroom.csv', index=False)
    to_latex(HEADROOM.round(3).astype(str), TABLES / 'table11_compute_headroom.tex',
             'Compute headroom: MAE gain of the best deep model over the best O(1) smoother, and the share of that gain '
             'recovered by CAF-adaptive at its validation-calibrated operating point.', 'tab:headroom')
    for r in HEADROOM.itertuples():
        claims.append(dict(claim=f'{r.dataset}: CAF-adaptive recovers >=50% of the smoother->deep-SOTA headroom',
                           holds=bool(r.headroom_recovered_pct >= 50),
                           evidence=f'{r.best_smoother} {r.smoother_mae:.2f} -> {r.best_deep} {r.deep_mae:.2f} '
                                    f'({r.deep_gain_over_smoother_pct:.1f}% gain, {r.deep_MFLOPs:.2f} MFLOPs); CAF-adaptive '
                                    f'{r.caf_adaptive_mae:.2f} ({r.headroom_recovered_pct:.0f}% recovered, '
                                    f'{r.flop_reduction_vs_deep_x:,.0f}x fewer FLOPs)'))
if 'CAF-adaptive' in EFF.index:
    lpc = EFF.loc['CAF-adaptive', 'links_per_core']
    claims.append(dict(claim='CAF-adaptive meets the 10 ms cadence on one CPU core (p99 latency < bin)',
                       holds=bool(EFF.loc['CAF-adaptive', 'cpu1t_p99_ms'] < BIN_MS),
                       evidence=f'p50 {EFF.loc["CAF-adaptive", "cpu1t_ms_b1"]:.2f} ms, p99 {EFF.loc["CAF-adaptive", "cpu1t_p99_ms"]:.2f} ms; '
                                f'{lpc:.0f} links/core (expected-latency estimate)'))
if len(CASC):
    rc = curve[curve.budget == .2].pivot_table(index='dataset', columns='router', values='test_mae')
    # routing is only meaningful where the deep exit improves on the cheap exit (E1 -> E2 gap > 0.2 %)
    _e1 = curve[(curve.router == 'learned') & (curve.budget == 0)].set_index('dataset').test_mae
    _e2 = curve[(curve.router == 'learned') & (curve.budget == 1.0)].set_index('dataset').test_mae
    rc = rc.loc[[ds for ds in rc.index if ds in _e1 and ds in _e2 and _e1[ds] - _e2[ds] > 2e-3 * _e1[ds]]]
    if len(rc) and {'learned', 'volatility', 'random'} <= set(rc.columns):
        claims.append(dict(claim='Learned router beats volatility and random routing at 20% budget on every dataset '
                                 'where the deep exit helps',
                           holds=bool(((rc.learned < rc.volatility) & (rc.learned < rc.random)).all()),
                           evidence='; '.join(f'{ds}: L {r.learned:.3f} / V {r.volatility:.3f} / R {r.random:.3f}'
                                              for ds, r in rc.iterrows())))
if len(ABL_DELTA):
    for v in ABL_DELTA.index:
        claims.append(dict(claim=f'Ablation "{v}" increases MAE on every dataset', holds=bool((ABL_DELTA.loc[v] > 0).all()),
                           evidence=', '.join(f'{ds} {ABL_DELTA.loc[v, ds]:+.1f}%' for ds in ABL_DELTA.columns)))
if len(TRANSFER):
    tm = TR[TR.model == PROPOSED]
    claims.append(dict(claim=f'{PROPOSED} transfers zero-shot with positive skill on all dataset pairs',
                       holds=bool(len(tm) and (tm.groupby(['source', 'target']).skill.mean() > 0).all()),
                       evidence=f'min pair skill {tm.groupby(["source", "target"]).skill.mean().min():.3f}' if len(tm) else 'n/a'))
# --- CAF-X (model-agnostic compute-adaptive serving, cell 20c) ---
if 'CAFX_MAIN' in globals() and len(CAFX_MAIN):
    top = CAFX_MAIN[CAFX_MAIN.headline]
    for r in top.itertuples():
        d = CAFX_DM[CAFX_DM.dataset == r.dataset] if len(CAFX_DM) else pd.DataFrame()
        claims.append(dict(claim=f'{r.dataset}: CAF-X[{r.target}] has the lowest test MAE of all evaluated models',
                           holds=bool(r.cafx_mae < AVG_MAE[r.dataset].drop(['CAF-adaptive', 'CAF-E1'], errors='ignore').min()),
                           evidence=f'CAF-X {r.cafx_mae:.3f} vs strongest {r.strongest_test_baseline} {r.strongest_mae:.3f} '
                                    f'({r.vs_strongest_pct:+.2f}%)'))
        if len(d):
            tg = d[d.baseline == r.target]
            if len(tg):
                claims.append(dict(claim=f'{r.dataset}: CAF-X significantly improves the model it wraps ({r.target}) while '
                                         f'running it on only {100 * r.escalation_rate:.0f}% of windows',
                                   holds=bool(tg.better.iloc[0]),
                                   evidence=f'{tg.rel_mae_change_pct.iloc[0]:+.2f}% [{tg.ci_lo_pct.iloc[0]:+.2f}, '
                                            f'{tg.ci_hi_pct.iloc[0]:+.2f}], p_Holm={tg.p_holm.iloc[0]:.1e}; '
                                            f'{r.flop_reduction_vs_target_x:.1f}x fewer FLOPs'))
            claims.append(dict(claim=f'{r.dataset}: CAF-X not significantly worse than any model (DM, Holm)',
                               holds=not bool(d.worse.any()),
                               evidence=f'significantly better than {int(d.better.sum())}/{len(d)}; worse than: '
                                        f'{", ".join(d.loc[d.worse, "baseline"]) or "-"}'))
            sb = d[d.baseline == r.strongest_test_baseline]
            if len(sb):
                claims.append(dict(claim=f'{r.dataset}: CAF-X non-inferior to the strongest baseline '
                                         f'({r.strongest_test_baseline}, margin 0.5%, 95% CI)',
                                   holds=bool(sb.ci_hi_pct.iloc[0] < 0.5),
                                   evidence=f'{sb.rel_mae_change_pct.iloc[0]:+.2f}% [{sb.ci_lo_pct.iloc[0]:+.2f}, '
                                            f'{sb.ci_hi_pct.iloc[0]:+.2f}] at {r.flop_reduction_vs_strongest_x:.1f}x fewer FLOPs'))
    g = CAFX_MAIN.groupby('dataset').apply(lambda x: bool((x.vs_target_pct < 0).all()))
    claims.append(dict(claim='CAF-X is model-agnostic: it lowers test MAE of every wrapped deep model on every dataset',
                       holds=bool(g.all()),
                       evidence='; '.join(f'{r.dataset}/{r.target} {r.vs_target_pct:+.2f}% @ {100 * r.escalation_rate:.0f}%'
                                          for r in CAFX_MAIN.itertuples())))
    if 'CAFX_SWEEP' in globals() and len(CAFX_SWEEP):
        sw = CAFX_SWEEP.merge(top[['dataset', 'target']], on=['dataset', 'target'])
        sw = sw[sw.budget == 0.2]
        sw = sw.groupby(['dataset', 'router', 'mode']).test_mae.mean().groupby(['dataset', 'router']).min().unstack()
        if {'learned', 'volatility', 'random'} <= set(sw.columns):
            claims.append(dict(claim='CAF-X learned router beats volatility and random routing at a 20% budget on every dataset',
                               holds=bool(((sw.learned < sw.volatility) & (sw.learned < sw.random)).all()),
                               evidence=', '.join(f'{d}: L {r.learned:.2f} / V {r.volatility:.2f} / R {r.random:.2f}'
                                                  for d, r in sw.iterrows())))
if 'SCORE' in globals() and len(SCORE) and 'CAF-X' in set(SCORE.model):
    acc = ['MAE', 'RMSE', 'p99_abs_err', 'burst_MAE']; cmpm = ['MFLOPs', 'cpu_ms', 'J_per_M']
    sr = SCORE_RANK[SCORE_RANK.model == 'CAF-X'].set_index('dataset')
    deep_only = SCORE[SCORE.MAE <= 1.01 * SCORE.groupby('dataset').MAE.transform('min')]   # accuracy tier: within 1% of best
    dr = deep_only.groupby('dataset')[cmpm].rank(method='min'); dr['dataset'] = deep_only.dataset; dr['model'] = deep_only.model
    drx = dr[dr.model == 'CAF-X'].set_index('dataset')
    drx = drx.reindex(DS)
    claims.append(dict(claim='CAF-X ranks first on at least one accuracy metric (MAE/RMSE/p99/burst) on every dataset',
                       holds=bool((sr[acc].min(1) == 1).all()),
                       evidence='; '.join(f'{d}: ' + ', '.join(f'{m} #{int(sr.loc[d, m])}' for m in acc) for d in sr.index)))
    claims.append(dict(claim='Among all models within 1% of the best test MAE, CAF-X is the cheapest (FLOPs, CPU latency, energy) on every dataset',
                       holds=bool((drx[cmpm] == 1).all().all()),
                       evidence='; '.join(f'{d}: ' + (', '.join(f'{m} #{drx.loc[d, m]:.0f}' for m in cmpm) if np.isfinite(drx.loc[d, 'MFLOPs']) else 'CAF-X not within 1%') for d in drx.index)))
# --- fair comparison incl. ensembles + compute-constrained CAF-X (cell 20e) ---
if 'FAIR' in globals() and len(FAIR):
    for ds in DS:
        g = FAIR[FAIR.dataset == ds].set_index('model')
        if 'CAF-X-eff' not in g.index:
            continue
        e = g.loc['CAF-X-eff']; sing = g[g.kind == 'single']; bs = sing.MAE.idxmin()
        d = FAIR_DM[(FAIR_DM.dataset == ds) & (FAIR_DM.proposed == 'CAF-X-eff')]
        db = d[d.baseline == bs]
        claims.append(dict(claim=f'{ds}: CAF-X-eff (FLOP cap pre-registered) is non-inferior to the best single SOTA model '
                                 f'({bs}) within 0.5% MAE (95% CI) at fewer FLOPs',
                           holds=bool(len(db) and db.ci_hi_pct.iloc[0] < 0.5 and e.MFLOPs < sing.loc[bs, 'MFLOPs']),
                           evidence=(f'{db.mae_change_pct.iloc[0]:+.2f}% [{db.ci_lo_pct.iloc[0]:+.2f}, {db.ci_hi_pct.iloc[0]:+.2f}]; '
                                     f'{e.MFLOPs:.3f} vs {sing.loc[bs, "MFLOPs"]:.3f} MFLOPs '
                                     f'({sing.loc[bs, "MFLOPs"] / max(e.MFLOPs, 1e-9):.1f}x), CPU {e.cpu_ms:.2f} vs '
                                     f'{sing.loc[bs, "cpu_ms"]:.2f} ms') if len(db) else 'n/a'))
        cheaper = g[(g.MFLOPs <= e.MFLOPs) & (g.index != 'CAF-X-eff') & g.MFLOPs.notna()]
        claims.append(dict(claim=f'{ds}: CAF-X-eff has lower test MAE than every model and ensemble with equal or fewer FLOPs',
                           holds=bool((cheaper.MAE > e.MAE).all()),
                           evidence=f'CAF-X-eff {e.MAE:.3f} @ {e.MFLOPs:.3f} MFLOPs; best cheaper: '
                                    + (f'{cheaper.MAE.idxmin()} {cheaper.MAE.min():.3f}' if len(cheaper) else '-')))
        claims.append(dict(claim=f'{ds}: CAF-X-eff lies on the MAE-FLOPs Pareto front of all models incl. every 2-model ensemble',
                           holds='CAF-X-eff' in FAIR_PARETO[ds], evidence=' -> '.join(FAIR_PARETO[ds])))
        claims.append(dict(claim=f'{ds}: CAF-X-eff not significantly worse than any single deep model (DM, Holm)',
                           holds=not bool(d[~d.baseline.str.startswith('Ens(')].worse.any()),
                           evidence='worse than: ' + (', '.join(d.loc[d.worse & ~d.baseline.str.startswith('Ens('), 'baseline']) or '-')
                                    + f'; better than {int(d.better.sum())}/{len(d)} competitors'))
        if 'CAF-X-acc' in g.index and (g.kind == 'ensemble').any():
            be = g[g.kind == 'ensemble'].MAE.idxmin()
            claims.append(dict(claim=f'{ds}: accuracy-mode CAF-X beats the best always-on 2-model ensemble ({be}) '
                                     '(fairness check: ensembling alone does not explain the gain)',
                               holds=bool(g.loc['CAF-X-acc', 'MAE'] < g.loc[be, 'MAE']),
                               evidence=f'{g.loc["CAF-X-acc", "MAE"]:.3f} @ {g.loc["CAF-X-acc", "MFLOPs"]:.2f} MFLOPs vs '
                                        f'{g.loc[be, "MAE"]:.3f} @ {g.loc[be, "MFLOPs"]:.2f} MFLOPs'))
# --- edge efficiency + data scale (cells 13c / 20f) ---
if 'EDGE' in globals() and len(EDGE):
    sota = [m for m in ('TimeXer', 'TimeMixer', 'TimesNet', 'TSMixer', 'iTransformer') if m in EDGE.index]
    for ds in DS:
        k = f'CAF-X-eff ({ds})'
        if k not in EDGE.index:
            continue
        e = EDGE.loc[k]
        for c, lab in (('MFLOPs', 'FLOPs'), ('cpu1_p50_ms', 'expected CPU latency'), ('int8_p50_ms', 'expected INT8 latency'),
                       ('core_s_per_1M', 'CPU core-seconds per 10^6 forecasts')):
            b = FAIR[(FAIR.dataset == ds) & (FAIR.kind == 'single')].set_index('model').MAE.idxmin() if 'FAIR' in globals() else None
            if b in EDGE.index:
                claims.append(dict(claim=f'{ds}: CAF-X-eff needs less {lab} than the most accurate single SOTA model ({b})',
                                   holds=bool(e[c] < EDGE.loc[b, c]),
                                   evidence=f'{e[c]:.4g} vs {EDGE.loc[b, c]:.4g} ({EDGE.loc[b, c] / max(e[c], 1e-12):.1f}x)'))
    if 'CAF-E1' in EDGE.index and sota:
        claims.append(dict(claim='CAF-E1 meets the 10 ms deadline on every projected embedded CPU (Pi 4, Pi 5, Jetson Nano)',
                           holds=bool((EDGE.loc['CAF-E1', [c for c in EDGE.columns if c.startswith('proj_ms')]] < BIN_MS).all()),
                           evidence=', '.join(f'{c[8:]} {EDGE.loc["CAF-E1", c]:.2f} ms' for c in EDGE.columns if c.startswith('proj_ms'))))
        claims.append(dict(claim='CAF-E1 serves more links per CPU core than every SOTA model while staying within 1% MAE of the best',
                           holds=bool((EDGE.loc['CAF-E1', 'links_per_core'] > EDGE.loc[sota, 'links_per_core']).all()
                                      and EDGE.loc['CAF-E1', 'MAE_above_best_pct'] < 1.0),
                           evidence=f'CAF-E1 {EDGE.loc["CAF-E1", "links_per_core"]:.1f} links/core, '
                                    f'{EDGE.loc["CAF-E1", "MAE_above_best_pct"]:+.2f}% MAE; best SOTA '
                                    f'{EDGE.loc[sota, "links_per_core"].max():.1f} links/core'))
if 'LOW' in globals() and len(LOW):
    for ds in LOW.dataset.unique():
        g = LOW[LOW.dataset == ds].set_index('model')
        claims.append(dict(claim=f'{ds}: with ONE training day, CAF or CAF-E1 has the lowest test MAE',
                           holds=bool(g.mae_1day.idxmin() in (PROPOSED, 'CAF-E1')),
                           evidence=', '.join(f'{m} {v:.2f}' for m, v in g.mae_1day.sort_values().items())))
    dg = LOW.groupby('model').degradation_pct.mean()
    claims.append(dict(claim='CAF-E1 degrades least when training data shrinks from 3 days to 1 day (mean over datasets)',
                       holds=bool(dg.idxmin() == 'CAF-E1'), evidence=', '.join(f'{m} {v:+.2f}%' for m, v in dg.sort_values().items())))
# --- theory validation (cell 20g) ---
if 'TH1S' in globals() and len(TH1S):
    ex = TH1S[[m for m in TH1S.index if m.startswith('CAF-E')]]
    claims.append(dict(claim='Theorem 1 verified: every CAF exit (E0, E1, E2) is load-scale equivariant (max rel. error < 1e-3)',
                       holds=bool((ex < 1e-3).all()), evidence=', '.join(f'{m} {v:.2e}' for m, v in ex.items())))
if 'TH2' in globals() and len(TH2):
    claims.append(dict(claim='Proposition 2 verified: >= 99% of escalation decisions unchanged under 1/4x-4x load scaling',
                       holds=bool((TH2.agreement >= .99).all()), evidence=f'min agreement {TH2.agreement.min():.4f}'))
if 'BL' in globals() and len(BL):
    claims.append(dict(claim='Proposition 5 verified: blended forecast never exceeds the convex bound (all days, seeds, a)',
                       holds=bool((BL.blend <= BL.bound + 1e-6).all()),
                       evidence=f'better than both ends on {100 * (BL.blend < BL.best_end).mean():.0f}% of day x seed x a'))
CLAIMS = pd.DataFrame(claims)
pd.set_option('display.max_colwidth', 200); display(CLAIMS)
CLAIMS.to_csv(TABLES / 'claims_checklist.csv', index=False)

lines = ['Auto-generated, evidence-bounded summary (edit before use):']
for ds in DS:
    if ds in AVG_MAE and PROPOSED in AVG_MAE.index:
        b = BEST_BASELINE[ds]; imp = 100 * (1 - AVG_MAE.loc[PROPOSED, ds] / AVG_MAE.loc[b, ds])
        row = DM[(DM.dataset == ds) & (DM.baseline == b)] if len(DM) else pd.DataFrame()
        sig = (f'DM p_Holm={row.p_holm.iloc[0]:.1e}, 95% CI [{100 * row.ci_lo.iloc[0]:.1f}, {100 * row.ci_hi.iloc[0]:.1f}]%'
               if len(row) else '')
        verb = 'reduces' if imp > 0 else 'does NOT reduce'
        lines.append(f'- {ds}: {PROPOSED} {verb} horizon-averaged test MAE vs the strongest baseline ({b}): '
                     f'{-imp:+.1f}% MAE ({sig}).')
for r in NONINF.itertuples() if len(NONINF) else []:
    lines.append(f'- {r.dataset}: CAF-adaptive escalates {100 * r.escalation_rate:.0f}% of windows, uses '
                 f'{r.flops_pct_of_full:.0f}% of full-model FLOPs, MAE change {r.rel_change_pct:+.2f}% '
                 f'[{r.ci_lo_pct:+.2f}, {r.ci_hi_pct:+.2f}] vs full CAF; {-r.impr_vs_best_pct:+.2f}% MAE vs '
                 f'{r.vs_best_baseline} at {r.flop_reduction_vs_best_x:,.0f}x fewer FLOPs.')
for r in HEADROOM.itertuples() if len(HEADROOM) else []:
    lines.append(f'- {r.dataset}: the best deep model ({r.best_deep}) gains {r.deep_gain_over_smoother_pct:.1f}% MAE over '
                 f'{r.best_smoother}; CAF-adaptive recovers {r.headroom_recovered_pct:.0f}% of that at '
                 f'{r.flop_reduction_vs_deep_x:,.0f}x fewer FLOPs.')
if 'CAF-adaptive' in EFF.index and PROPOSED in EFF.index:
    e = EFF.loc[['CAF-adaptive', PROPOSED]]
    lines.append(f'- Compute: CAF-adaptive {e.MFLOPs.iloc[0]:.3f} MFLOPs vs {e.MFLOPs.iloc[1]:.3f} (full); CPU 1-core '
                 f'{e.cpu1t_ms_b1.iloc[0]:.2f} vs {e.cpu1t_ms_b1.iloc[1]:.2f} ms; {e.links_per_core.iloc[0]:.0f} links/core.')
for r in (CAFX_MAIN.itertuples() if 'CAFX_MAIN' in globals() and len(CAFX_MAIN) else []):
    lines.append(f'- {r.dataset}: CAF-X[{r.target}]{" (headline)" if r.headline else ""} escalates '
                 f'{100 * r.escalation_rate:.0f}% of windows ({r.mode}); MAE {r.cafx_mae:.2f} = {r.vs_target_pct:+.2f}% '
                 f'vs {r.target} alone and {r.vs_strongest_pct:+.2f}% vs the strongest baseline '
                 f'({r.strongest_test_baseline}) at {r.flop_reduction_vs_target_x:.1f}x fewer FLOPs than {r.target}; '
                 f'expected CPU {r.cpu_ms_expected:.2f} ms.')
if 'WINS' in globals() and 'CAF-X' in WINS.index:
    lines.append('- Multi-metric scoreboard: CAF-X ranks first on ' + ', '.join(f'{c} ({int(v)}/{len(DS)})' for c, v in WINS.loc['CAF-X'].items() if v) + '.')
if 'FAIR' in globals() and len(FAIR):
    for ds in DS:
        g = FAIR[FAIR.dataset == ds].set_index('model')
        if 'CAF-X-eff' in g.index:
            bs = g[g.kind == 'single'].MAE.idxmin(); e = g.loc['CAF-X-eff']
            lines.append(f'- {ds}: CAF-X-eff[{EFF_SEL[ds]["target"]}] MAE {e.MAE:.2f} vs best single {bs} {g.loc[bs, "MAE"]:.2f} '
                         f'({100 * (e.MAE / g.loc[bs, "MAE"] - 1):+.2f}%) at {g.loc[bs, "MFLOPs"] / max(e.MFLOPs, 1e-9):.1f}x fewer FLOPs, '
                         f'CPU {e.cpu_ms:.2f} vs {g.loc[bs, "cpu_ms"]:.2f} ms; Pareto: {"CAF-X-eff" in FAIR_PARETO[ds]}.')
if 'EDGE' in globals() and len(EDGE):
    for m in [x for x in EDGE.index if x == 'CAF-E1' or x.startswith('CAF-X-eff')]:
        e = EDGE.loc[m]
        lines.append(f'- Edge: {m} {e.MAE_above_best_pct:+.2f}% MAE vs best; {e.MFLOPs:.3f} MFLOPs, CPU p50/p99 '
                     f'{e.cpu1_p50_ms:.2f}/{e.cpu1_p99_ms:.2f} ms, {e.links_per_core:.0f} links/core, Pi-4 projection '
                     f'{e["proj_ms Raspberry Pi 4 (A72)"]:.1f} ms.')
if 'LOW' in globals() and len(LOW):
    for ds in LOW.dataset.unique():
        g = LOW[LOW.dataset == ds].set_index('model').mae_1day.sort_values()
        lines.append(f'- {ds}, 1 training day: ' + ', '.join(f'{m} {v:.2f}' for m, v in g.items()))
(TABLES / 'auto_summary.txt').write_text('\n'.join(lines), encoding='utf-8'); print('\n'.join(lines))

manifest = dict(protocol=PROTOCOL, config_sha=CONFIG_SHA, config=CONFIG, finished=time.strftime('%Y-%m-%d %H:%M:%S'),
                elapsed_min=round(elapsed_min(), 1), device=GPU_NAME, python=sys.version, platform=platform.platform(),
                cpu=platform.processor() or platform.machine(), cpu_threads=os.cpu_count(), idle_gpu_watts=IDLE_W,
                versions={'numpy': np.__version__, 'pandas': pd.__version__, 'torch': torch.__version__,
                          'numba': numba.__version__, 'scipy': __import__('scipy').__version__,
                          'sklearn': __import__('sklearn').__version__},
                deep_runs={s: sum(1 for k in DEEP_PREDS if k[0] == s) for s in ('main', 'ablation', 'res50', 'lookback')},
                cascade_runs=len(CASCADE), caf_flops=CAF_FLOPS, operating_escalation=OP_RATE,
                skipped_runs=len(SKIPPED), seeds=SEEDS, parameters=PARAMS, traces=TRACE_META)
(RUN / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
zip_path = ROOT / f'COMSNETS2027_SysAI_results_{CONFIG_SHA}.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in RUN.rglob('*'):
        if p.is_file() and p.parent.name not in ('ckpt', 'preds'):
            z.write(p, p.relative_to(ROOT))
checkpoint('final', force=True)
stamp(f'Results archive: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB); total {elapsed_min():.0f} min')
if not PERSISTENT:
    try:
    #         from google.colab import files
    #         files.download(str(zip_path))
    except Exception:
        pass