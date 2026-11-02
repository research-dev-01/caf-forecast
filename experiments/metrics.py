"""
Metrics and result store.
Corresponds to Cell 9 of cmsnts_latest.ipynb.
"""

import numpy as np

RESULTS = []     # one row per (study, dataset, model, seed, test day, horizon)

def per_day_metrics(D, pred, split='test'):
    S = D.splits[split]; out = []
    err = pred - S.y
    for k, day in enumerate(S.days):
        m = S.day == k
        mae = np.abs(err[m]).mean(0); rmse = np.sqrt((err[m] ** 2).mean(0))
        pmae = np.abs(S.last[m, None] - S.y[m]).mean(0)
        for h in range(D.H):
            out.append(dict(day=day, h=h + 1, mae=mae[h], rmse=rmse[h], mase=mae[h] / D.naive_mae,
                            skill=1 - mae[h] / pmae[h]))
    return out

def record(study, D, model, seed, pred_test, extra=None):
    global RESULTS
    RESULTS = [r for r in RESULTS if not (r['study'] == study and r['dataset'] == D.name and r['model'] == model
                                          and r['seed'] == seed)]
    for r in per_day_metrics(D, pred_test):
        r.update(study=study, dataset=D.name, model=model, seed=seed, **(extra or {})); RESULTS.append(r)

def mae_of(D, pred, split):
    return float(np.abs(pred - D.splits[split].y).mean())