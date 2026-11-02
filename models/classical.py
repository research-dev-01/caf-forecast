"""
Classical baselines (hyper-parameters chosen on the validation day only).
Corresponds to Cell 10 of cmsnts_latest.ipynb.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from experiments.metrics import record, mae_of

CLASSICAL_PREDS = {}    # (study, dataset, model) -> {'val': array, 'test': array}

def ma_forecast(S, k):
    out = np.empty(S.n, np.float32); c = np.r_[0, np.cumsum(S.R, dtype=np.float64)]
    t = S.t; kk = min(k, S.L)
    out[:] = (c[t] - c[t - kk]) / kk
    return out

def ewma_forecast(S, a):
    out = np.empty(S.n, np.float32); off = 0
    for k in range(len(S.days)):
        m = np.where(S.day == k)[0]; t = S.t[m]
        start = t[0] - S.L; stop = t[-1]
        r = S.R[start:stop].astype(np.float64)
        e, _ = signal.lfilter([a], [1, a - 1], r, zi=[(1 - a) * r[0]])
        out[m] = e[t - 1 - start]
    return out

def tab_features(S, L):
    x = S.x_numpy(L=L); r = x[..., 0]
    feats = [r[:, -16:], x[:, -1, 1:]]
    for k in (4, 8, 16, 32, 64):
        kk = min(k, L); feats.append(r[:, -kk:].mean(1, keepdims=True)); feats.append(r[:, -kk:].std(1, keepdims=True))
    feats.append(x[:, -8:, 1:].mean(1))
    return np.concatenate(feats, 1).astype(np.float32)

def run_classical(study, D, verbose=True):
    S = D.splits; H = D.H; preds = {}
    rep = lambda v: np.repeat(v[:, None], H, 1)
    preds['Persistence'] = {s: rep(S[s].last) for s in ('val', 'test')}
    best = {}
    for name, grid, fn in [('MovingAvg', [2, 4, 8, 16, 32, 64], ma_forecast),
                           ('EWMA', [.02, .05, .1, .2, .3, .5, .8], ewma_forecast)]:
        grid = [g for g in grid if name != 'MovingAvg' or g <= D.L]
        cand = {g: {s: fn(S[s], g) for s in ('val', 'test')} for g in grid}
        vm = {g: np.abs(cand[g]['val'][:, None] - S['val'].y).mean(0) for g in grid}   # per-horizon val MAE
        ch = [min(grid, key=lambda g: vm[g][h]) for h in range(H)]
        preds[name] = {s: np.stack([cand[ch[h]][s] for h in range(H)], 1) for s in ('val', 'test')}
        best[name] = ch
    Xtr = S['train'].x_numpy(chans=[0]).reshape(S['train'].n, -1)
    Xv = S['val'].x_numpy(chans=[0]).reshape(S['val'].n, -1); Xte = S['test'].x_numpy(chans=[0]).reshape(S['test'].n, -1)
    ytr = D.norm_y(S['train'].y)
    rb = min([0.1, 1, 10, 100, 1000], key=lambda a: np.abs(
        D.to_mbps(Ridge(alpha=a).fit(Xtr, ytr).predict(Xv)) - S['val'].y).mean())
    rm = Ridge(alpha=rb).fit(Xtr, ytr)
    preds['Ridge-AR'] = {'val': D.to_mbps(rm.predict(Xv)), 'test': D.to_mbps(rm.predict(Xte))}; best['Ridge-AR'] = rb
    Ftr, Fv, Fte = (tab_features(S[s], D.L) for s in ('train', 'val', 'test'))
    rng = np.random.default_rng(0); sub = rng.choice(len(Ftr), min(60000, len(Ftr)), replace=False)
    gv, gt = [], []
    for h in range(H):
        g = HistGradientBoostingRegressor(max_iter=200, learning_rate=.05, max_leaf_nodes=31, random_state=0,
                                          early_stopping=False).fit(Ftr[sub], ytr[sub, h])
        gv.append(g.predict(Fv)); gt.append(g.predict(Fte))
    preds['HistGB'] = {'val': D.to_mbps(np.stack(gv, 1)), 'test': D.to_mbps(np.stack(gt, 1))}
    for m, p in preds.items():
        CLASSICAL_PREDS[(study, D.name, m)] = {k: v.astype(np.float32) for k, v in p.items()}
        record(study, D, m, 'det', p['test'])
    if verbose:
        print(D.name, {m: round(mae_of(D, p['val'], 'val'), 3) for m, p in preds.items()}, 'val MAE | choices', best)
    return preds

stamp('Classical baselines ...')
for ds, D in DATA.items():
    run_classical('main', D)
stamp('Classical baselines done.')