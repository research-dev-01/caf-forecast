"""
Compute-adaptive cascade: router fit (train days) -> budget calibration (validation day) -> one-shot test.
Corresponds to Cell 17 of cmsnts_latest.ipynb.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import (
    DEVICE, PROPOSED, SEEDS, CASCADE_TOL, BUDGET_GRID, ROUTER_EPOCHS, TABLES, LR
)
from utils.helpers import stamp, checkpoint
from experiments.metrics import record, per_day_metrics, mae_of
from experiments.engine import load_model

CASCADE = {}         # (study, dataset, model, seed) -> operating point
CASCADE_ROWS = []    # one row per (study, dataset, model, seed, router, budget)

def caf_outputs(model, D, split, chans, bs=8192):
    """E0/E1/E2 (z-space) and router features for every window of a split (torch, float32, on DEVICE)."""
    S = D.splits[split]; out = {k: [] for k in ('e0', 'e1', 'e2', 'f', 'sc')}; model.eval()
    with torch.no_grad(), autocast():
        for i in range(0, S.n, bs):
            x, _ = S.batch(torch.arange(i, min(i + bs, S.n), device=DEVICE), chans)
            r, ma, e0, e1 = model._cheap(x); e2 = model._deep(x, r, ma, e1)
            for k, v in zip(('e0', 'e1', 'e2', 'f', 'sc'),
                            (e0, e1, e2, model.router_features(x, r, e0, e1), model.local_scale(r.float())[:, 0])):
                out[k].append(v.float())
    return {k: torch.cat(v) for k, v in out.items()}

def fit_router(model, D, chans, seed):
    """Backbone frozen. Router regresses the per-window gain of E2 over E1, relative to the window's local scale
    (scale-free, like its features, so a validation-calibrated budget transfers to days with a different load level),
    on the TRAINING days; epoch selected on the validation day. Test days are never touched here."""
    with torch.no_grad():
        model.rf_mu.zero_(); model.rf_sd.fill_(1.0)
    tr = caf_outputs(model, D, 'train', chans)
    mu, sd = tr['f'].mean(0), tr['f'].std(0) + 1e-6
    with torch.no_grad():
        model.rf_mu.copy_(mu); model.rf_sd.copy_(sd)
    X = (tr['f'] - mu) / sd

    def gain(o, split):
        y = torch.tensor(D.norm_y(D.splits[split].y), device=DEVICE, dtype=torch.float32)
        return ((o['e1'] - y).abs() - (o['e2'] - y).abs()).mean(1) / o['sc']

    g = gain(tr, 'train'); va = caf_outputs(model, D, 'val', chans); gv = gain(va, 'val')
    set_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    for lyr in model.router:
        if isinstance(lyr, nn.Linear):
            lyr.reset_parameters()
    for p in model.router.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.router.parameters(), lr=2e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed); best, best_state, bad = float('inf'), None, 0
    for ep in range(ROUTER_EPOCHS):
        model.router.train(); perm = torch.randperm(len(X), generator=gen).to(DEVICE)
        for i in range(0, len(X), 2048):
            ii = perm[i:i + 2048]
            loss = F.huber_loss(model.router(X[ii]).squeeze(-1), g[ii], delta=1.0)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.router.eval()
        with torch.no_grad():
            vl = F.huber_loss(model.router(va['f']).squeeze(-1), gv, delta=1.0).item()
        if vl < best - 1e-7:
            best, bad = vl, 0; best_state = {k: v.detach().clone() for k, v in model.router.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                break
    model.router.load_state_dict(best_state)
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()
    return va

def budget_threshold(score_val, b):
    if b <= 0:
        return float('inf')
    if b >= 1:
        return -float('inf')
    return float(np.quantile(score_val, 1 - b))

def load_caf(study, D, name, spec, seed):
    """trained CAF backbone + its fitted router and calibrated operating point"""
    m = load_model(study, D, name, spec, seed)
    ck = torch.load(CKPT / f'{run_key(study, D, name, seed)}__router.pt', map_location=DEVICE, weights_only=False)
    m.router.load_state_dict(ck['router']); m.rf_mu.copy_(ck['rf_mu']); m.rf_sd.copy_(ck['rf_sd'])
    return m.eval(), ck

def cascade_eval(study, D, name, spec, seed, register=True):
    key = run_key(study, D, name, seed); f_r = CKPT / f'{key}__router.pt'
    f_p = PREDS / f'{key}__cascade.npz'; f_j = HIST / f'{key}__cascade.json'
    if f_r.exists() and f_p.exists() and f_j.exists():
        z = np.load(f_p); info = json.loads(f_j.read_text())
    else:
        model = load_model(study, D, name, spec, seed); chans = spec[1]
        va = fit_router(model, D, chans, seed); te = caf_outputs(model, D, 'test', chans)
        with torch.no_grad():
            sv = model.router(va['f']).squeeze(-1).float().cpu().numpy()
            st = model.router(te['f']).squeeze(-1).float().cpu().numpy()
        mb = lambda t: D.to_mbps(t.cpu().numpy()).astype(np.float32)
        e1v, e2v, e1t, e2t = mb(va['e1']), mb(va['e2']), mb(te['e1']), mb(te['e2'])
        yv, yt = D.splits['val'].y, D.splits['test'].y
        H = D.H; rng = np.random.default_rng(seed)
        routers = {
            'learned': (sv, st),
            'volatility': (va['f'][:, H].cpu().numpy(), te['f'][:, H].cpu().numpy()),   # 80 ms std / window std
            'random': (rng.random(len(sv)), rng.random(len(st))),
            'oracle (test labels)': ((np.abs(e1v - yv) - np.abs(e2v - yv)).mean(1),
                                     (np.abs(e1t - yt) - np.abs(e2t - yt)).mean(1)),
        }
        rows = []
        for rn, (a, b) in routers.items():
            for q in BUDGET_GRID:
                tau = budget_threshold(a, q); ev, et = a > tau, b > tau
                pv = np.where(ev[:, None], e2v, e1v); pt = np.where(et[:, None], e2t, e1t)
                rows.append(dict(router=rn, budget=q, tau=tau, val_rate=float(ev.mean()), test_rate=float(et.mean()),
                                 val_mae=float(np.abs(pv - yv).mean()), test_mae=float(np.abs(pt - yt).mean())))
        lr = [r for r in rows if r['router'] == 'learned']; full_v = [r for r in lr if r['budget'] == 1.0][0]['val_mae']
        op = min((r for r in lr if r['val_mae'] <= (1 + CASCADE_TOL) * full_v), key=lambda r: r['budget'])
        tau = op['tau']; av = np.where((sv > tau)[:, None], e2v, e1v); at = np.where((st > tau)[:, None], e2t, e1t)
        torch.save({'router': model.router.state_dict(), 'rf_mu': model.rf_mu.clone(), 'rf_sd': model.rf_sd.clone(),
                    'tau': tau, 'budget': op['budget']}, f_r)
        np.savez_compressed(f_p, e1_val=e1v, e1_test=e1t, ad_val=av.astype(np.float32), ad_test=at.astype(np.float32))
        info = dict(rows=rows, budget=op['budget'], tau=tau, test_rate=op['test_rate'], val_rate=op['val_rate'])
        f_j.write_text(json.dumps(info))
        z = np.load(f_p)
        del model; gc.collect()
    k = (study, D.name, name, seed); CASCADE[k] = {kk: info[kk] for kk in ('budget', 'tau', 'test_rate', 'val_rate')}
    global CASCADE_ROWS
    CASCADE_ROWS = [r for r in CASCADE_ROWS if (r['study'], r['dataset'], r['model'], r['seed']) != k]
    CASCADE_ROWS += [dict(r, study=study, dataset=D.name, model=name, seed=seed) for r in info['rows']]
    if register:
        for nm, a, b in (('CAF-E1', 'e1_val', 'e1_test'), ('CAF-adaptive', 'ad_val', 'ad_test')):
            DEEP_PREDS[(study, D.name, nm, seed)] = {'val': z[a], 'test': z[b]}
            record(study, D, nm, seed, z[b])
    return CASCADE[k]

stamp('Cascade calibration ...')
for seed in SEEDS:
    for ds, D in DATA.items():
        if ('main', ds, PROPOSED, seed) in DEEP_PREDS:
            cascade_eval('main', D, PROPOSED, MODEL_SPECS[PROPOSED], seed)
CASC = pd.DataFrame(CASCADE_ROWS)
op_t = pd.DataFrame([dict(dataset=k[1], seed=k[3], **v) for k, v in CASCADE.items() if k[0] == 'main'])
display(op_t.groupby('dataset')[['budget', 'val_rate', 'test_rate']].agg(['mean', 'std']).round(3))
stamp('Cascade done.')
checkpoint('cascade', force=True)