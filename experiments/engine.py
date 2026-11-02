"""
Training engine and main experiment runner.
Corresponds to Cell 15 and Cell 16 of cmsnts_latest.ipynb.
"""

import time
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from contextlib import nullcontext

from config import (
    DEVICE, AMP, MAX_EPOCHS, PATIENCE, BATCH_SIZE, LR, WEIGHT_DECAY,
    ANALYSIS_RESERVE_MIN, CKPT, PREDS, HIST, RUN, SEEDS, DEEP_MODELS
)
from utils.helpers import stamp, checkpoint, remaining_min
from experiments.metrics import record, per_day_metrics

DEEP_PREDS = {}      # (study, dataset, model, seed) -> {'val': (n,H), 'test': (n,H)} in Mbit/s
HISTORY = {}         # (study, dataset, model, seed) -> dict(history=[...], best_epoch, train_seconds, ...)
RUN_LOG = RUN / 'run_log.csv'

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def autocast():
    return torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=AMP)

def predict(model, D, split, chans, bs=4096, transform=None):
    S = D.splits[split]; out = []; model.eval()
    with torch.inference_mode(), autocast():
        for i in range(0, S.n, bs):
            x, _ = S.batch(torch.arange(i, min(i + bs, S.n), device=DEVICE), chans)
            if transform is not None:
                x = transform(x)
            out.append(model(x).float())
    return D.to_mbps(torch.cat(out).cpu().numpy()).astype(np.float32)

def run_key(study, D, name, seed):
    return f'{study}__{slug(D.name)}_{D.bin_ms}ms_L{D.L}__{slug(name)}__s{seed}'

def load_model(study, D, name, spec, seed):
    ctor, chans = spec
    m = ctor(n_in(chans), D.L, D.H).to(DEVICE)
    m.load_state_dict(torch.load(CKPT / f'{run_key(study, D, name, seed)}.pt', map_location=DEVICE,
                                 weights_only=False)['state'])
    return m.eval()

def run_deep(study, D, name, spec, seed, max_epochs=MAX_EPOCHS, verbose=False):
    key = run_key(study, D, name, seed); f_ck, f_pr, f_hi = CKPT / f'{key}.pt', PREDS / f'{key}.npz', HIST / f'{key}.json'
    k = (study, D.name if study != 'lookback' else f'{D.name}|L{D.L}', name, seed)
    if f_ck.exists() and f_pr.exists() and f_hi.exists():
        z = np.load(f_pr); DEEP_PREDS[k] = {'val': z['val'], 'test': z['test']}
        HISTORY[k] = json.loads(f_hi.read_text()); return HISTORY[k]
    ctor, chans = spec; set_seed(seed)
    model = ctor(n_in(chans), D.L, D.H).to(DEVICE)
    S = D.splits['train']; nb = max(1, S.n // BATCH_SIZE); total = max_epochs * nb
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: .05 + .95 * .5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    gen = torch.Generator().manual_seed(seed)
    mu0, sd0 = float(D.mu[0]), float(D.sd[0])
    pv = predict(model, D, 'val', chans)
    hist = [dict(epoch=0, train_loss=None, val_mae=mae_of(D, pv, 'val'),
                 val_rmse=float(np.sqrt(((pv - D.splits['val'].y) ** 2).mean())), lr=LR, seconds=0.0)]
    best, best_ep, best_state, bad, status = hist[0]['val_mae'], 0, {k_: v.detach().clone() for k_, v in model.state_dict().items()}, 0, 'ok'
    t_start = time.time()
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); mem0 = torch.cuda.memory_allocated()
    for ep in range(1, max_epochs + 1):
        t0 = time.time(); model.train(); perm = torch.randperm(S.n, generator=gen).to(DEVICE)
        tot = torch.zeros((), device=DEVICE); steps = 0
        for i in range(0, S.n, BATCH_SIZE):
            ii = perm[i:i + BATCH_SIZE]
            if len(ii) < 32:
                continue
            x, y = S.batch(ii, chans); yz = (y - mu0) / sd0
            with autocast():
                if hasattr(model, 'forward_exits'):          # CAF: deep supervision over the nested exits
                    outs = model.forward_exits(x)
                else:
                    outs = [model(x)]
            ws = getattr(model, 'EXIT_W', (1.0,))
            loss = sum(w * F.huber_loss(o.float(), yz, delta=1.0) for w, o in zip(ws, outs) if w > 0)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.detach(); steps += 1
        tl = (tot / max(steps, 1)).item()
        if not math.isfinite(tl):
            status = f'non-finite loss at epoch {ep}'; break
        pv = predict(model, D, 'val', chans); vm = mae_of(D, pv, 'val')
        hist.append(dict(epoch=ep, train_loss=tl, val_mae=vm, val_rmse=float(np.sqrt(((pv - D.splits['val'].y) ** 2).mean())),
                         lr=opt.param_groups[0]['lr'], seconds=time.time() - t0))
        if verbose:
            print(f'    ep {ep:2d} loss {tl:.4f} val MAE {vm:.3f}')
        if vm < best - 1e-6:
            best, best_ep, bad = vm, ep, 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    train_s = time.time() - t_start
    peak_mb = (torch.cuda.max_memory_allocated() - mem0) / 2**20 if DEVICE.type == 'cuda' else float('nan')
    model.load_state_dict(best_state)
    pv, pt = predict(model, D, 'val', chans), predict(model, D, 'test', chans)
    torch.save({'state': best_state, 'model': name, 'dataset': D.name, 'seed': seed, 'config_sha': CONFIG_SHA}, f_ck)
    np.savez_compressed(f_pr, val=pv, test=pt)
    info = dict(history=hist, best_epoch=best_ep, epochs_run=hist[-1]['epoch'], best_val_mae=best,
                train_seconds=train_s, sec_per_epoch=float(np.mean([h['seconds'] for h in hist[1:]])) if len(hist) > 1 else 0.0,
                params=count_params(model), status=status, chans=n_in(chans), peak_train_mem_mb=peak_mb,
                train_windows_seen=int(S.n * (hist[-1]['epoch'])))
    f_hi.write_text(json.dumps(info))
    DEEP_PREDS[k] = {'val': pv, 'test': pt}; HISTORY[k] = info
    row = dict(study=study, dataset=D.name, bin_ms=D.bin_ms, L=D.L, model=name, seed=seed, best_epoch=best_ep,
               epochs=info['epochs_run'], val_mae=best, test_mae=mae_of(D, pt, 'test'), train_s=round(train_s, 1),
               status=status, finished=time.strftime('%Y-%m-%d %H:%M:%S'))
    pd.DataFrame([row]).to_csv(RUN_LOG, mode='a', header=not RUN_LOG.exists(), index=False)
    del model, opt; gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return info

RUN_MINUTES = []     # wall-clock minutes of newly trained runs (for the time guard)

def guarded_run(study, D, name, spec, seed, reserve_min, **kw):
    key = run_key(study, D, name, seed)
    cached = all((p / f'{key}{ext}').exists() for p, ext in ((CKPT, '.pt'), (PREDS, '.npz'), (HIST, '.json')))
    est = 1.5 * (np.median(RUN_MINUTES) if RUN_MINUTES else 1.0)
    if not cached and remaining_min() - est < reserve_min:
        return None
    t0 = time.time(); info = run_deep(study, D, name, spec, seed, **kw)
    if not cached:
        RUN_MINUTES.append((time.time() - t0) / 60)
        checkpoint(f'{study}/{name}/s{seed}')
    return info

def run_main_experiment(DATA):
    """Run main grid: 11 deep models x 3 datasets x 5 seeds (seed-major order)."""
    global SKIPPED
    SKIPPED = []
    stamp(f'Main grid: {len(DEEP_MODELS)} models x {len(DATA)} datasets x {len(SEEDS)} seeds')
    SKIPPED = []
    stamp(f'Main grid: {len(DEEP_MODELS)} models × {len(DATA)} datasets × {len(SEEDS)} seeds')
    for seed in SEEDS:
        for ds, D in DATA.items():
            for name in DEEP_MODELS:
                info = guarded_run('main', D, name, MODEL_SPECS[name], seed, reserve_min=ANALYSIS_RESERVE_MIN + 30)
                if info is None:
                    SKIPPED.append(('main', ds, name, seed)); continue
                record('main', D, name, seed, DEEP_PREDS[('main', ds, name, seed)]['test'])
        done = sum(1 for k in DEEP_PREDS if k[0] == 'main' and k[3] == seed)
        stamp(f'seed {seed} finished ({done} runs); remaining budget {remaining_min():.0f} min')
        checkpoint(f'main grid seed {seed}', force=True)
    if SKIPPED:
        print(f'WARNING: {len(SKIPPED)} main runs skipped by the time guard; tables report the seeds that finished.')
    main_log = pd.DataFrame([dict(dataset=k[1], model=k[2], seed=k[3], best_epoch=v['best_epoch'], epochs=v['epochs_run'],
                                  val_mae=v['best_val_mae'], train_s=v['train_seconds'], status=v['status'])
                             for k, v in HISTORY.items() if k[0] == 'main'])
    display(main_log.groupby(['dataset', 'model'])[['val_mae', 'best_epoch', 'train_s']].mean().round(3).unstack(0))
    return HISTORY, DEEP_PREDS
