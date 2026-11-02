"""
Windowed splits with GPU-resident batching.
Corresponds to Cell 8 of cmsnts_latest.ipynb.
"""

import numpy as np
import torch
from config import DEVICE, LOOKBACK, HORIZON, BIN_MS, DATASETS
from data.features import to_channels, aggregate_bins

class Split:
    def __init__(self, days, Z, R, L, H):
        self.days, self.L, self.H = days, L, H
        self.Z = torch.tensor(np.concatenate(Z), device=DEVICE)
        self.R = np.concatenate(R).astype(np.float32)
        self.Rt = torch.tensor(self.R, device=DEVICE)
        idx, day_of, off = [], [], 0
        for k, r in enumerate(R):
            t = np.arange(off + L, off + len(r) - H + 1)
            idx.append(t); day_of.append(np.full(len(t), k)); off += len(r)
        self.t = np.concatenate(idx); self.day = np.concatenate(day_of)
        self.t_dev = torch.tensor(self.t, device=DEVICE)
        self.n = len(self.t)
        self._lag = torch.arange(-L, 0, device=DEVICE); self._ahead = torch.arange(0, H, device=DEVICE)
        self.y = self.R[self.t[:, None] + np.arange(H)]                      # (n, H) Mbit/s
        self.last = self.R[self.t - 1]                                       # persistence

    def batch(self, ii, chans=None):
        t = self.t_dev[ii]
        x = self.Z[t[:, None] + self._lag]
        if chans is not None:
            x = x[..., chans]
        return x, self.Rt[t[:, None] + self._ahead]

    def x_numpy(self, chans=None, L=None):
        L = L or self.L
        Zc = self.Z.cpu().numpy()
        if chans is not None:
            Zc = Zc[:, chans]
        return Zc[self.t[:, None] + np.arange(-L, 0)]

class TaskData:
    def __init__(self, name, bin_ms, L, H, acc_by_day=None, stats=None):
        cfg = DATASETS[name]; self.name, self.bin_ms, self.L, self.H = name, bin_ms, L, H
        get = (lambda d: acc_by_day[d]) if acc_by_day else (lambda d: TRACE[d] if bin_ms == BIN_MS
                                                              else aggregate_bins(TRACE[d], bin_ms // BIN_MS))
        ch = {d: to_channels(get(d), bin_ms) for s in ('train', 'val', 'test') for d in cfg[s]}
        tr = np.concatenate([ch[d] for d in cfg['train']])
        self.mu, self.sd = stats if stats is not None else (tr.mean(0), tr.std(0) + 1e-6)
        self.own_stats = (tr.mean(0), tr.std(0) + 1e-6)
        self.splits = {}
        for s in ('train', 'val', 'test'):
            Z = [((ch[d] - self.mu) / self.sd).astype(np.float32) for d in cfg[s]]
            R = [ch[d][:, 0] for d in cfg[s]]
            self.splits[s] = Split(cfg[s], Z, R, L, H)
        rtr = [ch[d][:, 0] for d in cfg['train']]
        self.naive_mae = float(np.mean(np.concatenate([np.abs(np.diff(r)) for r in rtr])))
        self.p95, self.p99 = np.percentile(tr[:, 0], [95, 99])

    def to_mbps(self, z):
        return np.maximum(z * self.sd[0] + self.mu[0], 0)

    def norm_y(self, y):
        return (y - float(self.mu[0])) / float(self.sd[0])

def build_task(name, bin_ms=BIN_MS, L=LOOKBACK, H=HORIZON):
    return TaskData(name, bin_ms, L, H)

DATA = {ds: build_task(ds) for ds in DATASETS}
for ds, D in DATA.items():
    print(ds, {s: D.splits[s].n for s in D.splits}, f'naive MAE {D.naive_mae:.2f} Mbit/s, p95 {D.p95:.1f} Mbit/s')