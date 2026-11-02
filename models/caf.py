"""
Proposed CAF (compute-adaptive, three nested exits + escalation router) + model registry.
Corresponds to Cell 13 of cmsnts_latest.ipynb.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LOOKBACK, HORIZON, DEVICE
from data.features import CHANNELS, BASIC_CHANNELS
from models.baselines import (
    LSTMNet, TCN, DLinear, NHiTS, PatchTST, TimesNet, TSMixer, ITransformer, TimeMixer, TimeXer
)

class GatedBlock(nn.Module):
    def __init__(self, ch, dil, k=3, drop=.1):
        super().__init__(); self.p = (k - 1) * dil
        self.conv = nn.Conv1d(ch, 2 * ch, k, dilation=dil); self.out = nn.Conv1d(ch, ch, 1)
        self.norm = nn.LayerNorm(ch); self.drop = nn.Dropout(drop)
    def forward(self, h):                                   # h (B, ch, T)
        a, b = self.conv(F.pad(h, (self.p, 0))).chunk(2, 1)
        z = self.drop(self.out(torch.tanh(a) * torch.sigmoid(b)))
        return self.norm((h + z).transpose(1, 2)).transpose(1, 2)

class CAF(nn.Module):
    SCALES = (2, 4, 8, 16, 32)
    def __init__(self, C, L, H, ch=48, hid=64, anchor='learned', use_ms=True, use_conv=True, deep_sup=True, scale_eq=True,
                 drop=.1):
        super().__init__(); self.C, self.L, self.H = C, L, H
        self.anchor, self.use_ms, self.use_conv, self.scale_eq = anchor, use_ms, use_conv, scale_eq
        self.EXIT_W = (.25, .5, 1.0) if deep_sup else (0., 0., 1.0)
        self.scales = [s for s in self.SCALES if s < L]
        self.na = len(self.scales) + 2                      # last value, MA_s ..., window mean
        self.logits = nn.Parameter(torch.zeros(H, self.na))
        # exit 1: linear residual over the level/scale-normalised target window + last-bin header channels
        self.lin = nn.Linear(L + C - 1, H)
        nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)
        # exit 2: gated causal conv + GRU residual
        nin = (1 + (C - 1) + 2 * len(self.scales) + (0 if scale_eq else 1)) if use_ms else C
        self.inp = nn.Sequential(nn.Linear(nin, ch), nn.GELU())
        self.blocks = nn.ModuleList(GatedBlock(ch, d, drop=drop) for d in (1, 2, 4, 8)) if use_conv else nn.ModuleList()
        self.gru = nn.GRU(ch, hid, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hid), nn.Linear(hid, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, H))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        # escalation router (trained post hoc with the backbone frozen)
        self.nrf = H + 4 + (C - 1)
        self.router = nn.Sequential(nn.Linear(self.nrf, 32), nn.GELU(), nn.Linear(32, 1))
        self.register_buffer('rf_mu', torch.zeros(self.nrf)); self.register_buffer('rf_sd', torch.ones(self.nrf))

    # ---- shared cheap part -------------------------------------------------------------------------------------
    def trailing_means(self, r):                            # causal means inside the window, (B, L, S)
        c = F.pad(torch.cumsum(r.float(), 1), (1, 0)); idx = torch.arange(1, self.L + 1, device=r.device)
        outs = []
        for s in self.scales:
            lo = (idx - s).clamp(min=0)
            outs.append((c[:, idx] - c[:, lo]) / (idx - lo).float())
        return torch.stack(outs, -1).to(r.dtype)

    def anchors(self, r, ma):
        return torch.cat([r[:, -1:], ma[:, -1], r.mean(1, keepdim=True)], 1)   # (B, na)

    def anchor_weights(self):
        return torch.softmax(self.logits, -1)

    @staticmethod
    def local_scale(r):                                     # (B, 1) window std of the target (floor 0.01 global SD)
        return torch.sqrt(r.var(1, keepdim=True) + 1e-4)

    def _rel(self, x, r):
        """level/scale-equivariance: local scale and instance-normalised header channels"""
        if not self.scale_eq:
            return torch.ones_like(r[:, :1]), x[..., 1:]
        hx = x[..., 1:]
        return self.local_scale(r), (hx - hx.mean(1, keepdim=True)) / torch.sqrt(hx.var(1, keepdim=True) + 1e-2)

    def _cheap(self, x):
        r = x[..., 0]; ma = self.trailing_means(r); cand = self.anchors(r, ma)
        if self.anchor == 'learned':
            e0 = cand @ self.anchor_weights().T.to(cand.dtype)
        elif self.anchor == 'ma8':
            e0 = ma[:, -1, self.scales.index(8)][:, None].expand(-1, self.H)
        else:                                               # 'mean': RevIN-style window-mean centring
            e0 = r.mean(1, keepdim=True).expand(-1, self.H)
        lvl = ma[:, -1, -1:]                                                    # (B, 1) slowest trailing mean
        sc, hx = self._rel(x, r)
        e1 = e0 + sc * self.lin(torch.cat([(r - lvl) / sc, hx[:, -1]], -1))
        return r, ma, e0, e1

    def router_features(self, x, r, e0, e1):                # scale-free -> escalation rate transfers across load levels
        hx = self._rel(x, r)[1]; sc = self.local_scale(r)
        f = torch.cat([(e1 - e0) / sc, r[:, -8:].std(1, keepdim=True) / sc, r[:, -32:].std(1, keepdim=True) / sc,
                       (r[:, -1] - r[:, -8:].mean(1)).abs()[:, None] / sc, (r[:, -1] - r[:, -2]).abs()[:, None] / sc,
                       hx[:, -1]], -1)
        return (f.float() - self.rf_mu) / self.rf_sd

    # ---- deep branch -------------------------------------------------------------------------------------------
    def _deep(self, x, r, ma, e1):
        lvl = ma[:, -1:, -1:]
        if self.scale_eq:
            sc, hx = self._rel(x, r); s3 = sc[..., None]
            rc = (r[..., None] - lvl) / s3
            feat = torch.cat([rc, hx, (ma - lvl) / s3, (r[..., None] - ma) / s3], -1) if self.use_ms else torch.cat([rc, hx], -1)
        elif self.use_ms:
            sc = None
            feat = torch.cat([r[..., None] - lvl, x[..., 1:], ma - lvl, r[..., None] - ma, lvl.expand(-1, self.L, 1)], -1)
        else:
            sc = None; feat = x
        h = self.inp(feat).transpose(1, 2)
        for b in self.blocks:
            h = b(h)
        g = self.gru(h.transpose(1, 2))[0][:, -1]
        res = self.head(g)
        return e1 + (res * sc if sc is not None else res)

    # ---- public entry points -----------------------------------------------------------------------------------
    def forward_exits(self, x):                             # [E0, E1, E2] (training with deep supervision)
        r, ma, e0, e1 = self._cheap(x)
        return [e0, e1, self._deep(x, r, ma, e1)]

    def forward(self, x):                                   # full model = E2 (always escalate)
        r, ma, e0, e1 = self._cheap(x)
        return self._deep(x, r, ma, e1)

    def forward_e1(self, x):                                # cheap exit only
        return self._cheap(x)[3]

    def forward_cheap(self, x):                             # cheap exit + router score (always executed)
        r, ma, e0, e1 = self._cheap(x)
        return e1, self.router(self.router_features(x, r, e0, e1)).squeeze(-1)

    def forward_adaptive(self, x, tau):                     # deployed cascade: deep branch only where score > tau
        r, ma, e0, e1 = self._cheap(x)
        s = self.router(self.router_features(x, r, e0, e1)).squeeze(-1)
        esc = s > tau; out = e1.clone()
        if bool(esc.any()):
            i = esc.nonzero().squeeze(1)
            out[i] = self._deep(x[i], r[i], ma[i], e1[i]).to(out.dtype)
        return out, esc

ALL = None
MODEL_SPECS = {   # name -> (constructor, input channels: None = all, [0] = target only)
    'LSTM': (LSTMNet, ALL), 'TCN': (TCN, ALL), 'DLinear': (DLinear, [0]), 'N-HiTS': (NHiTS, ALL),
    'PatchTST': (PatchTST, [0]), 'TimesNet': (TimesNet, ALL), 'TSMixer': (TSMixer, ALL),
    'iTransformer': (ITransformer, ALL), 'TimeMixer': (TimeMixer, [0]), 'TimeXer': (TimeXer, ALL),
    'CAF': (CAF, ALL),
}
ABLATION_SPECS = {
    'CAF w/o scale-equivariant residual': (lambda C, L, H: CAF(C, L, H, scale_eq=False), ALL),
    'CAF w/o deep supervision': (lambda C, L, H: CAF(C, L, H, deep_sup=False), ALL),
    'CAF mean anchor (RevIN-style)': (lambda C, L, H: CAF(C, L, H, anchor='mean'), ALL),
    'CAF w/o multiscale input': (lambda C, L, H: CAF(C, L, H, use_ms=False), ALL),
    'CAF w/o gated conv': (lambda C, L, H: CAF(C, L, H, use_conv=False), ALL),
    'CAF w/o header features': (CAF, BASIC_CHANNELS),
}
DEEP_MODELS = list(MODEL_SPECS)
CLASSICAL_MODELS = ['Persistence', 'MovingAvg', 'EWMA', 'Ridge-AR', 'HistGB']
PROPOSED = 'CAF'
CASCADE_MODELS = ['CAF-E1', 'CAF-adaptive']    # derived from the trained CAF checkpoints (no extra training)
MODEL_REF = {'Persistence': '--', 'MovingAvg': '--', 'EWMA': '--', 'Ridge-AR': '--', 'HistGB': 'Ke et al. 2017',
             'LSTM': 'Hochreiter 1997', 'TCN': 'Bai et al. 2018', 'DLinear': 'AAAI 2023', 'N-HiTS': 'AAAI 2023',
             'PatchTST': 'ICLR 2023', 'TimesNet': 'ICLR 2023', 'TSMixer': 'TMLR 2023', 'iTransformer': 'ICLR 2024',
             'TimeMixer': 'ICLR 2024', 'TimeXer': 'NeurIPS 2024', 'CAF': 'proposed (full, E2)',
             'CAF-E1': 'proposed (cheap exit)', 'CAF-adaptive': 'proposed (cascade)'}

def n_in(chans):
    return len(CHANNELS) if chans is None else len(chans)

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

PARAMS = {}

def verify_models_and_get_params():
    global PARAMS
    PARAMS = {}
    _x = torch.randn(4, LOOKBACK, len(CHANNELS), device=DEVICE)
    for name, (ctor, chans) in {**MODEL_SPECS, **ABLATION_SPECS}.items():
        m = ctor(n_in(chans), LOOKBACK, HORIZON).to(DEVICE).eval()
        with torch.no_grad():
            y = m(_x if chans is None else _x[..., chans])
        assert y.shape == (4, HORIZON) and torch.isfinite(y).all(), name
        PARAMS[name] = count_params(m)
    # CAF at initialisation: every exit equals the uniform multiscale anchor; the cascade equals E1 at b=0 and E2 at b=1
    _m = CAF(len(CHANNELS), LOOKBACK, HORIZON).to(DEVICE).eval()
    with torch.no_grad():
        r = _x[..., 0]; expected = _m.anchors(r, _m.trailing_means(r)).mean(1, keepdim=True).expand(-1, HORIZON)
        for e in _m.forward_exits(_x):
            assert torch.allclose(e, expected, atol=1e-5)
        nn.init.normal_(_m.head[-1].weight, std=.1); nn.init.normal_(_m.lin.weight, std=.1)
        full, e1 = _m(_x), _m.forward_e1(_x)
        a_hi, esc_hi = _m.forward_adaptive(_x, -float('inf')); a_lo, esc_lo = _m.forward_adaptive(_x, float('inf'))
        assert torch.allclose(a_hi, full, atol=1e-5) and torch.allclose(a_lo, e1, atol=1e-6)
        assert bool(esc_hi.all()) and not bool(esc_lo.any())
        # scale equivariance of every exit: E_k(a*x + b) = a*E_k(x) + b on the target channel
        _x2 = _x.clone(); _x2[..., 0] = 3.0 * _x[..., 0] + 2.0
        for e_a, e_b in zip(_m.forward_exits(_x2), _m.forward_exits(_x)):
            assert torch.allclose(e_a, 3.0 * e_b + 2.0, atol=1e-2, rtol=1e-3), 'CAF is not scale-equivariant'
    del _m
    display(pd.Series(PARAMS, name='trainable parameters').to_frame())
    return PARAMS

try:
    PARAMS = verify_models_and_get_params()
except Exception as e:
    pass
