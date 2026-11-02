"""
Baseline model zoo (compact re-implementations of the published architectures).
Corresponds to Cell 12 of cmsnts_latest.ipynb.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def revin(x, eps=1e-5):
    mu = x.mean(1, keepdim=True); sd = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + eps)
    return (x - mu) / sd, mu, sd

class LSTMNet(nn.Module):
    def __init__(self, C, L, H, hid=64):
        super().__init__(); self.rnn = nn.LSTM(C, hid, 2, batch_first=True, dropout=.1); self.head = nn.Linear(hid, H)
    def forward(self, x):
        return self.head(self.rnn(x)[0][:, -1])

class TCN(nn.Module):
    def __init__(self, C, L, H, ch=64, k=3, dil=(1, 2, 4, 8)):
        super().__init__(); wn = nn.utils.parametrizations.weight_norm
        self.inp = nn.Conv1d(C, ch, 1); self.k = k; self.dil = dil
        self.c1 = nn.ModuleList(wn(nn.Conv1d(ch, ch, k, dilation=d)) for d in dil)
        self.c2 = nn.ModuleList(wn(nn.Conv1d(ch, ch, k, dilation=d)) for d in dil)
        self.drop = nn.Dropout(.1); self.head = nn.Linear(ch, H)
    def forward(self, x):
        h = self.inp(x.transpose(1, 2))
        for c1, c2, d in zip(self.c1, self.c2, self.dil):
            p = (self.k - 1) * d
            z = self.drop(F.relu(c1(F.pad(h, (p, 0))))); z = self.drop(F.relu(c2(F.pad(z, (p, 0)))))
            h = F.relu(h + z)
        return self.head(h[:, :, -1])

def moving_avg(x, k):       # x (B, T, C), replicate padding, centred (as in Autoformer/DLinear)
    front = x[:, :1].repeat(1, (k - 1) // 2, 1); end = x[:, -1:].repeat(1, k // 2, 1)
    return F.avg_pool1d(torch.cat([front, x, end], 1).transpose(1, 2), k, 1).transpose(1, 2)

class DLinear(nn.Module):
    def __init__(self, C, L, H, k=25):
        super().__init__(); self.k = k
        self.season = nn.Linear(L, H); self.trend = nn.Linear(L, H)
        for m in (self.season, self.trend):
            nn.init.constant_(m.weight, 1 / L); nn.init.zeros_(m.bias)
    def forward(self, x):
        tr = moving_avg(x, self.k); se = x - tr
        return (self.season(se[..., 0]) + self.trend(tr[..., 0]))

class NHiTS(nn.Module):
    def __init__(self, C, L, H, hid=128, pools=(2, 2, 1), freqs=(4, 2, 1)):
        super().__init__(); self.L, self.H = L, H; self.pools = pools; self.knots = [max(1, H // f) for f in freqs]
        self.blocks = nn.ModuleList()
        for p, kn in zip(pools, self.knots):
            nin = math.ceil(L / p) * C
            self.blocks.append(nn.Sequential(nn.Linear(nin, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(),
                                             nn.Linear(hid, L + kn)))
    def forward(self, x):
        xn, mu, sd = revin(x); res = xn[..., 0]; ex = xn[..., 1:]; fc = 0
        for p, kn, blk in zip(self.pools, self.knots, self.blocks):
            inp = torch.cat([res[..., None], ex], -1).transpose(1, 2)
            inp = F.max_pool1d(inp, p, p, ceil_mode=True).flatten(1)
            th = blk(inp); res = res - th[:, :self.L]
            fc = fc + F.interpolate(th[:, None, self.L:], size=self.H, mode='linear', align_corners=True)[:, 0]
        return fc * sd[:, 0, :1] + mu[:, 0, :1]

class BNTransformerLayer(nn.Module):
    def __init__(self, d, heads, dff, drop=.1):
        super().__init__(); self.att = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Dropout(drop), nn.Linear(dff, d))
        self.bn1 = nn.BatchNorm1d(d); self.bn2 = nn.BatchNorm1d(d); self.drop = nn.Dropout(drop)
    def bn(self, m, z):
        return m(z.transpose(1, 2)).transpose(1, 2)
    def forward(self, z):
        z = self.bn(self.bn1, z + self.drop(self.att(z, z, z, need_weights=False)[0]))
        return self.bn(self.bn2, z + self.drop(self.ff(z)))

class PatchTST(nn.Module):
    def __init__(self, C, L, H, P=8, S=4, d=64, heads=4, layers=2):
        super().__init__(); self.P, self.S = P, S; self.N = (L - P) // S + 2
        self.emb = nn.Linear(P, d); self.pos = nn.Parameter(torch.randn(1, self.N, d) * .02)
        self.enc = nn.ModuleList(BNTransformerLayer(d, heads, 2 * d) for _ in range(layers))
        self.head = nn.Sequential(nn.Flatten(1), nn.Dropout(.1), nn.Linear(self.N * d, H))
    def forward(self, x):
        xn, mu, sd = revin(x[..., :1]); s = xn[..., 0]
        s = torch.cat([s, s[:, -1:].repeat(1, self.S)], 1)
        z = self.emb(s.unfold(1, self.P, self.S)) + self.pos
        for lyr in self.enc:
            z = lyr(z)
        return self.head(z) * sd[:, 0] + mu[:, 0]

class Inception(nn.Module):
    def __init__(self, cin, cout, n=4):
        super().__init__(); self.k = nn.ModuleList(nn.Conv2d(cin, cout, 2 * i + 1, padding=i) for i in range(n))
    def forward(self, z):
        return torch.stack([k(z) for k in self.k], -1).mean(-1)

class TimesBlock(nn.Module):
    def __init__(self, T, d, dff, k):
        super().__init__(); self.T, self.k = T, k
        self.conv = nn.Sequential(Inception(d, dff), nn.GELU(), Inception(dff, d))
    def forward(self, z):
        B, T, D = z.shape
        with torch.autocast(device_type=z.device.type, enabled=False):
            amp = torch.fft.rfft(z.float(), dim=1).abs()
        f = amp.mean(0).mean(-1); f[0] = 0
        top = torch.topk(f, self.k).indices.clamp(min=1).tolist(); w = amp.mean(-1)[:, top]
        outs = []
        for fi in top:
            p = max(1, T // fi); n = math.ceil(T / p) * p
            zz = F.pad(z, (0, 0, 0, n - T)).reshape(B, n // p, p, D).permute(0, 3, 1, 2)
            outs.append(self.conv(zz).permute(0, 2, 3, 1).reshape(B, n, D)[:, :T])
        w = torch.softmax(w, 1).to(z.dtype)
        return z + (torch.stack(outs, -1) * w[:, None, None]).sum(-1)

class TimesNet(nn.Module):
    def __init__(self, C, L, H, d=32, dff=32, k=3, layers=2):
        super().__init__(); self.H = H; T = L + H
        self.tok = nn.Conv1d(C, d, 3, padding=1, padding_mode='circular', bias=False)
        pe = torch.zeros(T, d); pos = torch.arange(T)[:, None]; div = torch.exp(torch.arange(0, d, 2) * (-math.log(1e4) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div); self.register_buffer('pe', pe[None])
        self.predict = nn.Linear(L, T); self.blocks = nn.ModuleList(TimesBlock(T, d, dff, k) for _ in range(layers))
        self.norm = nn.LayerNorm(d); self.proj = nn.Linear(d, 1); self.drop = nn.Dropout(.1)
    def forward(self, x):
        xn, mu, sd = revin(x)
        z = self.tok(xn.transpose(1, 2)).transpose(1, 2)
        z = self.predict(z.transpose(1, 2)).transpose(1, 2) + self.pe
        z = self.drop(z)
        for b in self.blocks:
            z = self.norm(b(z))
        return self.proj(z)[:, -self.H:, 0] * sd[:, 0, :1] + mu[:, 0, :1]

class TSMixer(nn.Module):
    def __init__(self, C, L, H, hid=64, blocks=2, drop=.1):
        super().__init__(); self.blocks = nn.ModuleList()
        for _ in range(blocks):
            self.blocks.append(nn.ModuleDict(dict(
                n1=nn.LayerNorm(C), t=nn.Sequential(nn.Linear(L, L), nn.ReLU(), nn.Dropout(drop)),
                n2=nn.LayerNorm(C), f=nn.Sequential(nn.Linear(C, hid), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid, C),
                                                     nn.Dropout(drop)))))
        self.out = nn.Linear(L, H)
    def forward(self, x):
        xn, mu, sd = revin(x); z = xn
        for b in self.blocks:
            z = z + b['t'](b['n1'](z).transpose(1, 2)).transpose(1, 2)
            z = z + b['f'](b['n2'](z))
        return self.out(z[..., 0]) * sd[:, 0, :1] + mu[:, 0, :1]

class ITransformer(nn.Module):
    def __init__(self, C, L, H, d=64, heads=4, layers=2):
        super().__init__(); self.emb = nn.Linear(L, d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 2 * d, .1, 'gelu', batch_first=True),
                                         layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.proj = nn.Linear(d, H)
    def forward(self, x):
        xn, mu, sd = revin(x)
        z = self.enc(self.emb(xn.transpose(1, 2)))
        return self.proj(z[:, 0]) * sd[:, 0, :1] + mu[:, 0, :1]

class TimeMixer(nn.Module):
    def __init__(self, C, L, H, d=16, dff=32, scales=2, k=25, layers=2):
        super().__init__(); self.Ls = [L // 2 ** i for i in range(scales + 1)]; self.k = k; self.layers = layers
        self.emb = nn.Linear(1, d)
        mlp = lambda a, b: nn.Sequential(nn.Linear(a, b), nn.GELU(), nn.Linear(b, b))
        self.up = nn.ModuleList(nn.ModuleList(mlp(self.Ls[i], self.Ls[i + 1]) for i in range(scales)) for _ in range(layers))
        self.down = nn.ModuleList(nn.ModuleList(mlp(self.Ls[i + 1], self.Ls[i]) for i in range(scales)) for _ in range(layers))
        self.ffn = nn.ModuleList(nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Linear(dff, d)) for _ in range(layers))
        self.norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.pred = nn.ModuleList(nn.Linear(l, H) for l in self.Ls); self.proj = nn.Linear(d, 1)
    def forward(self, x):
        xn, mu, sd = revin(x[..., :1])
        xs = [xn]
        for _ in range(len(self.Ls) - 1):
            xs.append(F.avg_pool1d(xs[-1].transpose(1, 2), 2).transpose(1, 2))
        zs = [self.emb(s) for s in xs]
        for l in range(self.layers):
            tr = [moving_avg(z, min(self.k, z.shape[1] | 1)) for z in zs]; se = [z - t for z, t in zip(zs, tr)]
            for i in range(len(se) - 1):       # bottom-up season mixing (fine -> coarse)
                se[i + 1] = se[i + 1] + self.up[l][i](se[i].transpose(1, 2)).transpose(1, 2)
            for i in reversed(range(len(tr) - 1)):  # top-down trend mixing (coarse -> fine)
                tr[i] = tr[i] + self.down[l][i](tr[i + 1].transpose(1, 2)).transpose(1, 2)
            zs = [self.norm[l](z + self.ffn[l](s + t)) for z, s, t in zip(zs, se, tr)]
        out = sum(self.proj(p(z.transpose(1, 2)).transpose(1, 2)) for p, z in zip(self.pred, zs))
        return out[..., 0] * sd[:, 0] + mu[:, 0]

class TimeXerLayer(nn.Module):
    def __init__(self, d, heads, dff, drop=.1):
        super().__init__()
        self.sa = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ca = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Conv1d(d, dff, 1), nn.GELU(), nn.Dropout(drop), nn.Conv1d(dff, d, 1))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d); self.drop = nn.Dropout(drop)
    def forward(self, z, ex):
        z = self.n1(z + self.drop(self.sa(z, z, z, need_weights=False)[0]))
        g = z[:, -1:]
        g = self.n2(g + self.drop(self.ca(g, ex, ex, need_weights=False)[0]))
        z = torch.cat([z[:, :-1], g], 1)
        return self.n3(z + self.drop(self.ff(z.transpose(1, 2)).transpose(1, 2)))

class TimeXer(nn.Module):
    def __init__(self, C, L, H, P=8, d=64, heads=4, layers=2):
        super().__init__(); self.P = P; self.N = L // P
        self.pemb = nn.Linear(P, d); self.pos = nn.Parameter(torch.randn(1, self.N, d) * .02)
        self.glob = nn.Parameter(torch.randn(1, 1, d) * .02); self.xemb = nn.Linear(L, d)
        self.layers = nn.ModuleList(TimeXerLayer(d, heads, 2 * d) for _ in range(layers))
        self.head = nn.Sequential(nn.Flatten(1), nn.Dropout(.1), nn.Linear((self.N + 1) * d, H))
    def forward(self, x):
        xn, mu, sd = revin(x)
        p = self.pemb(xn[:, -self.N * self.P:, 0].reshape(len(x), self.N, self.P)) + self.pos
        z = torch.cat([p, self.glob.expand(len(x), -1, -1)], 1)
        ex = self.xemb(xn[..., 1:].transpose(1, 2)) if x.shape[-1] > 1 else self.xemb(xn.transpose(1, 2))
        for lyr in self.layers:
            z = lyr(z, ex)
        return self.head(z) * sd[:, 0, :1] + mu[:, 0, :1]