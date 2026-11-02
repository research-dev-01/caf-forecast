"""
Compute profiling of every model (FLOPs, latency, memory, power) and edge-deployment benchmark.
Corresponds to Cell 28 and Cell 36 of cmsnts_latest.ipynb.
"""

import time
import subprocess
import threading
import copy
import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from config import (
    DEVICE, PROPOSED, LOOKBACK, HORIZON, T4_PEAK_TFLOPS, TABLES, DEEP_MODELS, CLASSICAL_MODELS
)
from utils.helpers import checkpoint, stamp
from experiments.engine import load_model

from torch.utils.flop_counter import FlopCounterMode

class Call(nn.Module):
    """exposes a non-default entry point (e.g. CAF.forward_cheap) as forward() for timing/counting"""
    def __init__(self, m, method, **kw):
        super().__init__(); self.m, self.method, self.kw = m, method, kw
    def forward(self, x):
        return getattr(self.m, self.method)(x, **self.kw)

def _rnn_flops(mod, shape):
    B, T = (shape[0], shape[1]) if mod.batch_first else (shape[1], shape[0])
    g = {'LSTM': 4, 'GRU': 3}.get(type(mod).__name__, 1); d = 2 if mod.bidirectional else 1
    f, inp = 0, mod.input_size
    for _ in range(mod.num_layers):
        f += 2 * g * mod.hidden_size * (inp + mod.hidden_size) * T * B * d; inp = mod.hidden_size * d
    return f

def count_flops(model, x, method='forward', **kw):
    """FLOPs per window. `model` must be a throw-away instance (it is moved to CPU / train mode)."""
    m = model.cpu().float().train(); x = x.detach().cpu().float(); seen = []
    hooks = [mod.register_forward_hook(lambda mod, a, o: seen.append((mod, tuple(a[0].shape))))
             for mod in m.modules() if isinstance(mod, nn.RNNBase)]
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            getattr(m, method)(x, **kw)
    finally:
        for h in hooks:
            h.remove()
    total = fc.get_total_flops()
    for mod, shp in seen:
        with torch.no_grad(), FlopCounterMode(display=False) as f2:
            mod(torch.zeros(shp))
        a = _rnn_flops(mod, shp)
        if f2.get_total_flops() < .5 * a:                   # kernel bypassed the counter (e.g. oneDNN RNN)
            total += a
    return total / x.shape[0]

def latency(model, x, reps, device):
    model.eval(); ts = []
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device.type == 'cuda' and AMP):
        for _ in range(10):
            model(x)
        for _ in range(reps):
            if device.type == 'cuda':
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); model(x); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
            else:
                t0 = time.perf_counter(); model(x); ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), float(np.percentile(ts, 95)), float(np.percentile(ts, 99))

def _gpu_power():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=2).stdout
        return float(out.strip().split('\n')[0])
    except Exception:
        return float('nan')

class PowerSampler:
    def __enter__(self):
        self.s, self.stop = [], threading.Event()
        def run():
            while not self.stop.is_set():
                self.s.append(_gpu_power()); time.sleep(.05)
        self.t = threading.Thread(target=run, daemon=True); self.t.start(); return self
    def __exit__(self, *a):
        self.stop.set(); self.t.join()
    @property
    def watts(self):
        v = [w for w in self.s if np.isfinite(w)]
        return float(np.mean(v)) if v else float('nan')

def gpu_throughput(model, xb, min_s=3.0):
    if DEVICE.type != 'cuda':
        return dict(windows_per_s=np.nan, peak_infer_mb=np.nan, watts=np.nan, J_per_M=np.nan, dyn_J_per_M=np.nan)
    model.eval()
    with torch.inference_mode(), autocast():
        for _ in range(3):
            model(xb)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated()
        with PowerSampler() as ps:
            t0 = time.perf_counter(); n = 0
            while True:
                model(xb); n += 1
                if n % 5 == 0:
                    torch.cuda.synchronize()
                    if time.perf_counter() - t0 > min_s:
                        break
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
    w = n * len(xb)
    return dict(windows_per_s=w / dt, peak_infer_mb=(torch.cuda.max_memory_allocated() - m0) / 2**20,
                watts=ps.watts, J_per_M=ps.watts * dt / w * 1e6, dyn_J_per_M=(ps.watts - IDLE_W) * dt / w * 1e6)

def cpu_predict(model, D, chans, idx, bs=2048):
    S = D.splits['test']; out = []
    with torch.inference_mode():
        for i in range(0, len(idx), bs):
            x, _ = S.batch(torch.as_tensor(idx[i:i + bs], device=DEVICE), chans)
            out.append(model(x.float().cpu()).float())
    return D.to_mbps(torch.cat(out).numpy())

def adaptive_latency(cheap, full, rate):
    """p50 = expected latency; p95/p99 conservatively add the whole full-model latency when the escalation rate
    exceeds the tail mass (an escalated window runs cheap path + deep path)"""
    return (cheap[0] + rate * full[0], cheap[1] + (full[1] if rate > .05 else 0.0), cheap[2] + (full[2] if rate > .01 else 0.0))

D0 = DATA[DS[0]]; S0 = D0.splits['test']; nthreads = torch.get_num_threads()
with PowerSampler() as _ps:
    time.sleep(2.0)
IDLE_W = _ps.watts
print(f'{GPU_NAME}: idle board power {IDLE_W:.1f} W | CPU: {platform.processor() or platform.machine()}, '
      f'{os.cpu_count()} logical cores')

# --- FLOPs (CAF entry points counted separately: full = E2 path, e1 = cheap exit, cheap = E1 + router) ---
_xc = torch.randn(2, LOOKBACK, len(CHANNELS))
FLOPS = {}
for m in DEEP_MODELS:
    ctor, chans = MODEL_SPECS[m]
    FLOPS[m] = count_flops(ctor(n_in(chans), LOOKBACK, HORIZON), _xc if chans is None else _xc[..., chans])
CAF_FLOPS = {k: count_flops(CAF(len(CHANNELS), LOOKBACK, HORIZON), _xc, method=meth)
             for k, meth in (('full', 'forward'), ('e1', 'forward_e1'), ('cheap', 'forward_cheap'))}
CAF_FLOPS['deep_branch'] = CAF_FLOPS['full'] - CAF_FLOPS['e1']
FLOPS['CAF-E1'] = CAF_FLOPS['e1']

def cascade_flops(rate):
    """average FLOPs per forecast when a fraction `rate` of windows is escalated"""
    return CAF_FLOPS['cheap'] + rate * CAF_FLOPS['deep_branch']

OP_RATE = {ds: float(np.mean([v['test_rate'] for k, v in CASCADE.items() if k[0] == 'main' and k[1] == ds]))
           for ds in DS if any(k[0] == 'main' and k[1] == ds for k in CASCADE)}
if OP_RATE:
    FLOPS['CAF-adaptive'] = cascade_flops(float(np.mean(list(OP_RATE.values()))))
CLASSICAL_FLOPS = {'Persistence': 0, 'MovingAvg': 2, 'EWMA': 3, 'Ridge-AR': 2 * LOOKBACK * HORIZON + HORIZON,
                   'HistGB': np.nan}                       # trees: comparisons, not FLOPs
print('CAF FLOPs/window:', {k: f'{v:,.0f}' for k, v in CAF_FLOPS.items()}, '| operating escalation rate:', OP_RATE)

# --- latency / throughput / memory / energy ---
eff = []
xb_idx = torch.arange(0, min(4096, S0.n), device=DEVICE)
caf_seed = next((k[3] for k in CASCADE if k[0] == 'main' and k[1] == DS[0]), None)
for m in DEEP_MODELS + ['CAF-E1'] + (['CAF-adaptive'] if caf_seed is not None and 'CAF-adaptive' in FLOPS else []):
    base = PROPOSED if m in CASCADE_MODELS else m
    ctor, chans = MODEL_SPECS[base]
    x1, _ = S0.batch(xb_idx[:1], chans); xb, _ = S0.batch(xb_idx, chans)
    if m == 'CAF-adaptive':
        trained, ck = load_caf('main', D0, PROPOSED, MODEL_SPECS[PROPOSED], caf_seed)
        gm = Call(trained, 'forward_adaptive', tau=ck['tau']); rate = OP_RATE.get(DS[0], float(np.mean(list(OP_RATE.values()))))
        cheap_g = latency(Call(trained, 'forward_cheap'), x1, 200, DEVICE); full_g = latency(trained, x1, 200, DEVICE)
        g_lat = adaptive_latency(cheap_g, full_g, rate)
        tp = gpu_throughput(gm, xb)
        with torch.inference_mode(), autocast():
            tp['escalated_in_batch'] = float(gm(xb)[1].float().mean())
        torch.set_num_threads(1)
        tc = copy.deepcopy(trained).cpu().float()
        cheap_c = latency(Call(tc, 'forward_cheap'), x1.float().cpu(), 100, torch.device('cpu'))
        full_c = latency(tc, x1.float().cpu(), 100, torch.device('cpu'))
        c_lat = adaptive_latency(cheap_c, full_c, rate)
        torch.set_num_threads(nthreads); params = PARAMS[PROPOSED]
    else:
        mod = ctor(n_in(chans), LOOKBACK, HORIZON).to(DEVICE).eval()
        gm = Call(mod, 'forward_e1') if m == 'CAF-E1' else mod
        g_lat = latency(gm, x1, 200, DEVICE); tp = gpu_throughput(gm, xb)
        torch.set_num_threads(1)
        mc = ctor(n_in(chans), LOOKBACK, HORIZON).eval()
        c_lat = latency(Call(mc, 'forward_e1') if m == 'CAF-E1' else mc, x1.float().cpu(), 100, torch.device('cpu'))
        torch.set_num_threads(nthreads)
        params = (mc.logits.numel() + sum(p.numel() for p in mc.lin.parameters())) if m == 'CAF-E1' else PARAMS[m]
    runs = [v for k, v in HISTORY.items() if k[0] == 'main' and k[2] == base]
    seen = np.mean([v.get('train_windows_seen', np.nan) for v in runs]) if runs else np.nan
    eff.append(dict(model=m, params=params, size_kB=params * 4 / 1024, MFLOPs=FLOPS[m] / 1e6, MMACs=FLOPS[m] / 2e6,
                    gpu_ms_b1=g_lat[0], gpu_p95_ms=g_lat[1], gpu_p99_ms=g_lat[2],
                    cpu1t_ms_b1=c_lat[0], cpu1t_p95_ms=c_lat[1], cpu1t_p99_ms=c_lat[2],
                    links_per_core=10 / c_lat[0], gpu_windows_per_s=tp['windows_per_s'],
                    achieved_TFLOPs=FLOPS[m] * tp['windows_per_s'] / 1e12,
                    pct_T4_fp16_peak=100 * FLOPS[m] * tp['windows_per_s'] / 1e12 / T4_PEAK_TFLOPS['fp16_tensor'],
                    peak_infer_mb=tp['peak_infer_mb'], board_W=tp['watts'], J_per_M_forecasts=tp['J_per_M'],
                    dyn_J_per_M_forecasts=tp['dyn_J_per_M'],
                    train_s=np.mean([v['train_seconds'] for v in runs]) if runs and m not in CASCADE_MODELS else np.nan,
                    sec_per_epoch=np.mean([v['sec_per_epoch'] for v in runs]) if runs and m not in CASCADE_MODELS else np.nan,
                    peak_train_mb=np.mean([v.get('peak_train_mem_mb', np.nan) for v in runs]) if runs else np.nan,
                    train_PFLOP=3 * FLOPS[base] * seen / 1e15 if m not in CASCADE_MODELS else np.nan,
                    avg_mae_rank=MEAN_RANK.get(m, np.nan)))
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
EFF = pd.DataFrame(eff).set_index('model')

# --- CPU dynamic INT8 quantisation: latency and accuracy change on a fixed test subsample (trained seed) ---
if RUN_INT8 and remaining_min() > 12:
    try:
        torch.backends.mha.set_fastpath_enabled(False)
    except Exception:
        pass
    sub = np.linspace(0, S0.n - 1, min(10000, S0.n)).astype(np.int64); y_sub = S0.y[sub]
    for m in DEEP_MODELS + ['CAF-E1']:
        if remaining_min() < 8:
            print('INT8 study stopped: time budget'); break
        base = PROPOSED if m == 'CAF-E1' else m
        seeds = sorted(k[3] for k in DEEP_PREDS if k[0] == 'main' and k[1] == DS[0] and k[2] == base)
        if not seeds:
            continue
        try:
            torch.set_num_threads(nthreads)
            fp = load_model('main', D0, base, MODEL_SPECS[base], seeds[0]).cpu().float().eval()
            q8 = torch.ao.quantization.quantize_dynamic(fp, {nn.Linear, nn.LSTM, nn.GRU}, dtype=torch.qint8).eval()
            if m == 'CAF-E1':
                fp, q8 = Call(fp, 'forward_e1'), Call(q8, 'forward_e1')
            chans = MODEL_SPECS[base][1]
            mae32 = np.abs(cpu_predict(fp, D0, chans, sub) - y_sub).mean()
            mae8 = np.abs(cpu_predict(q8, D0, chans, sub) - y_sub).mean()
            x1, _ = S0.batch(torch.arange(0, 1, device=DEVICE), chans)
            torch.set_num_threads(1)
            EFF.loc[m, 'cpu1t_int8_ms_b1'] = latency(q8, x1.float().cpu(), 100, torch.device('cpu'))[0]
            EFF.loc[m, 'int8_dMAE_pct'] = 100 * (mae8 / mae32 - 1)
        except Exception as e:
            print(f'INT8 {m}: not supported ({type(e).__name__}: {str(e)[:80]})')
        finally:
            torch.set_num_threads(nthreads)
    try:
        torch.backends.mha.set_fastpath_enabled(True)
    except Exception:
        pass
for c in ('cpu1t_int8_ms_b1', 'int8_dMAE_pct'):
    if c not in EFF:
        EFF[c] = np.nan
EFF['int8_speedup'] = EFF.cpu1t_ms_b1 / EFF.cpu1t_int8_ms_b1
EFF.to_csv(TABLES / 'efficiency_full.csv'); display(EFF.round(4))

TABLE8 = pd.DataFrame(index=EFF.index)
TABLE8['Params'] = EFF.params.map(lambda v: f'{v / 1e3:.1f}k')
TABLE8['MFLOPs'] = EFF.MFLOPs.map(lambda v: f'{v:.3f}' if v < 1 else f'{v:.2f}')
TABLE8['GPU ms p50/p99'] = [f'{a:.2f}/{b:.2f}' for a, b in zip(EFF.gpu_ms_b1, EFF.gpu_p99_ms)]
TABLE8['CPU ms p50/p99'] = [f'{a:.2f}/{b:.2f}' for a, b in zip(EFF.cpu1t_ms_b1, EFF.cpu1t_p99_ms)]
TABLE8['INT8 ms (dMAE%)'] = [('--' if not np.isfinite(a) else f'{a:.2f} ({b:+.1f})') for a, b in
                             zip(EFF.cpu1t_int8_ms_b1, EFF.int8_dMAE_pct)]
TABLE8['Links/core'] = EFF.links_per_core.map(lambda v: f'{v:.0f}')
TABLE8['GPU kwin/s'] = EFF.gpu_windows_per_s.map(lambda v: f'{v / 1e3:.0f}')
TABLE8['TFLOP/s'] = EFF.achieved_TFLOPs.map(lambda v: f'{v:.2f}')
TABLE8['J/1M fc'] = EFF.J_per_M_forecasts.map(lambda v: f'{v:.1f}' if np.isfinite(v) else '--')
TABLE8['Train PFLOP'] = EFF.train_PFLOP.map(lambda v: f'{v:.3f}' if np.isfinite(v) else '--')
TABLE8.index.name = 'Model'; display(TABLE8)
to_latex(TABLE8, TABLES / 'table8_compute_profile.tex',
         f'Compute profile per forecast window (L={LOOKBACK}, H={HORIZON}): FLOPs, single-window latency on {GPU_NAME} '
         '(fp16 AMP) and one CPU core (fp32; dynamic INT8 with test-MAE change), links served per core at the 10 ms '
         'cadence, batch throughput, achieved TFLOP/s, board energy per million forecasts and training compute. '
         'CAF-adaptive: expected values at the validation-calibrated escalation rate.', 'tab:compute',
         best_lower={'MFLOPs': EFF.MFLOPs, 'GPU ms p50/p99': EFF.gpu_ms_b1, 'CPU ms p50/p99': EFF.cpu1t_ms_b1,
                     'Links/core': -EFF.links_per_core, 'J/1M fc': EFF.J_per_M_forecasts})
checkpoint('analysis cell 27')


#       plus clearly-labelled projections to embedded CPUs. Inference only; models are loaded from saved checkpoints.
import io as _io, copy as _copy
from torch.utils.flop_counter import FlopCounterMode
EDGE_REF_GFLOPS = {'Raspberry Pi 4 (A72)': 3.5, 'Raspberry Pi 5 (A76)': 10.0, 'Jetson Nano CPU (A57)': 2.5}  # approx. 1-core SGEMM
EDGE_REPS, EDGE_Q_N = 200, 4096

def _rnn_flops(mod, inp):
    B, T = inp.shape[0], inp.shape[1]; g = 3 if isinstance(mod, nn.GRU) else 4
    f, d_in = 0, inp.shape[-1]
    for _ in range(mod.num_layers):
        f += 2 * g * (d_in * mod.hidden_size + mod.hidden_size * mod.hidden_size) * T * B + 6 * g * mod.hidden_size * T * B
        d_in = mod.hidden_size
    return f

def flops_per_forecast(model, x):
    """FLOPs per forecast (FlopCounterMode; recurrent layers added analytically if the counter misses them)"""
    model = model.float().eval(); seen = []
    hooks = [m.register_forward_hook(lambda m, i, o: seen.append((m, i[0].detach()))) for m in model.modules()
             if isinstance(m, (nn.GRU, nn.LSTM))]
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    total = fc.get_total_flops()
    for mod, inp in seen:
        with FlopCounterMode(display=False) as f2, torch.no_grad():
            mod(inp)
        a = _rnn_flops(mod, inp)
        if f2.get_total_flops() < .5 * a:
            total += a - f2.get_total_flops()
    return total / x.shape[0]

def _lat(model, x, reps=EDGE_REPS):
    ts = []
    with torch.inference_mode():
        for _ in range(10):
            model(x)
        for _ in range(reps):
            t0 = time.perf_counter(); model(x); ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), float(np.percentile(ts, 99))

def _thr(model, x, min_s=1.5):
    with torch.inference_mode():
        model(x); n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < min_s:
            model(x); n += len(x)
    return n / (time.perf_counter() - t0)

def _size_kb(model):
    buf = _io.BytesIO(); torch.save(model.state_dict(), buf); return buf.getbuffer().nbytes / 1024

def _host_gflops():
    torch.set_num_threads(1); a = torch.randn(256, 256); b = torch.randn(256, 256)
    for _ in range(5):
        a @ b
    t0 = time.perf_counter(); n = 0
    while time.perf_counter() - t0 < 1.0:
        a @ b; n += 1
    return 2 * 256 ** 3 * n / (time.perf_counter() - t0) / 1e9

def edge_profile(label, model_fn, chans, D, preds_fp32=None, train_s=np.nan):
    """model_fn() -> fresh fp32 CPU model with trained weights"""
    S = D.splits['test']; nthr = torch.get_num_threads()
    idx = torch.arange(0, min(EDGE_Q_N, S.n), device=DEVICE)
    xq, _ = S.batch(idx, chans); xq = xq.float().cpu(); x1 = xq[:1].contiguous(); xb = xq[:256].contiguous()
    m = model_fn().cpu().float().eval()
    row = dict(model=label, params=count_params(m), size_fp32_kB=_size_kb(m), MFLOPs=flops_per_forecast(m, xq[:2]) / 1e6)
    torch.set_num_threads(1)
    row['cpu1_p50_ms'], row['cpu1_p99_ms'] = _lat(m, x1); row['cpu1_fps'] = _thr(m, xb)
    try:
        mq = torch.ao.quantization.quantize_dynamic(_copy.deepcopy(m), {nn.Linear, nn.GRU, nn.LSTM}, dtype=torch.qint8).eval()
        row['int8_p50_ms'], _ = _lat(mq, x1, 100); row['size_int8_kB'] = _size_kb(mq)
        with torch.inference_mode():
            p32, p8 = m(xq).float().numpy(), mq(xq).float().numpy()
        y = S.y[:len(xq)]
        row['int8_dMAE_pct'] = 100 * (np.abs(D.to_mbps(p8) - y).mean() / np.abs(D.to_mbps(p32) - y).mean() - 1)
    except Exception as e:
        row.update(int8_p50_ms=np.nan, size_int8_kB=np.nan, int8_dMAE_pct=np.nan); print(f'  INT8 {label}: {e}')
    torch.set_num_threads(min(4, nthr)); row['cpu4_fps'] = _thr(m, xb); torch.set_num_threads(nthr)
    if DEVICE.type == 'cuda':
        mg = model_fn().to(DEVICE).eval(); xg = x1.to(DEVICE)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base_mem = torch.cuda.memory_allocated()
        ts = []
        with torch.inference_mode():
            for _ in range(10):
                mg(xg)
            for _ in range(100):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); mg(xg); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
        row['gpu_b1_ms'] = float(np.median(ts)); row['gpu_peak_MB'] = (torch.cuda.max_memory_allocated() - base_mem) / 2 ** 20 \
            + sum(p.numel() * 4 for p in mg.parameters()) / 2 ** 20
        del mg; torch.cuda.empty_cache()
    row['train_gpu_s'] = train_s
    row['links_per_core'] = BIN_MS / row['cpu1_p99_ms']
    row['core_s_per_1M'] = row['cpu1_p50_ms'] * 1e3
    for dev, gf in EDGE_REF_GFLOPS.items():
        row[f'proj_ms {dev}'] = row['cpu1_p50_ms'] * HOST_GFLOPS / gf
    return row

HOST_GFLOPS = _host_gflops(); torch.set_num_threads(max(1, os.cpu_count() or 1))
print(f'Host 1-core SGEMM: {HOST_GFLOPS:.1f} GFLOP/s; edge projections scale measured 1-core latency by the throughput ratio')

def _train_s(study, ds, m):
    v = [h['train_seconds'] for k, h in HISTORY.items() if k[0] == study and k[1] == ds and k[2] == m and 'train_seconds' in h]
    return float(np.mean(v)) if v else np.nan

# 20f | Edge efficiency (Table XV) and data scale (Table XVI): CAF-E1, CAF, CAF-X-eff vs every SOTA model; Figs. S11-S14
EDGE_DS = DS[0]

class _E1(nn.Module):                                                # CAF cheap exit as a stand-alone deployable model
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        return self.m.forward_e1(x)

def _first_seed(study, ds, m):
    return sorted(k[3] for k in DEEP_PREDS if k[0] == study and k[1] == ds and k[2] == m)[0]

def _edge_all():
    rows = []
    for m in DEEP_MODELS:
        if any(k[0] == 'main' and k[1] == EDGE_DS and k[2] == m for k in DEEP_PREDS):
            s = _first_seed('main', EDGE_DS, m)
            rows.append(edge_profile(m, lambda m=m, s=s: load_model('main', DATA[EDGE_DS], m, MODEL_SPECS[m], s),
                                     MODEL_SPECS[m][1], DATA[EDGE_DS], train_s=np.nanmean([_train_s('main', ds, m) for ds in DS])))
    s = _first_seed('main', EDGE_DS, PROPOSED)
    rows.append(edge_profile('CAF-E1', lambda: _E1(load_model('main', DATA[EDGE_DS], PROPOSED, MODEL_SPECS[PROPOSED], s)),
                             MODEL_SPECS[PROPOSED][1], DATA[EDGE_DS], train_s=np.nan))
    return pd.DataFrame(rows).set_index('model')

EDGE = cache_obj('edge_benchmark_v1', _edge_all)
if 'CAF-E1' in EFF.index:                                          # count only the exit's own parameters (E0 + E1)
    EDGE.loc['CAF-E1', 'params'] = EFF.loc['CAF-E1', 'params']; EDGE.loc['CAF-E1', 'size_fp32_kB'] = EFF.loc['CAF-E1', 'params'] * 4 / 1024

# composite rows: CAF-X-eff per dataset = E1 always + escalation rate x target (expected cost, measured components)
ADD = ['MFLOPs', 'cpu1_p50_ms', 'cpu1_p99_ms', 'int8_p50_ms', 'gpu_b1_ms', 'core_s_per_1M'] + \
      [c for c in EDGE.columns if c.startswith('proj_ms')]
if 'EFF_SEL' in globals():
    for ds, sel in EFF_SEL.items():
        parts = _parts(sel['target']); g = FAIR[(FAIR.dataset == ds) & (FAIR.model == 'CAF-X-eff')]
        if not len(g) or not all(p in EDGE.index for p in parts):
            continue
        rate = float(np.clip((g.MFLOPs.iloc[0] * 1e6 - CAF_FLOPS['cheap']) / max(FLOPS[sel['target']], 1), 0, 1))
        r = {c: EDGE.loc['CAF-E1', c] + rate * sum(EDGE.loc[p, c] for p in parts) for c in ADD}
        r['cpu1_p99_ms'] = EDGE.loc['CAF-E1', 'cpu1_p99_ms'] + sum(EDGE.loc[p, 'cpu1_p99_ms'] for p in parts)   # worst case
        r.update(params=EDGE.loc['CAF-E1', 'params'] + sum(EDGE.loc[p, 'params'] for p in parts),
                 size_fp32_kB=EDGE.loc['CAF-E1', 'size_fp32_kB'] + sum(EDGE.loc[p, 'size_fp32_kB'] for p in parts),
                 gpu_peak_MB=max(EDGE.loc[p, 'gpu_peak_MB'] for p in parts + ['CAF-E1']),
                 cpu1_fps=1e3 / r['cpu1_p50_ms'], cpu4_fps=np.nan, train_gpu_s=sum(EDGE.loc[p, 'train_gpu_s'] for p in parts),
                 escalation_rate=rate)
        r['links_per_core'] = BIN_MS / r['cpu1_p99_ms']
        EDGE.loc[f'CAF-X-eff ({ds})'] = pd.Series(r)

def _acc(m):
    v = []
    for ds in DS:
        best = AVG_MAE[ds].drop(['CAF-adaptive', 'CAF-E1'], errors='ignore').min()
        if m.startswith('CAF-X-eff'):
            if f'({ds})' not in m:
                continue
            mae = FAIR[(FAIR.dataset == ds) & (FAIR.model == 'CAF-X-eff')].MAE.iloc[0]
        else:
            mae = AVG_MAE.loc[m, ds] if m in AVG_MAE.index else np.nan
        v.append(100 * (mae / best - 1))
    return np.nanmean(v) if v else np.nan
EDGE['MAE_above_best_pct'] = [_acc(m) for m in EDGE.index]
EDGE.to_csv(TABLES / 'table15_edge_benchmark.csv'); display(EDGE.round(3))
t15 = EDGE[['MAE_above_best_pct', 'params', 'MFLOPs', 'cpu1_p50_ms', 'cpu1_p99_ms', 'int8_p50_ms', 'cpu4_fps', 'gpu_b1_ms',
            'gpu_peak_MB', 'size_fp32_kB', 'core_s_per_1M', 'proj_ms Raspberry Pi 4 (A72)']].copy()
t15.columns = ['ΔMAE %', 'Params', 'MFLOPs', 'CPU p50 ms', 'CPU p99 ms', 'INT8 ms', 'CPU-4T fc/s', 'T4 ms', 'T4 MB', 'fp32 kB',
               'core-s/1M', 'Pi 4 ms*']
to_latex(t15.round(3).astype(str), TABLES / 'table15_edge_benchmark.tex',
         'Edge-deployment benchmark (batch 1; CPU = 1 thread of this host). CAF-X-eff rows: expected cost = always-on E1 + '
         'measured escalation rate × target (p99: worst case, both paths). *Projected from measured 1-core latency by the '
         'SGEMM throughput ratio (not a measurement).', 'tab:edge',
         best_lower={c: pd.to_numeric(t15[c], errors='coerce') * (-1 if c == 'CPU-4T fc/s' else 1) for c in t15.columns})

# ---- data scale ----
LOW = []
for ds in DS:
    Dl = DATA_LOW.get(ds) if 'DATA_LOW' in globals() else None
    if Dl is None:
        continue
    yt = Dl.splits['test'].y
    for m in LOWDATA_MODELS:
        ks = sorted(k for k in DEEP_PREDS if k[0] == 'lowdata' and k[1] == Dl.name and k[2] == m)
        if len(ks) < len(LOWDATA_SEEDS):
            continue
        maes = [np.abs(DEEP_PREDS[k]['test'] - yt).mean() for k in ks]
        LOW.append(dict(dataset=ds, model=m, mae_1day=np.mean(maes), sd_1day=np.std(maes, ddof=1),
                        mae_3day=AVG_MAE.loc[m, ds] if m in AVG_MAE.index else np.nan))
        if m == PROPOSED:                                          # CAF-E1 from the same low-data checkpoints
            e1 = []
            for k in ks:
                mm = _E1(load_model('lowdata', Dl, PROPOSED, MODEL_SPECS[PROPOSED], k[3]))
                e1.append(np.abs(predict(mm, Dl, 'test', MODEL_SPECS[PROPOSED][1]) - yt).mean()); del mm
            LOW.append(dict(dataset=ds, model='CAF-E1', mae_1day=np.mean(e1), sd_1day=np.std(e1, ddof=1),
                            mae_3day=AVG_MAE.loc['CAF-E1', ds] if 'CAF-E1' in AVG_MAE.index else np.nan))
LOW = pd.DataFrame(LOW)
if len(LOW):
    LOW['degradation_pct'] = 100 * (LOW.mae_1day / LOW.mae_3day - 1)
    LOW.to_csv(TABLES / 'table16_low_data.csv', index=False); display(LOW.round(3))
    t16 = LOW.assign(cell=[f'{fmt_pm(a, b)} ({d:+.1f}%)' for a, b, d in zip(LOW.mae_1day, LOW.sd_1day, LOW.degradation_pct)]) \
        .pivot(index='model', columns='dataset', values='cell')
    t16.index.name = 'Model'
    to_latex(t16, TABLES / 'table16_low_data.tex',
             'Low-data regime: test MAE with ONE training day (mean ± SD, 3 seeds) and change vs three training days.',
             'tab:lowdata', best_lower={c: LOW[LOW.dataset == c].set_index('model').mae_1day for c in t16.columns})

EMOD = [m for m in EDGE.index if m in ('CAF-E1', PROPOSED, 'TimeXer', 'TimeMixer', 'TimesNet', 'TSMixer', 'iTransformer',
                                        'PatchTST', 'N-HiTS', 'DLinear') or m.startswith('CAF-X-eff')]
FIG_LOG3 = []

def _run(name, fn):
    try:
        fn(name); FIG_LOG3.append((name, 'ok'))
    except Exception as e:
        plt.close('all'); FIG_LOG3.append((name, f'{type(e).__name__}: {e}')); print(f'[{name}] skipped: {e}')

def figS11(name):
    axes_def = [('MFLOPs', 'MFLOPs / forecast'), ('cpu1_p50_ms', 'CPU 1-core p50 (ms)'), ('cpu1_p99_ms', 'CPU 1-core p99 (ms)'),
                ('int8_p50_ms', 'INT8 CPU (ms)'), ('gpu_peak_MB', 'T4 peak memory (MB)'), ('proj_ms Raspberry Pi 4 (A72)', 'Raspberry Pi 4 (ms, proj.)')]
    fig, axes = plt.subplots(2, 3, figsize=(COL2, 4.0), layout='constrained')
    E = EDGE.loc[EMOD]; E = E[E.MAE_above_best_pct < 3]
    for ax, (c, xl) in zip(axes.ravel(), axes_def):
        for m, r in E.iterrows():
            col = pc('CAF-X') if m.startswith('CAF-X') else pc(m)
            ax.scatter(max(r[c], 1e-4), r.MAE_above_best_pct, s=45 if m.startswith('CAF') else 18, color=col,
                       marker='*' if m.startswith('CAF') else 'o', edgecolor='k', lw=.3, zorder=3)
            ax.annotate(m.replace('CAF-X-eff ', 'X-eff '), (max(r[c], 1e-4), r.MAE_above_best_pct), fontsize=4.2, xytext=(2, 2),
                        textcoords='offset points')
        if 'ms' in c:
            ax.axvline(BIN_MS, color='#B22222', lw=.6, ls=':')
        ax.set_xscale('log'); ax.set_xlabel(xl, fontsize=6)
    for ax in axes[:, 0]:
        ax.set_ylabel('mean MAE above best (%)')
    fig.suptitle('Accuracy vs six deployment-cost axes (red dotted: 10 ms real-time deadline)', fontsize=7)
    savefig(fig, name)

def figS12(name):
    cols = [c for c in EDGE.columns if c.startswith('proj_ms')]
    E = EDGE.loc[EMOD, ['cpu1_p50_ms'] + cols]
    fig, ax = plt.subplots(figsize=(COL2, 2.5), layout='constrained')
    w = .8 / len(E.columns); x = np.arange(len(E))
    for j, c in enumerate(E.columns):
        ax.bar(x + (j - (len(E.columns) - 1) / 2) * w, E[c], w,
               label='this host (measured)' if c == 'cpu1_p50_ms' else c.replace('proj_ms ', '') + ' (proj.)')
    ax.axhline(BIN_MS, color='#B22222', lw=.8, ls='--')
    ax.set_yscale('log'); ax.set_xticks(x, [m.replace('CAF-X-eff ', 'X-eff ') for m in E.index], rotation=35, fontsize=5)
    ax.set_ylabel('latency per forecast (ms, 1 core)'); ax.legend(fontsize=5, ncol=4)
    savefig(fig, name)

def figS13(name):
    fig, axes = plt.subplots(1, len(DS), figsize=(COL2, 2.3), layout='constrained')
    for ax, ds in zip(np.atleast_1d(axes), DS):
        g = LOW[LOW.dataset == ds]; best = min(g.mae_1day.min(), g.mae_3day.min())
        for r in g.itertuples():
            ax.plot([1, 3], [100 * (r.mae_1day / best - 1), 100 * (r.mae_3day / best - 1)], '-o', ms=3, color=pc(r.model),
                    lw=1.5 if r.model.startswith('CAF') else .8, label=r.model)
        ax.set_xticks([1, 3], ['1 day', '3 days']); ax.set_title(ds); ax.set_xlabel('training data')
    np.atleast_1d(axes)[0].set_ylabel('test MAE above best (%)'); np.atleast_1d(axes)[-1].legend(fontsize=5)
    savefig(fig, name)

def figS14(name):
    E = EDGE.loc[[m for m in EMOD if not m.startswith('CAF-X-eff') or m.endswith(f'({EDGE_DS})')]]
    fig, ax = plt.subplots(figsize=(3.5, 2.6), layout='constrained')
    ax.scatter(E.links_per_core, E.MAE_above_best_pct, s=10 + 60 * np.sqrt(E.size_fp32_kB / E.size_fp32_kB.max()),
               c=[pc('CAF-X') if m.startswith('CAF-X') else pc(m) for m in E.index], edgecolor='k', lw=.3)
    for m, r in E.iterrows():
        ax.annotate(m.replace('CAF-X-eff ', 'X-eff '), (r.links_per_core, r.MAE_above_best_pct), fontsize=4.5, xytext=(2, 2),
                    textcoords='offset points')
    ax.set_xscale('log'); ax.set_xlabel('links served per CPU core at 10 ms cadence (p99)'); ax.set_ylabel('MAE above best (%)')
    ax.set_title('serving capacity vs accuracy (bubble ∝ √model size)', fontsize=6.5)
    savefig(fig, name)

_run('figS11_accuracy_vs_cost', figS11); _run('figS12_edge_feasibility', figS12); _run('figS14_capacity', figS14)
if len(LOW):
    _run('figS13_data_scale', figS13)
print(FIG_LOG3)
checkpoint('edge + low-data', force=True)
