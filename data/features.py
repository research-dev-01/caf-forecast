"""
Per-bin channels and dataset characterisation (paper Table I + Fig. 1).
Corresponds to Cell 6 of cmsnts_latest.ipynb.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from config import BIN_MS, DATASETS, TABLES
from utils.helpers import stamp, checkpoint, savefig

CHANNELS = ['rate_mbps', 'kpps', 'mean_size', 'size_std', 'tcp_share', 'udp_share', 'syn_log', 'finrst_log',
            'small_share', 'large_share']
BASIC_CHANNELS = [0, 1, 2, 3]   # volume-only channels (used by the 'w/o header features' ablation)

def to_channels(acc, bin_ms):
    sec = bin_ms / 1000
    pk = acc[:, 0].astype(np.float64); by = acc[:, 1].astype(np.float64)
    safe_pk = np.maximum(pk, 1); safe_by = np.maximum(by, 1)
    mean = np.where(pk > 0, by / safe_pk, 0)
    var = np.maximum(np.where(pk > 0, acc[:, 2] / safe_pk, 0) - mean ** 2, 0)
    ch = np.stack([by * 8 / sec / 1e6, pk / sec / 1e3, mean, np.sqrt(var),
                   np.where(by > 0, acc[:, 3] / safe_by, 0), np.where(by > 0, acc[:, 4] / safe_by, 0),
                   np.log1p(acc[:, 5]), np.log1p(acc[:, 6]),
                   np.where(pk > 0, acc[:, 7] / safe_pk, 0), np.where(pk > 0, acc[:, 8] / safe_pk, 0)], 1)
    assert np.isfinite(ch).all()
    return ch.astype(np.float32)

def aggregate_bins(acc, factor):
    n = len(acc) // factor * factor
    return acc[:n].reshape(-1, factor, acc.shape[1]).sum(1)

def hurst_aggvar(x):
    ms, vs = [], []
    for lv in range(0, 10):
        m = 2 ** lv; n = len(x) // m
        if n < 32:
            break
        ms.append(m); vs.append(x[:n * m].reshape(n, m).mean(1).var())
    return 1 + np.polyfit(np.log(ms), np.log(vs), 1)[0] / 2

DATA_TABLE = None
ds_summary = None

def compute_dataset_characterization(TRACE, TRACE_META):
    global DATA_TABLE, ds_summary
    rows = []
    for ds, cfg in DATASETS.items():
        for split in ('train', 'val', 'test'):
            for d in cfg[split]:
                acc = TRACE[d]; r = to_channels(acc, BIN_MS)[:, 0]; m = TRACE_META[d]
                rows.append({'dataset': ds, 'split': split, 'date': d, 'seconds': m['seconds'],
                             'packets_M': acc[:, 0].sum() / 1e6, 'mean_mbps': r.mean(), 'p99_mbps': np.percentile(r, 99),
                             'cv_10ms': r.std() / r.mean(), 'acf1_10ms': np.corrcoef(r[:-1], r[1:])[0, 1],
                             'hurst': hurst_aggvar(r), 'tcp_byte_share': acc[:, 3].sum() / acc[:, 1].sum(),
                             'udp_byte_share': acc[:, 4].sum() / acc[:, 1].sum(),
                             'syn_per_s': acc[:, 5].sum() / m['seconds'], 'out_of_order': m['out_of_order_packets'],
                             'captured_gt_original': m['captured_over_original'], 'non_ip': m['non_ip_packets']})
    DATA_TABLE = pd.DataFrame(rows)
    DATA_TABLE.to_csv(TABLES / 'capture_statistics.csv', index=False)
    display(DATA_TABLE.round(3))
    ds_summary = DATA_TABLE.groupby('dataset').agg(
        days=('date', 'count'), seconds_per_day=('seconds', 'mean'), packets_M=('packets_M', 'sum'),
        mean_mbps=('mean_mbps', 'mean'), mean_mbps_sd=('mean_mbps', 'std'), p99_mbps=('p99_mbps', 'mean'),
        cv=('cv_10ms', 'mean'), acf1=('acf1_10ms', 'mean'), hurst=('hurst', 'mean'), tcp_share=('tcp_byte_share', 'mean'),
        syn_per_s=('syn_per_s', 'mean'))
    display(ds_summary.round(3))

    fig, axes = plt.subplots(2, len(DATASETS), figsize=(7.2, 3.6), layout='constrained')
    for j, (ds, cfg) in enumerate(DATASETS.items()):
        r = to_channels(TRACE[cfg['test'][0]], BIN_MS)[:, 0]
        seg = r[:2000]; tt = np.arange(len(seg)) * BIN_MS / 1000
        axes[0, j].plot(tt, seg, lw=.4, color='#888'); axes[0, j].plot(tt, pd.Series(seg).rolling(100, min_periods=1).mean(), lw=1, color='C3')
        axes[0, j].set_title(f'{ds} ({cfg["test"][0]})'); axes[0, j].set_xlabel('time (s)')
        axes[0, 0].set_ylabel('rate (Mbit/s), 10 ms bins')
        lags = np.unique(np.logspace(0, np.log10(500), 40).astype(int)); rc = r - r.mean()
        acf = [np.dot(rc[:-k], rc[k:]) / np.dot(rc, rc) for k in lags]
        axes[1, j].semilogx(lags * BIN_MS, acf, marker='.', ms=3); axes[1, j].set_xlabel('lag (ms)')
        axes[1, 0].set_ylabel('autocorrelation')
    savefig(fig, 'fig1_dataset_traces_acf')
    return DATA_TABLE, ds_summary
