"""
Configuration and experiment environment setup for CAF.
Corresponds to Cell 1 and Cell 2 of cmsnts_latest.ipynb.
"""

import os, sys, io, re, gc, json, time, math, zlib, socket, ssl, hashlib, random, platform, warnings, zipfile, copy
import subprocess, threading
import http.client, urllib.request, urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

NOTEBOOK_START = time.time()
try:
    import numba
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'numba'], check=True)
    import numba
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from scipy import stats, signal
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings('ignore', category=UserWarning)

PROTOCOL = 'COMSNETS27-SysAI-CAF-v2'   # v2: scale-equivariant exits + scale-free router, 300 s captures
TIME_BUDGET_HOURS = 5.0        # total wall-clock budget for this session (Colab T4 sessions allow ~5-6 h)
SEEDS = [11, 29, 47, 83, 101]  # independent training seeds for every deep model
BIN_MS = 10                    # aggregation bin = forecast cadence
LOOKBACK = 64                  # 64 bins = 640 ms causal history
HORIZON = 10                   # direct forecasts for 1..10 bins ahead (10..100 ms)
REPORT_STEPS = [1, 5, 10]      # horizons shown in the paper tables (10, 50, 100 ms)
CAPTURE_SECONDS = 300          # time-based prefix of every capture (s); 180 if bandwidth is poor, 0 = whole capture
MAX_EPOCHS = 30
PATIENCE = 5                   # early stopping on validation MAE
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
USE_AMP = True                 # fp16 autocast on the T4
DOWNLOAD_WORKERS = 6           # concurrent MAWI streams
BOOTSTRAP_REPS = 1000
BOOTSTRAP_BLOCK_S = 1.0        # moving-block length (seconds) for bootstrap and DM HAC bandwidth
ALPHA = 0.05

# --- compute-adaptive cascade (CAF) ---
BUDGET_GRID = [0, .05, .1, .15, .2, .3, .4, .5, .75, 1.0]   # fraction of windows escalated to the deep exit
NONINF_MARGIN = 0.01          # non-inferiority margin of CAF-adaptive vs full CAF on test (reported with 95% CI)
CASCADE_TOL = 0.005            # operating point: smallest budget whose VALIDATION MAE is within 0.5 % of the full model
ROUTER_EPOCHS = 12

# --- secondary studies (ordered by importance; the time guard drops the tail first) ---
RUN_ABLATIONS = True
RUN_TRANSFER = True
RUN_LOOKBACK_STUDY = True      # L in {16, 32, 128}: compute/accuracy sensitivity
RUN_RESOLUTION_STUDY = True    # 50 ms bins (the v1 run used only ~1.8 h of the 5 h budget; time guard drops it if short)
RUN_INT8 = True                # CPU dynamic INT8 quantisation (latency + accuracy)
ANALYSIS_RESERVE_MIN = 35      # minutes always kept for cascade/efficiency/statistics/export
ALLOW_CPU = os.environ.get('ALLOW_CPU', '0').lower() in ('1', 'true', 'yes') # debugging only; configurable via env var

# T4 datasheet peaks used to express achieved throughput as utilisation
T4_PEAK_TFLOPS = {'fp16_tensor': 65.0, 'fp32': 8.1}
MAWI_URL = 'https://mawi.wide.ad.jp/mawi/samplepoint-F/{year}/{date}1400.pcap.gz'
MAWI_MIRROR = 'http://mawi.nezu.wide.ad.jp/mawi/samplepoint-F/{year}/{date}1400.pcap.gz'   # used if the primary fails

DATASETS = {
    'MAWI-2019': {'link': 'transit only', 'era': '2019 (pre-pandemic)',
                  'train': ['20190305', '20190312', '20190319'], 'val': ['20190326'],
                  'test': ['20190402', '20190409', '20190416']},
    'MAWI-2022': {'link': 'transit only', 'era': '2022 (post-pandemic)',
                  'train': ['20220308', '20220315', '20220322'], 'val': ['20220329'],
                  'test': ['20220405', '20220412', '20220419']},
    'MAWI-2025': {'link': 'transit + BBIX IX', 'era': '2025 (current)',
                  'train': ['20250902', '20250909', '20250916'], 'val': ['20250930'],
                  'test': ['20251007', '20251014', '20251021']},
}


try:
    from utils.helpers import _restore_checkpoints
    _restore_checkpoints()
except Exception:
    pass

ALL_DATES = [d for ds in DATASETS.values() for role in ('train', 'val', 'test') for d in ds[role]]

# ---- device detection ----
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    GPU_NAME = torch.cuda.get_device_name(0)
else:
    assert ALLOW_CPU, 'Select Runtime > Change runtime type > T4 GPU (or set ALLOW_CPU=True).'
    DEVICE = torch.device('cpu')
    GPU_NAME = 'cpu'
AMP = bool(USE_AMP and DEVICE.type == 'cuda')

# ---- storage: Google Drive (Colab) > /kaggle/working (Kaggle) > local ----
PLATFORM = 'kaggle' if Path('/kaggle/working').is_dir() else 'colab' if Path('/content').is_dir() else 'local'
PERSISTENT, ROOT = False, None
if os.environ.get('NCC_ROOT'):
    ROOT, PERSISTENT = Path(os.environ['NCC_ROOT']), True
elif PLATFORM == 'colab':
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        ROOT, PERSISTENT = Path('/content/drive/MyDrive/COMSNETS2027_SysAI_CAF'), True
    except Exception as mount_error:
        print('Google Drive unavailable:', type(mount_error).__name__, str(mount_error)[:150])
if ROOT is None:
    ROOT = {'kaggle': Path('/kaggle/working'), 'colab': Path('/content')}.get(PLATFORM, Path.cwd()) / 'COMSNETS2027_SysAI_CAF'
ROOT.mkdir(parents=True, exist_ok=True)
CKPT_DIR = ROOT.parent / 'CMS2027_checkpoints'
CHECKPOINT_EVERY_MIN = 12
SHARDS_NEEDED = not (PERSISTENT and PLATFORM == 'colab')
IMMUTABLE_PARTS = {'trace_cache', 'ckpt', 'preds', 'hist', 'cache'}

CONFIG = {k: globals()[k] for k in ['PROTOCOL', 'BIN_MS', 'LOOKBACK', 'HORIZON', 'CAPTURE_SECONDS', 'MAX_EPOCHS',
                                     'PATIENCE', 'BATCH_SIZE', 'LR', 'WEIGHT_DECAY', 'USE_AMP', 'DATASETS', 'MAWI_URL',
                                     'BUDGET_GRID', 'CASCADE_TOL', 'ROUTER_EPOCHS', 'NONINF_MARGIN']}
CONFIG_SHA = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:12]
RUN = ROOT / f'run_{CONFIG_SHA}'

_shared_cache = ROOT.parent / 'NCC2027_MultiMAWI' / 'trace_cache'
DATA_CACHE = _shared_cache if _shared_cache.is_dir() and any(_shared_cache.glob('*.npz')) else ROOT / 'trace_cache'
CKPT, PREDS, HIST, TABLES, FIGS = (RUN / s for s in ('ckpt', 'preds', 'hist', 'tables', 'figs'))
for p in (DATA_CACHE, CKPT, PREDS, HIST, TABLES, FIGS):
    p.mkdir(parents=True, exist_ok=True)
(RUN / 'config.json').write_text(json.dumps(CONFIG, indent=2))
CACHE = RUN / 'cache'
CACHE.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 8, 'axes.grid': True, 'grid.alpha': .3,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'legend.fontsize': 6})

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
