# Compute-Adaptive Cascaded Forecasting for Line-Rate Backbone Traffic Monitoring (CAF)

Official research implementation and benchmarking suite for **CAF** (Compute-Adaptive Cascaded Forecasting) and the **CAF-X** serving layer, targeted for high-speed network telemetry monitoring (IEEE COMSNETS 2027 · Systems for the Future of AI/ML).

---

## Technical Overview

High-speed backbone telemetry operates under strict real-time deadlines: a closed-loop controller must emit wire bit-rate forecasts over sub-second horizons (10–100 ms) at a 10 ms cadence (100 Hz). While deep neural time-series models (e.g., PatchTST, iTransformer, TimesNet) achieve high benchmark accuracy, their static multi-megaFLOP evaluation on every window incurs excessive latency and power overhead on edge routers and monitoring appliances.

**CAF** addresses this constraint via:
1. **Multi-Scale Moving Average Anchors ($E_0$):** Zero-compute baseline anchors across sliding timescales $s \in \{2, 4, 8, 16, 32\}$.
2. **Scale-Equivariant Linear Residual Exit ($E_1$):** Evaluates incoming windows using a lightweight $1.6\,\text{kFLOP}$ linear projection conditioned on running scale estimates ($S_t$) and 10 packet-header features.
3. **Deep Residual Backbone ($E_2$):** A gated dilated convolutional network with deep supervision across exits.
4. **Adaptive Escalation Router / CAF-X:** A calibrated classifier trained on historical prediction errors that routes stationary windows to $E_1$ and selectively escalates difficult burst transients to deep backbones ($E_2$ or external foundation models).

---

## Project Structure

```
caf_project/
├── README.md                      # Repository documentation and reproduction instructions
├── requirements.txt               # Pinned Python package dependencies
├── .gitignore                     # Git ignore rules for caches, checkpoints, and binaries
├── config.py                      # Global experiment configurations, dataset splits, and constants
├── main.py                        # Central CLI execution and verification entry point
├── data/
│   ├── __init__.py
│   ├── pcap_parser.py             # Numba JIT streaming PCAP parser and GzipHTTPStream
│   ├── acquisition.py             # Parallel HTTP range-request downloader for WIDE/MAWI traces
│   ├── features.py                # 10-channel per-bin extraction, Hurst exponent, Table I
│   └── dataset.py                 # Chronological windowed splits (L=64, H=10) & GPU batching
├── models/
│   ├── __init__.py
│   ├── classical.py               # Statistical baselines: Persistence, MA, EWMA, Ridge-AR, HistGB
│   ├── baselines.py               # SOTA deep baseline zoo (10 models: PatchTST, iTransformer, etc.)
│   └── caf.py                     # Proposed CAF architecture, nested exits, and parameter counts
├── experiments/
│   ├── __init__.py
│   ├── metrics.py                 # Evaluation metrics: MAE, RMSE, MASE, Skill Score
│   ├── engine.py                  # PyTorch training engine, AdamW optimizer, early stopping
│   ├── cascade.py                 # Router fitting, budget threshold calibration, cascade sweeps
│   ├── ablations.py               # Architectural ablations (Table V) and data-scale study
│   ├── profiling.py               # FLOP counting, CPU/GPU latency profiling, edge benchmarks
│   └── cafx.py                    # CAF-X model-agnostic serving and fair ensemble comparisons
├── utils/
│   ├── __init__.py
│   ├── helpers.py                 # Atomic checkpoint shards, caching, timing, savefig
│   ├── statistics.py              # Diebold-Mariano (HAC), Holm-Bonferroni, Wilcoxon, CD diagram
│   └── visualization.py           # Publication figures (Figs. 1-3, S1-S10, diagnostic panels)
└── analysis/
    ├── __init__.py
    ├── results_tables.py          # Results aggregation and LaTeX tables (Tables II-VII)
    └── claims_check.py            # Automated checklist verification of empirical claims
```

---

## Requirements and Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.11, and 3.12)
- NVIDIA GPU with CUDA support recommended (e.g., Tesla T4, RTX 3090/4090) for full training grid. CPU-only execution is supported for verification and inference.

### Installation
```bash
git clone https://github.com/research-dev-01/caf-forecast.git
cd caf-forecast
pip install -r requirements.txt
```

---

## Dataset Requirements

Experiments evaluate 21 traces across three historical operational eras from the WIDE Project trans-Pacific backbone archive (MAWI Working Group, Samplepoint-F):
- **MAWI-2019 (Pre-pandemic):** 7 daily 5-minute captures (March–April 2019)
- **MAWI-2022 (Post-pandemic):** 7 daily 5-minute captures (March–April 2022)
- **MAWI-2025 (Current 10G/BBIX IX link):** 7 daily 5-minute captures (September–October 2025)

Raw PCAP streams are fetched directly from `https://mawi.wide.ad.jp/` via parallel HTTP range requests and decompressed on the fly into 10 ms aggregated bins using Numba JIT parsing. Pre-parsed trace caches (`.npz`) are stored in `trace_cache/`.

---

## Usage and Execution

### 1. Structural Verification (Fast)
Verify that all 11 model architectures, parameter counts, tensor shapes, and scale-equivariance checks pass:
```bash
python main.py --stage verify --allow-cpu
```

### 2. Trace Acquisition & Binning
Download and parse all 21 MAWI traces (requires network connection and ~15 GB temporary bandwidth):
```bash
python main.py --stage acquire
```

### 3. Classical Baselines
Fit validation-tuned statistical forecasters (Persistence, Moving Average, EWMA, Ridge-AR, HistGB):
```bash
python main.py --stage classical
```

### 4. Deep Model Training Grid
Train the 11 deep models across 3 datasets and 5 random seeds (165 total training runs; recommended on GPU):
```bash
python main.py --stage train
```

### 5. Cascade Router Calibration
Train the lightweight escalation router on training days and calibrate operating budgets on the validation set:
```bash
python main.py --stage cascade
```

### 6. Results, Tables, and Figures
Generate all LaTeX tables (Tables II–VII) and publication figures:
```bash
python main.py --stage tables
python main.py --stage figures
```

### 7. End-to-End Pipeline
Execute the full benchmark sequence:
```bash
python main.py --stage all
```

---

## Evaluation Metrics

- **MAE (Mbit/s):** Primary accuracy metric across prediction horizons ($H \in [1, 10]$, representing $10\,\text{ms}$ to $100\,\text{ms}$).
- **Skill Score:** Relative accuracy improvement against the persistence baseline:
  $$\text{Skill} = 1 - \frac{\text{MAE}}{\text{MAE}_{\text{Persistence}}}$$
- **Computational Cost:** FLOPs per window measured via `torch.utils.flop_counter.FlopCounterMode`.
- **Latency & Throughput:** Single-core CPU latency (median and 95th percentile) and multiplexing capacity (links served per CPU core under deadline $\le 10\,\text{ms}$).
