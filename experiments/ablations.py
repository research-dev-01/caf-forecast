"""
Ablations and secondary studies (architectural components & data-scale study).
Corresponds to Cell 18 and Cell 19 of cmsnts_latest.ipynb.
"""

import numpy as np
import pandas as pd
from config import (
    SEEDS, RUN_ABLATIONS, ABLATION_SPECS, RUN, PROPOSED, MODEL_SPECS
)
from utils.helpers import stamp, checkpoint, remaining_min
from experiments.engine import guarded_run, run_deep

EXTRA_SEEDS = SEEDS[:3]
stamp('Ablations ...')
if RUN_ABLATIONS:
    for seed in SEEDS:
        for ds, D in DATA.items():
            for name, spec in ABLATION_SPECS.items():
                if guarded_run('ablation', D, name, spec, seed, reserve_min=ANALYSIS_RESERVE_MIN + 15) is None:
                    SKIPPED.append(('ablation', ds, name, seed)); continue
                record('ablation', D, name, seed, DEEP_PREDS[('ablation', ds, name, seed)]['test'])
    # does the cascade still work without deep supervision? (router + budget sweep on the ablated backbone)
    _nds = 'CAF w/o deep supervision'
    for seed in SEEDS:
        for ds, D in DATA.items():
            if ('ablation', ds, _nds, seed) in DEEP_PREDS:
                cascade_eval('ablation', D, _nds, ABLATION_SPECS[_nds], seed, register=False)

RES_DATA = {}
RES_MODELS = ['DLinear', 'PatchTST', 'iTransformer', 'TimeXer', PROPOSED]
if RUN_RESOLUTION_STUDY and remaining_min() > ANALYSIS_RESERVE_MIN + 20:
    stamp('Resolution study (50 ms bins) ...')
    for ds in DATASETS:
        RES_DATA[ds] = build_task(ds, bin_ms=5 * BIN_MS)
        run_classical('res50', RES_DATA[ds], verbose=False)
    for seed in EXTRA_SEEDS:
        for ds, D in RES_DATA.items():
            for name in RES_MODELS:
                if guarded_run('res50', D, name, MODEL_SPECS[name], seed, reserve_min=ANALYSIS_RESERVE_MIN + 10) is None:
                    SKIPPED.append(('res50', ds, name, seed)); continue
                record('res50', D, name, seed, DEEP_PREDS[('res50', ds, name, seed)]['test'])

LB_DATA = {}
LB_MODELS = ['DLinear', 'PatchTST', 'iTransformer', PROPOSED]
LB_VALUES = [16, 32, 128]
if RUN_LOOKBACK_STUDY and remaining_min() > ANALYSIS_RESERVE_MIN + 15:
    stamp('Look-back study ...')
    for L in LB_VALUES:
        for ds in DATASETS:
            LB_DATA[(ds, L)] = build_task(ds, L=L)
    for seed in EXTRA_SEEDS:
        for (ds, L), D in LB_DATA.items():
            for name in LB_MODELS:
                if guarded_run('lookback', D, name, MODEL_SPECS[name], seed, reserve_min=ANALYSIS_RESERVE_MIN + 5) is None:
                    SKIPPED.append(('lookback', f'{ds}|L{L}', name, seed)); continue
                RESULTS.extend([dict(r, study='lookback', dataset=ds, model=name, seed=seed, L=L)
                                for r in per_day_metrics(D, DEEP_PREDS[('lookback', f'{ds}|L{L}', name, seed)]['test'])])
pd.DataFrame(SKIPPED, columns=['study', 'dataset', 'model', 'seed']).to_csv(RUN / 'skipped_runs.csv', index=False)
stamp(f'Training finished; {len(SKIPPED)} runs skipped by the time guard; remaining {remaining_min():.0f} min')
checkpoint('all training', force=True)

#       only; same validation/test days). CAF (and its 0.8k-parameter exit E1) vs the strongest SOTA families, 3 seeds.
RUN_LOWDATA = True
LOWDATA_MODELS = [m for m in [PROPOSED, 'TimeXer', 'TimeMixer', 'TSMixer', 'TimesNet'] if m in MODEL_SPECS]
LOWDATA_SEEDS = SEEDS[:3]
DATA_LOW = {}

def build_lowdata(ds):
    tmp = f'{ds} (1 train day)'
    DATASETS[tmp] = dict(DATASETS[ds], train=DATASETS[ds]['train'][-1:])
    try:
        return TaskData(tmp, BIN_MS, LOOKBACK, HORIZON)
    finally:
        del DATASETS[tmp]                                          # keep DS / tables unchanged

if RUN_LOWDATA:
    stamp(f'Low-data study: {LOWDATA_MODELS} × {len(DATA)} datasets × {len(LOWDATA_SEEDS)} seeds (1 training day)')
    for ds in DATA:
        DATA_LOW[ds] = build_lowdata(ds)
    for name in LOWDATA_MODELS:                                      # model-major: a model is complete or absent
        for ds, D in DATA_LOW.items():
            for seed in LOWDATA_SEEDS:
                if guarded_run('lowdata', D, name, MODEL_SPECS[name], seed, reserve_min=ANALYSIS_RESERVE_MIN + 40) is None:
                    SKIPPED.append(('lowdata', ds, name, seed))
        checkpoint(f'low-data {name}', force=True)
    stamp(f'Low-data study done; remaining {remaining_min():.0f} min')
