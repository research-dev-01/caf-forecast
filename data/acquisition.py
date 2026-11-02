"""
Acquire all captures (parallel streams, cached, resumable).
Corresponds to Cell 5 of cmsnts_latest.ipynb.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import ALL_DATES, DOWNLOAD_WORKERS, RUN
from utils.helpers import stamp, checkpoint
from data.pcap_parser import acquire_trace

TRACE, TRACE_META = {}, {}

def acquire_with_fallback(d):
    try:
        return d, acquire_trace(d, mirror=False)
    except Exception as e:
        print(f"Primary mirror failed for {d} ({e}); trying fallback ...")
        return d, acquire_trace(d, mirror=True)

def acquire_all_captures():
    """Download / cache all PCAP traces across all dataset dates."""
    global TRACE, TRACE_META
    stamp(f'Acquiring {len(ALL_DATES)} captures with {DOWNLOAD_WORKERS} parallel streams ...')
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futs = {pool.submit(acquire_with_fallback, d): d for d in ALL_DATES}
        for fut in as_completed(futs):
            d, (series, meta) = fut.result()
            TRACE[d], TRACE_META[d] = series, meta
            stamp(f'  [{len(TRACE):2d}/{len(ALL_DATES)}] {d} ready ({meta.get("packets", 0):,} pkts, {series.shape[0]} bins)')
    (RUN / 'trace_manifest.json').write_text(json.dumps(TRACE_META, indent=2))
    stamp('All captures available.')
    checkpoint('traces', force=True)
    return TRACE, TRACE_META
