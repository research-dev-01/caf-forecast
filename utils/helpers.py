"""
Runtime, storage, provenance and time-budget helpers.
Corresponds to Cell 2 of cmsnts_latest.ipynb.
"""

import time
import re
import shutil
import pickle
import zipfile
import matplotlib.pyplot as plt
from pathlib import Path
from config import (
    MAWI_MIRROR, MAWI_URL, NOTEBOOK_START, TIME_BUDGET_HOURS, ROOT, CKPT_DIR,
    CHECKPOINT_EVERY_MIN, SHARDS_NEEDED, IMMUTABLE_PARTS, CACHE, FIGS
)

def mawi_url(date, mirror=False):
    return (MAWI_MIRROR if mirror else MAWI_URL).format(year=date[:4], date=date)


def _restore_checkpoints():
    """Re-hydrate ROOT from checkpoint shards / a previous Kaggle output found in obvious places (never overwrites)."""
    cands = []
    for base in (Path('/content'), Path.cwd(), CKPT_DIR, Path('/content/drive/MyDrive')):
        if base.is_dir():
            cands += list(base.glob('CMS2027_ckpt_*.zip'))
    if Path('/kaggle/input').is_dir():
        cands += list(Path('/kaggle/input').glob('**/CMS2027_ckpt_*.zip'))
        cands += [d for d in Path('/kaggle/input').glob('**/COMSNETS2027_SysAI_CAF') if d.is_dir()]
    n_files, used = 0, set()
    for src in sorted(set(cands), key=lambda p: p.name):
        try:
            if src.is_dir():
                if src.resolve() == ROOT.resolve():
                    continue
                for f in src.rglob('*'):
                    dst = ROOT / f.relative_to(src)
                    if f.is_file() and not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, dst); n_files += 1
                used.add(str(src)); continue
            if src.name in used:
                continue
            used.add(src.name)
            with zipfile.ZipFile(src) as z:
                for info in z.infolist():
                    dst = ROOT / info.filename
                    if not info.is_dir() and not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        with z.open(info) as fi, open(dst, 'wb') as fo:
                            shutil.copyfileobj(fi, fo, 1 << 20)
                        n_files += 1
            if SHARDS_NEEDED and src.parent.resolve() != CKPT_DIR.resolve():
                CKPT_DIR.mkdir(parents=True, exist_ok=True)
                if not (CKPT_DIR / src.name).exists():
                    shutil.copy2(src, CKPT_DIR / src.name)
        except Exception as e:
            print(f'  checkpoint source {src} skipped: {type(e).__name__}: {e}')
    if n_files:
        print(f'Resumed: restored {n_files} files from {len(used)} checkpoint source(s) into {ROOT}')

def elapsed_min():
    return (time.time() - NOTEBOOK_START) / 60

def remaining_min():
    return TIME_BUDGET_HOURS * 60 - elapsed_min()

def stamp(msg):
    print(f'[{elapsed_min():6.1f} min] {msg}', flush=True)

def slug(s):
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')

_CK = {'t': time.time(), 'done': None}

def _shard_names():
    names = set()
    for s in CKPT_DIR.glob('CMS2027_ckpt_*.zip'):
        try:
            with zipfile.ZipFile(s) as z:
                names.update(z.namelist())
        except Exception:
            pass
    return names

def checkpoint(tag='', force=False):
    if not SHARDS_NEEDED or (not force and time.time() - _CK['t'] < CHECKPOINT_EVERY_MIN * 60):
        return
    t0 = time.time()
    try:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        if _CK['done'] is None:
            _CK['done'] = _shard_names()
        new, small = [], []
        for p in ROOT.rglob('*'):
            if not p.is_file() or p.suffix in ('.zip', '.tmp', '.part'):
                continue
            rel = p.relative_to(ROOT).as_posix()
            if IMMUTABLE_PARTS & set(rel.split('/')[:-1]):
                if rel not in _CK['done']:
                    new.append((p, rel))
            else:
                small.append((p, rel))
        if new:
            shard = CKPT_DIR / f'CMS2027_ckpt_{time.strftime("%Y%m%d_%H%M%S")}_{len(new):05d}.zip'
            tmp = shard.with_suffix('.part')
            with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_STORED) as z:
                for p, rel in new:
                    z.write(p, rel)
            tmp.replace(shard); _CK['done'].update(rel for _, rel in new)
        out = CKPT_DIR / 'CMS2027_ckpt_zzzz_outputs.zip'; tmp = out.with_suffix('.part')
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
            for p, rel in small:
                z.write(p, rel)
        tmp.replace(out)
        _CK['t'] = time.time()
        size = sum(f.stat().st_size for f in CKPT_DIR.glob('*.zip')) / 1e6
        print(f'[checkpoint{" " + tag if tag else ""}] +{len(new)} artefacts, {len(small)} outputs -> {CKPT_DIR} '
              f'({size:.0f} MB, {time.time() - t0:.1f} s)', flush=True)
    except Exception as e:
        print(f'[checkpoint {tag}] failed (run continues): {type(e).__name__}: {e}')

def cache_obj(name, fn):
    f = CACHE / f'{slug(name)}.pkl'
    if f.exists():
        try:
            return pickle.loads(f.read_bytes())
        except Exception:
            pass
    obj = fn(); tmp = f.with_suffix('.tmp'); tmp.write_bytes(pickle.dumps(obj)); tmp.replace(f)
    return obj

def savefig(fig, name):
    fig.savefig(FIGS / f'{name}.pdf', bbox_inches='tight'); fig.savefig(FIGS / f'{name}.png', bbox_inches='tight')
    plt.show(); plt.close(fig)
