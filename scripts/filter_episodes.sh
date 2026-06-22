#!/bin/bash
#OAR -n TabICL_Filter
#OAR -l gpu=0,walltime=4:00:00

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

source ~/thoth_storage/miniconda3/etc/profile.d/conda.sh
conda activate multivariate-icl
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PYTHONPATH:$(pwd)

# ── Parameters (override via env or edit here) ──────────────────────────────────
INPUT_DIR="${INPUT_DIR:-data/pit_episodes}"
OUTPUT_DIR="${OUTPUT_DIR:-data/pit_episodes_filtered}"
MANIFEST="${MANIFEST:-data/filter_manifest.json}"
KEEP_FRACTION="${KEEP_FRACTION:-0.65}"
WORKERS="${WORKERS:-4}"
K="${K:-5}"
B_BOOTSTRAP="${B_BOOTSTRAP:-200}"
N_TREES="${N_TREES:-64}"
MIN_NBHD="${MIN_NBHD:-32}"
SEED="${SEED:-0}"
SPLIT="${SPLIT:-train}"

echo "======================================================="
echo " filter_episodes"
echo "   input  : $INPUT_DIR"
echo "   output : $OUTPUT_DIR"
echo "   manifest: $MANIFEST"
echo "   keep   : $KEEP_FRACTION"
echo "   workers: $WORKERS  K=$K  B=$B_BOOTSTRAP  trees=$N_TREES"
echo " (Job ID: ${OAR_JOB_ID:-local})"
echo "======================================================="

python - <<PYEOF
import sys, json, time, logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

sys.path.insert(0, "src")
from dataset_filter import FilterConfig, FilterResult, filter_episode_file, select_for_pretraining

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

INPUT_DIR      = Path("$INPUT_DIR")
OUTPUT_DIR     = Path("$OUTPUT_DIR")
MANIFEST       = Path("$MANIFEST")
KEEP_FRACTION  = float("$KEEP_FRACTION")
WORKERS        = int("$WORKERS")
SPLIT          = "$SPLIT"

config = FilterConfig(
    K=$K,
    B_bootstrap=$B_BOOTSTRAP,
    n_trees=$N_TREES,
    min_neighborhood_size=$MIN_NBHD,
    seed=$SEED,
)

episode_files = sorted(INPUT_DIR.glob("episode_*.pt"))
if not episode_files:
    log.error("No episode_*.pt files found in %s", INPUT_DIR)
    sys.exit(1)
log.info("Found %d episode files", len(episode_files))

# ── Phase 1: score every dataset ────────────────────────────────────────────────

def _filter_one(path):
    try:
        return str(path), filter_episode_file(str(path), config, split=SPLIT)
    except Exception as e:
        log.warning("Skipping %s: %s", path, e)
        return str(path), []

raw = {}
t0 = time.time()
if WORKERS > 1:
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(_filter_one, f): f for f in episode_files}
        done = 0
        for fut in as_completed(futs):
            path, results = fut.result()
            raw[path] = results
            done += 1
            if done % max(1, len(episode_files) // 10) == 0 or done == len(episode_files):
                log.info("  %d/%d  (%.0fs)", done, len(episode_files), time.time()-t0)
else:
    for i, f in enumerate(episode_files):
        path, results = _filter_one(f)
        raw[path] = results
        if (i+1) % max(1, len(episode_files) // 10) == 0 or i+1 == len(episode_files):
            log.info("  %d/%d  (%.0fs)", i+1, len(episode_files), time.time()-t0)

# ── Phase 2: global threshold calibration ───────────────────────────────────────
flat_results, flat_index = [], []
for path in sorted(raw):
    for b, r in enumerate(raw[path]):
        flat_results.append(r)
        flat_index.append((path, b))

if not flat_results:
    log.error("All files failed — nothing to calibrate.")
    sys.exit(1)

log.info("Calibrating global threshold over %d datasets …", len(flat_results))
global_keep = select_for_pretraining(
    flat_results,
    target_keep_fraction=KEEP_FRACTION,
    seed=$SEED,
)

# ── Phase 3: write filtered .pt files ───────────────────────────────────────────
import torch

per_file_keep = {}
for (path, b), keep in zip(flat_index, global_keep):
    per_file_keep.setdefault(path, []).append(keep)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
n_files_written = n_datasets_written = 0
for src in episode_files:
    mask = per_file_keep.get(str(src), [])
    keep_idx = [i for i, k in enumerate(mask) if k]
    if not keep_idx:
        continue
    ep = torch.load(str(src), map_location="cpu", weights_only=False)
    out = {}
    for key, val in ep.items():
        if isinstance(val, torch.Tensor) and val.ndim >= 1 and val.shape[0] == len(mask):
            out[key] = val[keep_idx]
        else:
            out[key] = val
    torch.save(out, str(OUTPUT_DIR / src.name))
    n_files_written += 1
    n_datasets_written += len(keep_idx)

# ── Phase 4: write manifest ──────────────────────────────────────────────────────
scores = np.array([r.score for r in flat_results])
kept   = np.array(global_keep)
n_pit  = sum(r.pit_calibration.get("pit_suspect", False) for r in flat_results)

manifest = {
    "summary": {
        "n_episode_files": len(episode_files),
        "n_datasets_total": len(flat_results),
        "n_datasets_kept": int(kept.sum()),
        "kept_fraction": round(float(kept.mean()), 4),
        "n_pit_suspect": int(n_pit),
        "score_mean": round(float(scores.mean()), 5),
        "score_std":  round(float(scores.std()),  5),
    },
    "config": {
        "K": $K, "B_bootstrap": $B_BOOTSTRAP,
        "n_trees": $N_TREES, "min_neighborhood_size": $MIN_NBHD,
        "keep_fraction": KEEP_FRACTION,
    },
    "episodes": {
        Path(p).name: {
            "keep":   per_file_keep.get(p, []),
            "scores": [round(r.score, 5) for r in raw[p]],
        }
        for p in sorted(raw)
    },
}
MANIFEST.parent.mkdir(parents=True, exist_ok=True)
with open(MANIFEST, "w") as f:
    json.dump(manifest, f, indent=2)

s = manifest["summary"]
print()
print("=======================================================")
print(f"  Episodes processed : {s['n_episode_files']}")
print(f"  Datasets total     : {s['n_datasets_total']}")
print(f"  Datasets kept      : {s['n_datasets_kept']}  ({100*s['kept_fraction']:.1f}%)")
print(f"  Score mean / std   : {s['score_mean']} / {s['score_std']}")
print(f"  PIT suspect        : {s['n_pit_suspect']}")
print(f"  Output dir         : $OUTPUT_DIR")
print(f"  Manifest           : $MANIFEST")
print("=======================================================")
PYEOF

echo "Done."
