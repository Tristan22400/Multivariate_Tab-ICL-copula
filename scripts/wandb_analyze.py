"""
wandb_analyze.py — W&B run analysis for multivariate-tab-icl project.

Filters runs by dataset, model type, and time window, then downloads metrics
and produces CSV summaries, plots, and a report focused on detecting
constant-covariance failure (no xi-conditioning).

Usage:
    conda run -n multivariate-icl python scripts/wandb_analyze.py \
        --project multivariate-tab-icl \
        --dataset pit_hyperplane_multimodal \
        --model copula_tabicl_v2 \
        --since 2026-05-20 --until 2026-06-05 \
        --out ./wandb_analysis/
"""

import argparse
import json
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Metric keys to pull from history
# ---------------------------------------------------------------------------

TRAIN_KEYS = [
    "train/copula_nll",
    "train/copula_gain",
    "train/oracle_nll_y",
    "train/mse",
    "train/alpha",
    "train/nll_weight",
    "train/lr",
    "train/pred_off_diag_var",
    "train/oracle_off_diag_var",
    "train/div",
]

VAL_KEYS = [
    "val/copula_nll",
    "val/joint_y_nll",
    "val/oracle_nll",
    "val/train_nll",
    "val/oracle_nll_z",
    "val/oracle_frac",
    "val/copula_gain",
    "val/hetero_gain",
    "val/energy_score",
    "val/oas_nll",
    "val/knn5_cov_nll",
    "val/linear_factor_nll",
]

ATTN_KEYS = [
    "attn/val_mean_abs_offdiag",
    "attn/val_std_offdiag_xi",
    "attn/val_var_frob_from_I",
]

ALL_HISTORY_KEYS = TRAIN_KEYS + VAL_KEYS + ATTN_KEYS

# Config fields to extract and store
CONFIG_WHITELIST = [
    "seed",
    "training.lr",
    "training.steps",
    "training.warmup_steps",
    "training.nll_weight",
    "training.aux_mse_weight",
    "training.aux_mse_anneal_frac",
    "training.dataset_dir",
    "training.n_think",
    "model.d_model",
    "model.d_hidden",
    "model.n_heads",
    "model.n_layers",
    "model.n_layers_s1",
    "model.n_layers_s2",
    "model.n_layers_s3",
    "model.rank",
    "model.n_inducing",
    "model.n_cls",
    "data.p_range",
    "data.d_range",
    "data.n_train_range",
    "data.n_test_range",
    "data.r_data",
    "data.cov_type",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="W&B run analysis for copula-intra")
    p.add_argument("--entity", default=None, help="W&B entity (user or team)")
    p.add_argument("--project", default="multivariate-tab-icl")
    p.add_argument("--dataset", action="append", default=[], metavar="DATASET",
                   help="Dataset name substring (repeatable); matches dataset_dir basename or run name")
    p.add_argument("--model", action="append", default=[], metavar="MODEL",
                   help="Model name substring (repeatable); matches config or run name prefix")
    p.add_argument("--since", default=None, help="ISO date lower bound on created_at (e.g. 2026-05-20)")
    p.add_argument("--until", default=None, help="ISO date upper bound on created_at (e.g. 2026-06-05)")
    p.add_argument("--state", default="finished,running",
                   help="Comma-separated run states (finished, running, crashed, failed)")
    p.add_argument("--tag", action="append", default=[], metavar="TAG",
                   help="W&B tag filter (repeatable, AND logic)")
    p.add_argument("--limit", type=int, default=50, help="Max runs to fetch")
    p.add_argument("--out", default="./wandb_analysis", help="Output directory")
    p.add_argument("--history-samples", type=int, default=2000,
                   help="Downsample points per run history")
    p.add_argument("--no-cache", action="store_true", help="Bypass local history cache")
    p.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    return p.parse_args()


def flat_config(cfg: dict, prefix="") -> dict:
    """Recursively flatten a nested config dict with dot-separated keys."""
    out = {}
    for k, v in cfg.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flat_config(v, full))
        else:
            out[full] = v
    return out


def extract_config_fields(run) -> dict:
    cfg = flat_config(run.config) if isinstance(run.config, dict) else {}
    result = {}
    for field in CONFIG_WHITELIST:
        result[field] = cfg.get(field)
    # Infer model name from run name or config
    name = run.name or ""
    model_name = cfg.get("model._target_", cfg.get("model.name", ""))
    if not model_name:
        # Try to extract from run name pattern: <model_name>_lr=...
        m = re.match(r"^([a-zA-Z0-9_]+)_lr=", name)
        model_name = m.group(1) if m else ""
    result["model_name"] = model_name
    # Dataset dir basename
    dataset_dir = result.get("training.dataset_dir") or ""
    result["dataset"] = Path(dataset_dir).name if dataset_dir else _extract_dataset_from_name(name)
    return result


def _extract_dataset_from_name(run_name: str) -> str:
    m = re.search(r"data=([^_\s]+)", run_name)
    return m.group(1) if m else ""


def get_final_tail(df: pd.DataFrame, key: str, frac: float = 0.1) -> float:
    """Mean of the last `frac` of non-NaN values for `key` in df."""
    if key not in df.columns:
        return float("nan")
    s = df[key].dropna()
    if len(s) == 0:
        return float("nan")
    n = max(1, int(len(s) * frac))
    return float(s.iloc[-n:].mean())


def detect_constant_cov(df: pd.DataFrame, summary: dict) -> tuple[str, list[str]]:
    """
    Returns (severity, reasons) where severity ∈ {OK, SUSPECT, COLLAPSED}.
    Uses last-10%-of-training tail averages where available.
    """
    reasons = []

    std_xi = get_final_tail(df, "attn/val_std_offdiag_xi")
    pred_var = get_final_tail(df, "train/pred_off_diag_var")
    oracle_var = get_final_tail(df, "train/oracle_off_diag_var")
    oracle_frac = get_final_tail(df, "val/oracle_frac")
    copula_gain = get_final_tail(df, "val/copula_gain")
    copula_nll = get_final_tail(df, "val/copula_nll")
    oas_nll = get_final_tail(df, "val/oas_nll")

    if not np.isnan(std_xi) and std_xi < 1e-3:
        reasons.append(f"attn/val_std_offdiag_xi={std_xi:.2e} < 1e-3 (xi-invariant predictions)")

    if not np.isnan(pred_var) and not np.isnan(oracle_var) and oracle_var > 0:
        ratio = pred_var / oracle_var
        if ratio < 1e-4:
            reasons.append(f"pred_off_diag_var/oracle_off_diag_var={ratio:.2e} < 1e-4 (collapsed scale)")

    if not np.isnan(oracle_frac) and oracle_frac > 0.9 and not np.isnan(copula_gain) and copula_gain < 0.01:
        reasons.append(f"oracle_frac={oracle_frac:.3f} > 0.9 and copula_gain={copula_gain:.4f} ≈ 0 (≈ N(0,I) prior)")

    if (not np.isnan(copula_nll) and not np.isnan(oas_nll) and copula_nll > oas_nll
            and not np.isnan(std_xi) and std_xi < 1e-2):
        reasons.append(f"copula_nll={copula_nll:.4f} > oas_nll={oas_nll:.4f} while xi-flat (beaten by shrinkage)")

    if len(reasons) == 0:
        return "OK", []
    elif len(reasons) == 1:
        return "SUSPECT", reasons
    else:
        return "COLLAPSED", reasons


def build_wandb_filter(args) -> dict:
    """Build MongoDB-style W&B filter dict."""
    and_clauses = []

    states = [s.strip() for s in args.state.split(",")]
    and_clauses.append({"state": {"$in": states}})

    if args.since:
        and_clauses.append({"created_at": {"$gte": args.since}})
    if args.until:
        # Add one day so --until is inclusive
        until_dt = datetime.fromisoformat(args.until)
        and_clauses.append({"created_at": {"$lte": until_dt.strftime("%Y-%m-%d")}})

    for tag in args.tag:
        and_clauses.append({"tags": tag})

    # Dataset / model filters via run-name regex (W&B supports $regex on display_name)
    datasets = []
    for d in args.dataset:
        for part in d.split(","):
            part = part.strip()
            if part:
                datasets.append(part)
    if datasets:
        regex = "|".join(re.escape(d) for d in datasets)
        and_clauses.append({"display_name": {"$regex": regex}})

    models = []
    for m in args.model:
        for part in m.split(","):
            part = part.strip()
            if part:
                models.append(part)
    if models:
        regex = "|".join(re.escape(m) for m in models)
        and_clauses.append({"display_name": {"$regex": regex}})

    return {"$and": and_clauses} if and_clauses else {}


def load_or_fetch_history(run, out_dir: Path, samples: int, no_cache: bool) -> pd.DataFrame:
    cache_path = out_dir / ".cache" / f"{run.id}.parquet"
    if not no_cache and cache_path.exists():
        print(f"  [cache] Loading history from {cache_path}")
        return pd.read_parquet(cache_path)

    available = set(run.history(samples=1, pandas=True).columns)
    keys_to_fetch = [k for k in ALL_HISTORY_KEYS if k in available or True]  # try all, W&B skips missing

    try:
        df = run.history(keys=keys_to_fetch + ["_step"], samples=samples, pandas=True)
    except Exception as e:
        print(f"  [warn] history fetch failed: {e}")
        df = pd.DataFrame()

    if not df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
    return df


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns to history dataframe."""
    if "val/joint_y_nll" in df.columns and "val/oracle_nll" in df.columns:
        df["derived/joint_gap"] = df["val/joint_y_nll"] - df["val/oracle_nll"]
    if "val/copula_nll" in df.columns:
        if "val/oas_nll" in df.columns:
            df["derived/vs_oas"] = df["val/copula_nll"] - df["val/oas_nll"]
        if "val/knn5_cov_nll" in df.columns:
            df["derived/vs_knn5"] = df["val/copula_nll"] - df["val/knn5_cov_nll"]
        if "val/linear_factor_nll" in df.columns:
            df["derived/vs_linear"] = df["val/copula_nll"] - df["val/linear_factor_nll"]
    if "val/copula_nll" in df.columns and "val/train_nll" in df.columns:
        df["derived/overfit_gap"] = df["val/copula_nll"] - df["val/train_nll"]
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(run_id: str, name: str, df: pd.DataFrame, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warn] matplotlib not available, skipping plots")
        return

    plot_dir = out_dir / "plots" / run_id
    plot_dir.mkdir(parents=True, exist_ok=True)

    step_col = "_step" if "_step" in df.columns else df.index

    # --- NLL curves ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"NLL Curves — {name}", fontsize=9)

    ax = axes[0]
    ax.set_title("Training NLL")
    for key, label in [
        ("train/copula_nll", "train copula NLL"),
        ("train/oracle_nll_y", "oracle NLL (Y)"),
    ]:
        if key in df.columns:
            ax.plot(df[step_col], df[key], label=label, alpha=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("NLL")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.set_title("Val NLL — Y-space comparison (model vs oracle)")
    for key, label, ls in [
        ("val/joint_y_nll", "val joint_y_nll (MODEL)", "-"),
        ("val/oracle_nll", "val oracle_nll (ORACLE)", "--"),
        ("val/copula_nll", "val copula_nll (Z-space)", ":"),
        ("val/oracle_nll_z", "val oracle_nll_z (Z-space oracle)", "-."),
        ("val/oas_nll", "OAS baseline", ":"),
        ("val/knn5_cov_nll", "kNN-5 baseline", ":"),
    ]:
        if key in df.columns:
            ax.plot(df[step_col], df[key], label=label, linestyle=ls, alpha=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("NLL")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_dir / "nll_curves.png", dpi=120)
    plt.close(fig)

    # --- xi-conditioning ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f"xi-Conditioning Diagnostics — {name}", fontsize=9)

    ax = axes[0]
    ax.set_title("attn/val_std_offdiag_xi\n(0 → constant cov, no xi-conditioning)")
    if "attn/val_std_offdiag_xi" in df.columns:
        ax.plot(df[step_col], df["attn/val_std_offdiag_xi"], color="red")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.set_title("Off-diag variance: pred vs oracle")
    for key, label, color in [
        ("train/pred_off_diag_var", "pred", "blue"),
        ("train/oracle_off_diag_var", "oracle", "green"),
    ]:
        if key in df.columns:
            ax.plot(df[step_col], df[key], label=label, color=color, alpha=0.8)
    ax.set_xlabel("step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.set_title("val/oracle_frac\n(1 → model ≈ N(0,I) prior)")
    if "val/oracle_frac" in df.columns:
        ax.plot(df[step_col], df["val/oracle_frac"], color="orange")
        ax.axhline(0.9, color="red", linestyle="--", alpha=0.5, label="threshold=0.9")
        ax.legend(fontsize=8)
    ax.set_xlabel("step")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_dir / "xi_conditioning.png", dpi=120)
    plt.close(fig)

    # --- gains ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Gains — {name}", fontsize=9)

    ax = axes[0]
    ax.set_title("copula_gain and hetero_gain")
    for key, label in [("val/copula_gain", "copula_gain"), ("val/hetero_gain", "hetero_gain")]:
        if key in df.columns:
            ax.plot(df[step_col], df[key], label=label)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.set_title("Model vs baselines (val/copula_nll − baseline)")
    for key, label in [
        ("derived/vs_oas", "vs OAS"),
        ("derived/vs_knn5", "vs kNN-5"),
        ("derived/vs_linear", "vs linear"),
    ]:
        if key in df.columns:
            ax.plot(df[step_col], df[key], label=label)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", label="baseline parity")
    ax.set_xlabel("step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_dir / "gains.png", dpi=120)
    plt.close(fig)


def make_compare_plot(dataset: str, run_histories: list[tuple[str, str, pd.DataFrame]], out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Cross-run comparison — dataset: {dataset}", fontsize=10)

    ax = axes[0]
    ax.set_title("val/copula_nll")
    ax2 = axes[1]
    ax2.set_title("attn/val_std_offdiag_xi (0 → no xi-conditioning)")

    for run_id, name, df in run_histories:
        step_col = "_step" if "_step" in df.columns else df.index
        short = name[:40]
        if "val/copula_nll" in df.columns:
            ax.plot(df[step_col], df["val/copula_nll"], label=short, alpha=0.7)
        if "attn/val_std_offdiag_xi" in df.columns:
            ax2.plot(df[step_col], df["attn/val_std_offdiag_xi"], label=short, alpha=0.7)

    for a in (ax, ax2):
        a.set_xlabel("step")
        a.legend(fontsize=6)
        a.grid(True, alpha=0.3)

    fig.tight_layout()
    safe_ds = re.sub(r"[^a-zA-Z0-9_-]", "_", dataset)
    fig.savefig(out_dir / "plots" / f"_compare_{safe_ds}.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(runs_df: pd.DataFrame, out_dir: Path, args):
    n_total = len(runs_df)
    n_collapsed = (runs_df["cov_verdict"] == "COLLAPSED").sum() if "cov_verdict" in runs_df.columns else 0
    n_suspect = (runs_df["cov_verdict"] == "SUSPECT").sum() if "cov_verdict" in runs_df.columns else 0

    lines = [
        "# W&B Run Analysis Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Project: `{args.project}`",
        f"Filters — dataset: `{args.dataset}`, model: `{args.model}`, "
        f"since: `{args.since}`, until: `{args.until}`, state: `{args.state}`",
        "",
        f"## Summary",
        f"- Runs matched: **{n_total}**",
        f"- COLLAPSED (constant cov failure): **{n_collapsed}**",
        f"- SUSPECT: **{n_suspect}**",
        f"- OK: **{n_total - n_collapsed - n_suspect}**",
        "",
    ]

    # Primary comparability: joint_y_nll vs oracle_nll
    lines += [
        "## Primary Metric Pair (Y-space, comparable)",
        "",
        "| run_id | name | joint_y_nll (model) | oracle_nll (oracle) | joint_gap | verdict |",
        "|--------|------|---------------------|---------------------|-----------|---------|",
    ]
    cols_needed = {"run_id", "name", "cov_verdict"}
    for _, row in runs_df.iterrows():
        jynll = row.get("final/val/joint_y_nll", float("nan"))
        onll = row.get("final/val/oracle_nll", float("nan"))
        gap = row.get("final/derived/joint_gap", float("nan"))
        lines.append(
            f"| {row.get('run_id','')} | {str(row.get('name',''))[:40]} | "
            f"{_fmt(jynll)} | {_fmt(onll)} | {_fmt(gap)} | {row.get('cov_verdict','?')} |"
        )
    lines.append("")

    # Top 5 by copula_gain
    if "final/val/copula_gain" in runs_df.columns:
        top5 = runs_df.nlargest(5, "final/val/copula_gain")
        lines += [
            "## Top 5 Runs by val/copula_gain",
            "",
            "| run_id | name | copula_gain | oracle_frac | std_offdiag_xi | verdict |",
            "|--------|------|-------------|-------------|----------------|---------|",
        ]
        for _, row in top5.iterrows():
            lines.append(
                f"| {row.get('run_id','')} | {str(row.get('name',''))[:40]} | "
                f"{_fmt(row.get('final/val/copula_gain'))} | "
                f"{_fmt(row.get('final/val/oracle_frac'))} | "
                f"{_fmt(row.get('final/attn/val_std_offdiag_xi'))} | "
                f"{row.get('cov_verdict','?')} |"
            )
        lines.append("")

    # Worst offenders
    bad = runs_df[runs_df.get("cov_verdict", pd.Series(dtype=str)).isin(["COLLAPSED", "SUSPECT"])]
    if not bad.empty:
        lines += ["## Flagged Runs (COLLAPSED / SUSPECT)", ""]
        for _, row in bad.iterrows():
            lines.append(f"### {row.get('run_id','')} — {row.get('name','')}")
            lines.append(f"- Verdict: **{row.get('cov_verdict','?')}**")
            reasons_raw = row.get("cov_reasons", "")
            if reasons_raw:
                for r in str(reasons_raw).split("|"):
                    if r.strip():
                        lines.append(f"  - {r.strip()}")
            lines.append("")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"\nReport written to {report_path}")


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history").mkdir(exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    try:
        import wandb
    except ImportError:
        print("ERROR: wandb not installed. Run: pip install wandb", file=sys.stderr)
        sys.exit(1)

    api = wandb.Api(timeout=60)
    path = f"{args.entity}/{args.project}" if args.entity else args.project

    print(f"Querying W&B: {path}")
    filters = build_wandb_filter(args)
    if filters:
        print(f"Filters: {json.dumps(filters, indent=2)}")

    try:
        runs = api.runs(path=path, filters=filters or None, per_page=args.limit)
        runs = list(runs)[:args.limit]
    except Exception as e:
        print(f"ERROR fetching runs: {e}", file=sys.stderr)
        sys.exit(1)

    if not runs:
        print("No runs matched the filters.")
        return

    print(f"\nMatched {len(runs)} run(s):\n")
    header = f"{'ID':12s}  {'Name':50s}  {'Model':25s}  {'Dataset':30s}  {'State':10s}  {'Created':12s}"
    print(header)
    print("-" * len(header))
    for r in runs:
        cfg = extract_config_fields(r)
        created = (r.created_at or "")[:10]
        print(f"{r.id:12s}  {(r.name or '')[:50]:50s}  {cfg['model_name'][:25]:25s}  "
              f"{cfg['dataset'][:30]:30s}  {r.state:10s}  {created:12s}")
    print()

    all_run_rows = []
    run_histories: dict[str, list] = {}  # dataset → list of (id, name, df)

    for run in runs:
        print(f"Processing {run.id} — {run.name} ...")
        try:
            cfg = extract_config_fields(run)
            dataset = cfg["dataset"]

            # History
            df = load_or_fetch_history(run, out_dir, args.history_samples, args.no_cache)
            if df.empty:
                print(f"  [warn] No history rows for {run.id}")

            df = compute_derived(df)

            # Save per-run history CSV
            hist_path = out_dir / "history" / f"{run.id}.csv"
            df.to_csv(hist_path, index=False)

            # Detect constant-cov failure
            verdict, reasons = detect_constant_cov(df, run.summary._json_dict if hasattr(run.summary, "_json_dict") else {})

            # Build summary row
            row = {"run_id": run.id, "name": run.name, "state": run.state,
                   "created_at": (run.created_at or "")[:19],
                   "runtime_s": run.summary.get("_runtime"),
                   "last_step": run.summary.get("_step"),
                   "cov_verdict": verdict,
                   "cov_reasons": " | ".join(reasons)}
            row.update({f"cfg/{k}": v for k, v in cfg.items()})

            # Final tail values of key metrics
            tail_keys = ALL_HISTORY_KEYS + [
                "derived/joint_gap", "derived/vs_oas", "derived/vs_knn5",
                "derived/vs_linear", "derived/overfit_gap"
            ]
            for k in tail_keys:
                row[f"final/{k}"] = get_final_tail(df, k)

            # Missing keys
            missing = [k for k in ALL_HISTORY_KEYS if k not in df.columns or df[k].isna().all()]
            row["missing_keys"] = ",".join(missing)

            all_run_rows.append(row)

            # Register for compare plot
            run_histories.setdefault(dataset, []).append((run.id, run.name or run.id, df))

            # Per-run plots
            if not args.no_plots:
                make_plots(run.id, run.name or run.id, df, out_dir)

        except Exception:
            print(f"  [ERROR] Failed processing {run.id}:")
            traceback.print_exc()

    if not all_run_rows:
        print("No runs processed successfully.")
        return

    runs_df = pd.DataFrame(all_run_rows)
    runs_csv = out_dir / "runs.csv"
    runs_df.to_csv(runs_csv, index=False)
    print(f"\nSummary written to {runs_csv}")

    # Cross-run compare plots per dataset
    if not args.no_plots:
        for dataset, histories in run_histories.items():
            if len(histories) >= 2:
                make_compare_plot(dataset, histories, out_dir)

    write_report(runs_df, out_dir, args)

    # Print verdict table to stdout
    print("\n=== Constant-Covariance Verdict ===")
    print(f"{'run_id':12s}  {'verdict':10s}  {'std_xi':10s}  {'oracle_frac':12s}  {'joint_gap':10s}  {'name'}")
    print("-" * 90)
    for _, row in runs_df.iterrows():
        print(
            f"{row.get('run_id',''):12s}  "
            f"{row.get('cov_verdict','?'):10s}  "
            f"{_fmt(row.get('final/attn/val_std_offdiag_xi')):10s}  "
            f"{_fmt(row.get('final/val/oracle_frac')):12s}  "
            f"{_fmt(row.get('final/derived/joint_gap')):10s}  "
            f"{str(row.get('name',''))[:45]}"
        )


if __name__ == "__main__":
    main()
