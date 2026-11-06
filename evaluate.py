#!/usr/bin/env python3
"""Aggregate saved assay metrics and optionally bootstrap four structural contrasts.

This script summarizes existing assay-level results; it does not train models,
generate predictions, or reconstruct assay metrics from individual variants.
For screening endpoints, ProteinGym_official names the aggregation hierarchy,
not an official ProteinGym screening metric.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


METADATA = ("DMS_id", "UniProt_ID", "coarse_selection_type")
CLUSTER = "cluster_representative"
MODELS = ("hierarchical_cover", "ridge_cover", "additive_context", "fixed_fusion")
CONTRASTS = (
    ("hierarchical_cover", "fixed_fusion"),
    ("hierarchical_cover", "additive_context"),
    ("hierarchical_cover", "ridge_cover"),
    ("additive_context", "fixed_fusion"),
)
AGGREGATIONS = ("DMS_macro", "UniProt_macro", "ProteinGym_official")


def aggregate(values: pd.DataFrame) -> dict[str, float]:
    """Average assays, proteins, or protein-by-selection-type groups equally."""
    dms = float(values["value"].mean())
    uniprot = float(values.groupby("UniProt_ID", sort=True)["value"].mean().mean())
    official = float(
        values.groupby(["UniProt_ID", "coarse_selection_type"], sort=True)["value"]
        .mean()
        .groupby("coarse_selection_type", sort=True)
        .mean()
        .mean()
    )
    return {"DMS_macro": dms, "UniProt_macro": uniprot, "ProteinGym_official": official}


def common_cluster_counts(
    values: pd.DataFrame, seed: int, repeats: int
) -> tuple[np.ndarray, dict[str, int]]:
    """Use one common set of resampled clusters for every contrast."""
    clusters = sorted(values[CLUSTER].astype(str).unique().tolist())
    codes = {cluster: index for index, cluster in enumerate(clusters)}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(clusters), size=(repeats, len(clusters)))
    counts = np.apply_along_axis(
        lambda row: np.bincount(row, minlength=len(clusters)), 1, draws
    ).astype(np.float64)
    return counts, codes


def bootstrap_aggregates(
    values: pd.DataFrame, counts: np.ndarray, cluster_codes: dict[str, int]
) -> dict[str, np.ndarray]:
    """Recompute the aggregation hierarchy using each cluster's multiplicity.

    A cluster selected twice contributes its proteins twice. Selection types
    retain equal final weight; assays are never treated as independent draws.
    """
    assay_cluster = values[CLUSTER].astype(str).map(cluster_codes).to_numpy(int)
    assay_values = values["value"].to_numpy(float)
    assay_weights = counts[:, assay_cluster]
    dms = (assay_weights @ assay_values) / assay_weights.sum(axis=1)

    proteins = (
        values.groupby([CLUSTER, "UniProt_ID"], sort=True)["value"].mean().reset_index()
    )
    protein_cluster = proteins[CLUSTER].astype(str).map(cluster_codes).to_numpy(int)
    protein_values = proteins["value"].to_numpy(float)
    protein_weights = counts[:, protein_cluster]
    uniprot = (protein_weights @ protein_values) / protein_weights.sum(axis=1)

    protein_selection = (
        values.groupby([CLUSTER, "UniProt_ID", "coarse_selection_type"], sort=True)["value"]
        .mean()
        .reset_index()
    )
    category_draws = []
    for category, group in protein_selection.groupby("coarse_selection_type", sort=True):
        codes = group[CLUSTER].astype(str).map(cluster_codes).to_numpy(int)
        weights = counts[:, codes]
        denominator = weights.sum(axis=1)
        if np.any(denominator <= 0):
            raise ValueError(
                f"A bootstrap draw contains no assays for selection type {category!r}. "
                "The specified hierarchy is undefined for that draw; no draws were dropped."
            )
        category_draws.append((weights @ group["value"].to_numpy(float)) / denominator)
    official = np.vstack(category_draws).mean(axis=0)
    if not all(np.isfinite(item).all() for item in (dms, uniprot, official)):
        raise ValueError("Non-finite cluster-bootstrap aggregate")
    return {"DMS_macro": dms, "UniProt_macro": uniprot, "ProteinGym_official": official}


def p_two_sided(draws: np.ndarray) -> float:
    """Finite-resampling, two-sided sign probability, including exact zero."""
    lower = (np.sum(draws <= 0.0) + 1) / (len(draws) + 1)
    upper = (np.sum(draws >= 0.0) + 1) / (len(draws) + 1)
    return float(min(1.0, 2.0 * min(lower, upper)))


def holm(values: list[float]) -> list[float]:
    count = len(values)
    order = np.argsort(values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def validate_assays(values: pd.DataFrame, metrics: list[str], bootstrap: bool) -> None:
    required = [*METADATA, *metrics, *([CLUSTER] if bootstrap else [])]
    missing = sorted(set(required) - set(values.columns))
    if missing:
        raise ValueError(f"Missing input columns: {', '.join(missing)}")
    if values.empty or values[required].isna().any().any():
        raise ValueError("Input is empty or contains missing metadata/metric values")
    if not values["DMS_id"].is_unique:
        raise ValueError("Each assay must appear once per group; use --group-by for long tables")
    for column in [*METADATA, *([CLUSTER] if bootstrap else [])]:
        if values[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Empty identifier in {column}")
    if not np.isfinite(values[metrics].to_numpy(dtype=float)).all():
        raise ValueError("Metrics must be finite numbers; missing values are not silently excluded")
    if bootstrap and values.groupby("UniProt_ID")[CLUSTER].nunique().max() != 1:
        raise ValueError("Every UniProt target must belong to exactly one sequence cluster")


def core_contrasts(
    assays: pd.DataFrame, seed: int = 20260808, repeats: int = 10000
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Four paired contrasts, with Holm correction separately per aggregation."""
    if repeats <= 0 or seed < 0:
        raise ValueError("repeats must be positive and seed must be non-negative")
    validate_assays(assays, list(MODELS), bootstrap=True)
    assays = assays.sort_values("DMS_id").reset_index(drop=True)
    counts, cluster_codes = common_cluster_counts(assays, seed, repeats)
    saved = {
        "cluster_ids": np.asarray(list(cluster_codes)),
        "cluster_multiplicities": counts.astype(np.int32),
    }
    rows = []
    for high, low in CONTRASTS:
        values = assays[[*METADATA, CLUSTER]].copy()
        values["value"] = assays[high].to_numpy(float) - assays[low].to_numpy(float)
        points = aggregate(values)
        draws = bootstrap_aggregates(values, counts, cluster_codes)
        wins = int((values["value"] > 1e-12).sum())
        ties = int((np.abs(values["value"]) <= 1e-12).sum())
        for aggregation in AGGREGATIONS:
            array = draws[aggregation]
            bounds = np.quantile(array, [0.025, 0.975])
            saved[f"{high}__minus__{low}__{aggregation}"] = array
            rows.append({
                "comparison": f"{high} - {low}",
                "higher_method": high,
                "lower_method": low,
                "aggregation": aggregation,
                "effect": points[aggregation],
                "ci_lower": float(bounds[0]),
                "ci_upper": float(bounds[1]),
                "p_two_sided": p_two_sided(array),
                "wins": wins,
                "ties": ties,
                "losses": len(assays) - wins - ties,
                "n_assays": len(assays),
                "bootstrap_repeats": repeats,
            })
    summary = pd.DataFrame(rows)
    summary["holm_adjusted_p"] = np.nan
    for aggregation in AGGREGATIONS:
        mask = summary["aggregation"] == aggregation
        summary.loc[mask, "holm_adjusted_p"] = holm(summary.loc[mask, "p_two_sided"].tolist())
    summary["holm_family"] = "four_structural_contrasts_per_aggregation"
    summary["holm_family_size"] = len(CONTRASTS)
    summary["significant_holm"] = summary["holm_adjusted_p"] < 0.05
    summary["ci_scope"] = "individual_unadjusted_percentile_interval"
    return summary, saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/results/per_assay_core.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/statistics"))
    parser.add_argument("--metrics", nargs="+", default=list(MODELS), metavar="COLUMN",
                        help="Wide numeric metric columns; default: four core model columns")
    parser.add_argument("--group-by", nargs="+", default=[], metavar="COLUMN",
                        help="For long endpoint tables, group by method/endpoint/cutoff before aggregation")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Bootstrap the four structural contrasts (requires the four model columns)")
    parser.add_argument("--repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()
    if args.bootstrap and args.group_by:
        parser.error("--bootstrap requires one wide four-model table, without --group-by")
    if args.repeats <= 0 or args.seed < 0:
        parser.error("--repeats must be positive and --seed must be non-negative")
    if len(set(args.metrics)) != len(args.metrics) or set(args.metrics) & set(args.group_by):
        parser.error("--metrics must be unique and distinct from --group-by columns")

    assays = pd.read_csv(args.input, dtype={key: str for key in (*METADATA, CLUSTER)})
    missing_groups = sorted(set(args.group_by) - set(assays.columns))
    if missing_groups:
        parser.error(f"Missing grouping columns: {', '.join(missing_groups)}")
    if assays.empty or assays[args.group_by].isna().any().any():
        parser.error("Input is empty or has missing group labels")
    groups = assays.groupby(args.group_by, sort=True) if args.group_by else [((), assays)]
    rows = []
    for key, group in groups:
        group = group.sort_values("DMS_id").reset_index(drop=True)
        validate_assays(group, args.metrics, bootstrap=False)
        key = key if isinstance(key, tuple) else (key,)
        labels = dict(zip(args.group_by, key))
        for metric in args.metrics:
            values = group[list(METADATA)].copy()
            values["value"] = group[metric].to_numpy(float)
            for aggregation, value in aggregate(values).items():
                rows.append({**labels, "metric_column": metric, "aggregation": aggregation,
                             "value": value, "n_assays": len(group),
                             "n_uniprot_targets": group["UniProt_ID"].nunique()})
    aggregates = pd.DataFrame(rows)
    summary, draws = core_contrasts(assays, args.seed, args.repeats) if args.bootstrap else (None, None)
    args.output.mkdir(parents=True, exist_ok=True)
    aggregates.to_csv(args.output / "aggregate_metrics.csv", index=False)
    if summary is not None:
        summary.to_csv(args.output / "paired_core_contrasts.csv", index=False)
        np.savez_compressed(args.output / "bootstrap_draws.npz", **draws)
    metadata = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": args.input.name,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "metrics": args.metrics, "group_by": args.group_by,
        "aggregation_levels": list(AGGREGATIONS),
        "bootstrap_enabled": args.bootstrap,
        "bootstrap_repeats": args.repeats if args.bootstrap else None,
        "bootstrap_seed": args.seed if args.bootstrap else None,
        "bootstrap_confidence_quantiles": [0.025, 0.975] if args.bootstrap else None,
        "assay_tie_tolerance": 1e-12 if args.bootstrap else None,
        "core_contrasts": list(CONTRASTS) if args.bootstrap else None,
        "scope": "Aggregation of saved assay metrics; no training or prediction regeneration",
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(aggregates.to_string(index=False))
    if summary is not None:
        print(summary[summary["aggregation"] == "ProteinGym_official"].to_string(index=False))


if __name__ == "__main__":
    main()
