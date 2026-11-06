#!/usr/bin/env python3
"""Train COVER or apply the frozen manuscript models to scored libraries."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import cover

ROOT = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prediction = sub.add_parser("predict", help="Rank complete, pre-scored candidate libraries")
    prediction.add_argument("--input", type=Path, required=True)
    prediction.add_argument("--models", type=Path, default=ROOT / "models/frozen_ridge.json")
    prediction.add_argument("--model", choices=cover.MODEL_NAMES, default="full")
    prediction.add_argument("--fold", choices=["ensemble", "oof", "0", "1", "2", "3", "4"], default="ensemble")
    prediction.add_argument("--output", type=Path, required=True)
    training = sub.add_parser("train", help="Run the nested five-fold Ridge protocol")
    training.add_argument("--input", type=Path, required=True)
    training.add_argument("--models", nargs="+", choices=cover.MODEL_NAMES, default=["full", "fixed"])
    training.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = pd.read_csv(args.input, float_precision="round_trip")
    required_scores = list(cover.EXPERTS)
    eligible = raw.dropna(subset=required_scores + (["DMS_score"] if args.command == "train" else [])).copy()
    if eligible.empty:
        raise ValueError("No complete scored variants remain")
    frame = cover.prepare_features(eligible, with_target=args.command == "train")
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": args.command,
        "input_sha256": digest(args.input),
        "code_sha256": {name: digest(ROOT / name) for name in ["cover.py", "run.py"]},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "input_rows": len(raw), "eligible_rows": len(frame),
    }
    if args.command == "predict":
        bundle = json.loads(args.models.read_text(encoding="utf-8"))
        models = bundle["models"]
        selected = [m for m in models if m["model"] == args.model]
        if len(selected) != 5 or {int(m["outer_fold"]) for m in selected} != set(range(5)):
            raise ValueError("A complete set of five frozen outer-fold fits is required")
        if not bundle.get("assay_folds"):
            raise ValueError("The model bundle must record its training assay-to-fold assignments")
        known = pd.Series(bundle["assay_folds"], dtype=int)
        if not known.isin(range(5)).all():
            raise ValueError("Invalid model-bundle fold assignments")
        known_fold = frame["DMS_id"].map(known)
        if args.fold == "ensemble":
            if known_fold.notna().any():
                raise ValueError("Training-cohort assays require their held-out fit; use --fold oof")
            scores = cover.predict_ensemble(frame, selected)
        else:
            if args.fold == "oof":
                if known_fold.isna().any():
                    raise ValueError("OOF mode requires assay identifiers in this model bundle's training cohort")
                assignments = known_fold.to_numpy(int)
            else:
                assignments = np.full(len(frame), int(args.fold))
                if (known_fold.notna() & (known_fold != int(args.fold))).any():
                    raise ValueError("Requested model was not held out from these training-cohort assays")
            scores = np.empty(len(frame), dtype=np.float64)
            for model in selected:
                mask = assignments == int(model["outer_fold"])
                if mask.any():
                    scores[mask] = cover.predict(frame.loc[mask], model)
        result = frame[["DMS_id", "mutant"]].copy()
        result["prediction"] = scores
        result["predicted_fractional_rank"] = result.groupby("DMS_id")["prediction"].transform(
            lambda x: x.rank(method="average") / (len(x) + 1)
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        record.update(model=args.model, fold=args.fold, models_sha256=digest(args.models))
        report = args.output.with_suffix(args.output.suffix + ".run.json")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        # nested_fit keeps the outer fold out of vocabularies, tuning, and fitting.
        fitted = cover.nested_fit(frame, model_names=tuple(args.models))
        # The core function returns serializable models, OOF predictions, and selection records.
        models = [m for group in fitted["models"].values() for m in group]
        predictions, selection = fitted["oof_predictions"], fitted["selection"]
        assay_folds = {str(key): int(value) for key, value in frame.groupby("DMS_id")["fold"].first().items()}
        (args.output / "models.json").write_text(json.dumps({"models": models, "assay_folds": assay_folds}, indent=2) + "\n", encoding="utf-8")
        predictions.to_csv(args.output / "oof_predictions.csv.gz", index=False)
        selection.to_csv(args.output / "alpha_selection.csv", index=False)
        record.update(models=args.models, alphas=[1, 10, 100], folds=5)
        report = args.output / "run.json"
    report.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
