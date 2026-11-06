#!/usr/bin/env python3
"""Join the public ProteinGym v1.3 score archive to COVER's frozen cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cover import EXPERTS

ROOT = Path(__file__).resolve().parent
ARCHIVE_SHA256 = "3fd7cdb5e78f1d43cabfabfeb6578c252b63af23ba2ab44db0094dc3a42de36d"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True, help="Unmodified zero_shot_substitutions_scores.zip from ProteinGym v1.3")
    parser.add_argument("--output", type=Path, required=True, help="Joined raw-score table (.csv or .csv.gz)")
    args = parser.parse_args()
    archive_hash = sha256(args.scores)
    if archive_hash != ARCHIVE_SHA256:
        raise ValueError("Score archive SHA256 differs from the manuscript's frozen ProteinGym v1.3 input")
    metadata_path = ROOT / "data/frozen/metadata.csv"
    sample_path = ROOT / "data/frozen/training_sample.csv.gz"
    metadata = pd.read_csv(metadata_path)
    sample = pd.read_csv(sample_path)
    required = ["mutant", "DMS_score", *EXPERTS]
    tables = []
    with zipfile.ZipFile(args.scores) as archive:
        members = {Path(name).stem: name for name in archive.namelist() if name.endswith(".csv")}
        for dms_id in metadata["DMS_id"]:
            if dms_id not in members:
                raise ValueError(f"Missing assay in score archive: {dms_id}")
            with archive.open(members[dms_id]) as handle:
                frame = pd.read_csv(handle, usecols=lambda column: column in required)
            if set(required) - set(frame.columns):
                raise ValueError(f"Missing required scores for {dms_id}")
            frame = frame[required].dropna(subset=required).copy()
            if not np.isfinite(frame[["DMS_score", *EXPERTS]].to_numpy(float)).all():
                raise ValueError(f"Nonfinite scores in {dms_id}")
            frame.insert(0, "DMS_id", dms_id)
            tables.append(frame)
    data = pd.concat(tables, ignore_index=True).merge(metadata, on="DMS_id", how="left", validate="many_to_one", sort=False)
    keys = pd.MultiIndex.from_frame(data[["DMS_id", "mutant"]])
    sample_keys = pd.MultiIndex.from_frame(sample[["DMS_id", "mutant"]])
    if keys.has_duplicates or sample_keys.has_duplicates:
        raise ValueError("Duplicate cohort or training-sample identifiers")
    data["selected_for_training_when_outer_train"] = keys.isin(sample_keys)
    if len(data) != 2465767 or data["DMS_id"].nunique() != 217:
        raise ValueError("The reconstructed cohort differs from the manuscript")
    if int(data["selected_for_training_when_outer_train"].sum()) != len(sample):
        raise ValueError("Frozen training IDs did not map one-to-one")
    counts = data.groupby("DMS_id").size()
    expected = pd.read_csv(ROOT / "data/results/per_assay_core.csv").set_index("DMS_id")["n_variants"]
    if not counts.sort_index().equals(expected.sort_index()):
        raise ValueError("Assay-specific cohort counts differ from the manuscript")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "score_archive_sha256": archive_hash,
        "metadata_sha256": sha256(metadata_path),
        "training_sample_sha256": sha256(sample_path),
        "script_sha256": sha256(__file__),
        "output_sha256": sha256(args.output),
        "assays": int(data["DMS_id"].nunique()), "variants": len(data),
        "training_identifiers": len(sample), "model_fitting_performed": False,
    }
    args.output.with_suffix(args.output.suffix + ".run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
