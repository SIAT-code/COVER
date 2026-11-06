"""Regression and protocol tests; synthetic inputs are not experimental data."""
import json
import unittest
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import cover
import evaluate

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(ROOT / "examples/ralbp1.csv")
        cls.frame = cover.prepare_features(cls.raw, with_target=True)
        cls.models = json.loads((ROOT / "models/frozen_ridge.json").read_text())["models"]
        cls.results = pd.read_csv(ROOT / "data/results/per_assay_core.csv")

    def test_real_case_predictions_match_saved_results(self):
        expected = self.results.set_index("DMS_id").loc[self.raw.DMS_id.iloc[0]]
        columns = {"full": "hierarchical_cover", "fixed": "fixed_fusion",
                   "no_interactions": "additive_context", "no_context_main_effects": "ridge_cover"}
        for name, column in columns.items():
            model = next(m for m in self.models if m["model"] == name and m["outer_fold"] == 1)
            prediction = cover.predict(self.frame, model)
            observed = cover.spearman(prediction, self.frame.DMS_score)
            self.assertAlmostEqual(observed, expected[column], places=12)

    def test_prediction_does_not_require_or_use_outcomes(self):
        model = next(m for m in self.models if m["model"] == "full" and m["outer_fold"] == 1)
        unlabeled = cover.prepare_features(self.raw.drop(columns="DMS_score"))
        np.testing.assert_array_equal(cover.predict(unlabeled, model), cover.predict(self.frame, model))

    def test_token_order_and_invalid_positions(self):
        np.testing.assert_array_equal(cover.mutation_context("A2V:G8D", 10),
                                      cover.mutation_context("G8D:A2V", 10))
        for mutant in ["A2V:A2D", "G11D", "A0V", "A2*", "A2X"]:
            with self.assertRaises(ValueError):
                cover.mutation_context(mutant, 10)

    def test_unseen_categories_and_saved_dimensions(self):
        model = next(m for m in self.models if m["model"] == "full")
        changed = self.frame.iloc[:3].copy()
        changed[cover.CATEGORY_COLUMNS] = "unseen"
        context = cover.encode_context(changed, model["schema"])
        self.assertEqual(context.shape, (3, 19))
        np.testing.assert_array_equal(context[:, 10:], np.zeros((3, 9)))
        dimensions = {"full": 139, "fixed": 6, "no_interactions": 25, "no_context_main_effects": 120}
        for saved in self.models:
            cover.validate_model(saved)
            self.assertEqual(len(saved["coefficients"]), dimensions[saved["model"]] + 1)

    def test_intercept_is_not_penalized(self):
        # A numerical regression property, not a biological experiment.
        rng = np.random.default_rng(42)
        x = rng.normal(size=(80, 7))
        y = rng.normal(size=80)
        first = cover.fit_ridge(x, y, 100)
        shifted = cover.fit_ridge(x, y + 5, 100)
        self.assertAlmostEqual(shifted[0] - first[0], 5, places=12)
        np.testing.assert_allclose(shifted[1:], first[1:], atol=1e-12)

    def test_split_leakage_is_rejected(self):
        # Five synthetic partitions suffice to exercise the grouping preflight.
        frame = pd.concat([self.frame.iloc[:20].assign(DMS_id=f"synthetic_{f}", fold=f,
                          UniProt_ID="shared_target", cluster_representative=f"c{f}",
                          selected_for_training_when_outer_train=True) for f in range(5)], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "cross-fold UniProt_ID"):
            cover.nested_fit(frame, model_names=("fixed",))

    def test_nested_fit_excludes_target_labels(self):
        # Synthetic partition labels exercise the controller, not a new assay evaluation.
        frame = pd.concat([self.frame.iloc[:20].assign(DMS_id=f"synthetic_{f}", fold=f,
                          UniProt_ID=f"target{f}", cluster_representative=f"c{f}",
                          selected_for_training_when_outer_train=True) for f in range(5)], ignore_index=True)
        first = cover.nested_fit(frame, model_names=("fixed",))
        changed = frame.copy()
        mask = changed.fold == 0
        changed.loc[mask, "DMS_score"] = -changed.loc[mask, "DMS_score"]
        changed.loc[mask, "fitness_rank"] = 1 - changed.loc[mask, "fitness_rank"]
        second = cover.nested_fit(changed, model_names=("fixed",))
        first_model, second_model = first["models"]["fixed"][0], second["models"]["fixed"][0]
        self.assertEqual(first_model["alpha"], second_model["alpha"])
        np.testing.assert_array_equal(first_model["coefficients"], second_model["coefficients"])
        self.assertFalse(first["oof_predictions"].fixed.isna().any())

    def test_custom_model_bundle_uses_its_own_folds(self):
        # Swapped fold identifiers emulate an alternative allocation bundle.
        models = json.loads((ROOT / "models/frozen_ridge.json").read_text())["models"]
        for model in models:
            if model["outer_fold"] in (0, 1):
                model["outer_fold"] = 1 - model["outer_fold"]
        bundle = {"models": models, "assay_folds": {self.raw.DMS_id.iloc[0]: 0}}
        expected_model = next(m for m in self.models if m["model"] == "full" and m["outer_fold"] == 1)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "models.json").write_text(json.dumps(bundle))
            result = subprocess.run([sys.executable, str(ROOT / "run.py"), "predict",
                       "--input", str(ROOT / "examples/ralbp1.csv"), "--models", str(folder / "models.json"),
                       "--fold", "oof", "--output", str(folder / "predictions.csv")],
                       capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            observed = pd.read_csv(folder / "predictions.csv").prediction.to_numpy()
            np.testing.assert_allclose(observed, cover.predict(self.frame, expected_model), atol=1e-14, rtol=0)

    def test_aggregate_hierarchy_and_holm(self):
        # Deliberately unequal group sizes distinguish the three estimands.
        frame = pd.DataFrame({"UniProt_ID": ["a", "a", "b", "c"],
                              "coarse_selection_type": ["x", "x", "x", "y"],
                              "value": [0., 0., 1., 1.]})
        result = evaluate.aggregate(frame)
        self.assertAlmostEqual(result["DMS_macro"], 0.5)
        self.assertAlmostEqual(result["UniProt_macro"], 2 / 3)
        self.assertAlmostEqual(result["ProteinGym_official"], 0.75)
        np.testing.assert_allclose(evaluate.holm([0.007, 0.0432, 0.90411, 0.05019]),
                                   [0.028, 0.1296, 0.90411, 0.1296])

    def test_frozen_groups_and_training_sample(self):
        metadata = pd.read_csv(ROOT / "data/frozen/metadata.csv")
        sample = pd.read_csv(ROOT / "data/frozen/training_sample.csv.gz")
        self.assertEqual(len(metadata), 217)
        self.assertEqual(metadata.UniProt_ID.nunique(), 186)
        self.assertEqual(metadata.cluster_representative.nunique(), 172)
        for grouping in ["UniProt_ID", "cluster_representative"]:
            self.assertEqual(metadata.groupby(grouping).fold.nunique().max(), 1)
        self.assertFalse(sample.duplicated(["DMS_id", "mutant"]).any())
        self.assertLessEqual(sample.groupby("DMS_id").size().max(), 300)
        self.assertEqual(set(sample.DMS_id), set(metadata.DMS_id))


if __name__ == "__main__":
    unittest.main()
