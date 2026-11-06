"""COVER: context-conditioned Ridge stacking of six protein-fitness scores.

Numerical methods are adapted from run_phase2_gate2a.py,
run_phase2e_context_controls.py and run_phase2_corrected_ridge.py in the
original COVER implementation. This module does not compute expert scores.
Ranks are defined over complete candidate libraries, identified by DMS_id.
Importing this module does not read data, fit models or write files.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata


EXPERTS = [
    "ESM2_650M", "GEMME", "MSA_Transformer_ensemble",
    "Tranception_L", "SaProt_650M_AF2", "VenusREM",
]
CATEGORY_COLUMNS = ["coarse_selection_type", "taxon", "MSA_Neff_L_category"]
MUTATION_FEATURES = [
    "mutation_log_count", "mutation_mean_normalized_position",
    "mutation_position_population_std",
    *[f"substituted_class_helmert_{i}" for i in range(1, 6)],
]
NUMERIC_FEATURES = MUTATION_FEATURES + [
    "log_sequence_length_div_10", "log_msa_neff_l_div_10",
]
RANK_COLUMNS = [f"rank_{expert}" for expert in EXPERTS]
MODEL_NAMES = ("full", "fixed", "no_interactions", "no_context_main_effects")
ALPHAS = (1.0, 10.0, 100.0)
RESIDUE_CLASSES = ("AVILM", "FWY", "STNQC", "DE", "KRH", "GP")
AA_CLASS = {aa: i for i, group in enumerate(RESIDUE_CLASSES) for aa in group}
TOKEN_RE = re.compile(r"^([A-Z])([1-9][0-9]*)([A-Z])$")


def _require_columns(frame, columns):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")


def helmert_basis(n):
    """Fixed orthonormal Helmert columns, in the supplied level order."""
    if n < 1:
        raise ValueError("A categorical vocabulary must contain at least one level")
    basis = np.zeros((n, n - 1), dtype=np.float64)
    for column in range(n - 1):
        denominator = np.sqrt((column + 1) * (column + 2))
        basis[:column + 1, column] = 1.0 / denominator
        basis[column + 1, column] = -(column + 1) / denominator
    return basis


COMPOSITION_BASIS = helmert_basis(6)


def fractional_rank(values):
    """Average rank/(n+1), stored as float32 as in the benchmark inputs."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Ranks require a nonempty finite one-dimensional score array")
    return (rankdata(values, method="average") / (len(values) + 1.0)).astype(np.float32)


def mutation_context(mutant, sequence_length):
    """Eight context coordinates for colon-separated substitutions."""
    length = float(sequence_length)
    if not np.isfinite(length) or length <= 0 or length != int(length):
        raise ValueError("seq_len must be a positive integer")
    positions, classes = [], []
    for token in str(mutant).split(":"):
        match = TOKEN_RE.fullmatch(token)
        if match is None or match[1] not in AA_CLASS or match[3] not in AA_CLASS:
            raise ValueError(f"Invalid standard-amino-acid substitution: {token!r}")
        position = int(match[2])
        if position > length:
            raise ValueError(f"Position {position} exceeds seq_len {int(length)}")
        positions.append(position)
        classes.append(AA_CLASS[match[3]])
    if len(set(positions)) != len(positions):
        raise ValueError(f"Repeated substitution position in {mutant!r}")
    normalized = np.asarray(positions, dtype=np.float64) / length
    proportions = np.asarray([classes.count(i) / len(classes) for i in range(6)])
    return np.concatenate((
        [np.log1p(len(positions)), normalized.mean(), normalized.std(ddof=0)],
        proportions @ COMPOSITION_BASIS,
    )).astype(np.float32)


def prepare_features(frame, *, with_target=False):
    """Prepare complete-library ranks and numeric context without filtering rows.

    Each row needs DMS_id, mutant, six expert scores, seq_len, MSA_Neff_L and
    the three categorical fields. Only training/evaluation preparation needs
    DMS_score (with_target=True). Fitness values never enter predictor features.
    Missing MSA_Neff_L is replaced by zero; missing category values are Unknown.
    """
    columns = ["DMS_id", "mutant", "seq_len", "MSA_Neff_L", *EXPERTS, *CATEGORY_COLUMNS]
    _require_columns(frame, columns + (["DMS_score"] if with_target else []))
    if frame.empty or frame[["DMS_id", "mutant"]].isna().any().any():
        raise ValueError("Each nonempty library needs assay and mutation identifiers")
    data = frame.copy().reset_index(drop=True)
    data["DMS_id"] = data["DMS_id"].astype(str)
    data["mutant"] = data["mutant"].astype(str)
    if data.duplicated(["DMS_id", "mutant"]).any():
        raise ValueError("Duplicate assay/mutation identifiers")
    metadata = ["seq_len", "MSA_Neff_L", *CATEGORY_COLUMNS]
    if (data.groupby("DMS_id")[metadata].nunique(dropna=False) > 1).any().any():
        raise ValueError("Protein and assay metadata must be constant within each library")
    score_columns = EXPERTS + (["DMS_score"] if with_target else [])
    if not np.isfinite(data[score_columns].to_numpy(np.float64)).all():
        raise ValueError("Scores must be finite; select the eligible library before preparation")
    for score in score_columns:
        destination = "fitness_rank" if score == "DMS_score" else f"rank_{score}"
        data[destination] = data.groupby("DMS_id", sort=False)[score].transform(fractional_rank).astype(np.float32)
    mutation = np.asarray([
        mutation_context(mutant, length)
        for mutant, length in zip(data["mutant"], data["seq_len"])
    ], dtype=np.float32)
    data[MUTATION_FEATURES] = mutation
    depth = data["MSA_Neff_L"].fillna(0).to_numpy(np.float64)
    if not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError("MSA_Neff_L must be nonnegative and finite, or missing")
    data["log_sequence_length_div_10"] = (np.log1p(data["seq_len"].to_numpy(float)) / 10.0).astype(np.float32)
    data["log_msa_neff_l_div_10"] = (np.log1p(depth) / 10.0).astype(np.float32)
    return data


def fit_schema(training_frame):
    """Learn category vocabularies exclusively from the supplied fit partition."""
    _require_columns(training_frame, ["DMS_id", *CATEGORY_COLUMNS])
    if training_frame.empty:
        raise ValueError("Cannot fit a schema on an empty training partition")
    if (training_frame.groupby("DMS_id")[CATEGORY_COLUMNS].nunique(dropna=False) > 1).any().any():
        raise ValueError("Categorical metadata must be constant within an assay")
    metadata = training_frame.drop_duplicates("DMS_id")
    return {column: sorted(metadata[column].fillna("Unknown").astype(str).unique().tolist())
            for column in CATEGORY_COLUMNS}


def context_names(schema):
    names = list(NUMERIC_FEATURES)
    for column in CATEGORY_COLUMNS:
        levels = schema[column]
        if not levels or levels != sorted(set(levels)) or not all(isinstance(v, str) for v in levels):
            raise ValueError(f"Invalid sorted category vocabulary for {column}")
        names.extend(f"{column}__helmert_{i}" for i in range(1, len(levels)))
    return names


def encode_context(prepared_frame, schema):
    """Apply a saved schema; unseen categories receive an all-zero contrast."""
    _require_columns(prepared_frame, [*NUMERIC_FEATURES, *CATEGORY_COLUMNS])
    context_names(schema)
    blocks = [prepared_frame[NUMERIC_FEATURES].to_numpy(np.float32)]
    for column in CATEGORY_COLUMNS:
        levels = schema[column]
        basis = helmert_basis(len(levels))
        values = prepared_frame[column].fillna("Unknown").astype(str).to_numpy()
        block = np.zeros((len(values), len(levels) - 1), dtype=np.float32)
        for index, level in enumerate(levels):
            block[values == level] = basis[index].astype(np.float32)
        blocks.append(block)
    result = np.column_stack(blocks).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Context contains nonfinite values")
    return result


def feature_names(schema, model="full"):
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model specification: {model}")
    if model == "fixed":
        return list(RANK_COLUMNS)
    names = context_names(schema)
    result = list(RANK_COLUMNS)
    if model != "no_context_main_effects":
        result.extend(names)
    if model != "no_interactions":
        result.extend(f"{rank} x {name}" for name in names for rank in RANK_COLUMNS)
    return result


def design_matrix(ranks, context, model="full"):
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model specification: {model}")
    ranks = np.asarray(ranks)
    context = np.asarray(context)
    if ranks.ndim != 2 or ranks.shape[1] != len(EXPERTS) or context.ndim != 2 or len(ranks) != len(context):
        raise ValueError("Expected six expert ranks and one context row per variant")
    if not np.isfinite(ranks).all() or not np.isfinite(context).all():
        raise ValueError("Predictor inputs must be finite")
    if model == "no_context_main_effects":
        # Preserve the original 120-column ablation's float32 product boundary.
        ranks32, context32 = ranks.astype(np.float32), context.astype(np.float32)
        return np.column_stack([ranks32, *[
            ranks32 * context32[:, [i]] for i in range(context32.shape[1])
        ]]).astype(np.float64)
    ranks64, context64 = ranks.astype(np.float64), context.astype(np.float64)
    if model == "fixed":
        return ranks64
    blocks = [ranks64, context64]
    if model == "full":
        blocks.extend(ranks64 * context64[:, [i]] for i in range(context64.shape[1]))
    return np.column_stack(blocks)


def rank_diagnostic(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    augmented = np.column_stack([np.ones(len(matrix), dtype=np.float64), matrix])
    singular = np.linalg.svd(augmented, compute_uv=False)
    threshold = max(augmented.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.sum(singular > threshold))
    return {"rows": len(augmented), "columns_including_intercept": augmented.shape[1],
            "numerical_rank": rank, "full_rank": rank == augmented.shape[1]}


def fit_ridge(matrix, target, alpha):
    """Solve summed squared error + alpha*||beta||^2, with a free intercept."""
    matrix, target = np.asarray(matrix, np.float64), np.asarray(target, np.float64)
    if matrix.ndim != 2 or target.shape != (len(matrix),) or not len(matrix):
        raise ValueError("Incompatible or empty Ridge inputs")
    if not np.isfinite(matrix).all() or not np.isfinite(target).all() or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge inputs must be finite and alpha must be positive")
    if not rank_diagnostic(matrix)["full_rank"]:
        raise ValueError("Rank-deficient design under the benchmark fitting protocol")
    augmented = np.column_stack([np.ones(len(matrix), dtype=np.float64), matrix])
    penalty = np.eye(augmented.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ target)


def make_model(model, coefficients, schema, alpha, outer_fold):
    """Return a JSON-serializable model with the complete inference schema."""
    if model == "fixed":
        schema = {}
    record = {"format_version": 1, "model": model, "outer_fold": int(outer_fold),
              "alpha": float(alpha), "experts": list(EXPERTS), "schema": schema,
              "feature_names": feature_names(schema, model),
              "coefficients": np.asarray(coefficients, dtype=np.float64).tolist()}
    validate_model(record)
    return record


def validate_model(model):
    if model.get("format_version") != 1 or model.get("experts") != EXPERTS:
        raise ValueError("Unsupported model format or expert order")
    expected = feature_names(model["schema"], model["model"])
    if model.get("feature_names") != expected:
        raise ValueError("Saved feature order does not match the saved schema")
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    if coefficients.shape != (len(expected) + 1,) or not np.isfinite(coefficients).all():
        raise ValueError("Invalid model coefficient vector")
    if not np.isfinite(model["alpha"]) or model["alpha"] <= 0:
        raise ValueError("Invalid saved Ridge penalty")


def _predict_arrays(ranks, context, coefficients, model, chunk_size=100_000):
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    result = np.empty(len(ranks), dtype=np.float64)
    for start in range(0, len(ranks), chunk_size):
        stop = min(start + chunk_size, len(ranks))
        matrix = design_matrix(ranks[start:stop], context[start:stop], model)
        result[start:stop] = coefficients[0] + matrix @ coefficients[1:]
    return result


def predict(prepared_frame, model, *, chunk_size=100_000):
    """Predict raw scores from prepared unlabeled candidate libraries."""
    validate_model(model)
    _require_columns(prepared_frame, RANK_COLUMNS)
    ranks = prepared_frame[RANK_COLUMNS].to_numpy(np.float32)
    context = (np.empty((len(ranks), 0), dtype=np.float32) if model["model"] == "fixed"
               else encode_context(prepared_frame, model["schema"]))
    return _predict_arrays(ranks, context, np.asarray(model["coefficients"], np.float64),
                           model["model"], chunk_size)


def predict_ensemble(prepared_frame, models, *, chunk_size=100_000):
    """Average raw predictions from the supplied frozen outer-fold models."""
    if not models or len({m["model"] for m in models}) != 1:
        raise ValueError("An ensemble must contain one nonempty model specification")
    if len({m["outer_fold"] for m in models}) != len(models):
        raise ValueError("Duplicate outer-fold models in ensemble")
    return np.mean([predict(prepared_frame, m, chunk_size=chunk_size) for m in models], axis=0)


def spearman(prediction, outcome):
    prediction, outcome = np.asarray(prediction, float), np.asarray(outcome, float)
    if prediction.shape != outcome.shape or prediction.ndim != 1:
        raise ValueError("Spearman inputs must be aligned vectors")
    if not np.isfinite(prediction).all() or not np.isfinite(outcome).all():
        raise ValueError("Spearman inputs must be finite")
    if len(prediction) < 2 or np.ptp(prediction) == 0 or np.ptp(outcome) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(prediction, method="average"),
                            rankdata(outcome, method="average"))[0, 1])


def _assay_mean_spearman(frame, prediction):
    values = [spearman(prediction[indices], frame["DMS_score"].to_numpy(float)[indices])
              for indices in frame.groupby("DMS_id", sort=True).indices.values()]
    if not np.isfinite(values).any():
        raise ValueError("No defined validation assay Spearman correlations")
    return float(np.nanmean(values))


def nested_fit(prepared_frame, model_names=("full", "fixed"), alphas=ALPHAS):
    """Fit independently tuned models using the five frozen folds and sample IDs.

    Input additionally requires DMS_score, fitness_rank, fold, UniProt_ID,
    cluster_representative and selected_for_training_when_outer_train.
    No split or training subsample is generated here. Returns model records,
    out-of-fold raw predictions and the inner-selection table.
    """
    required = ["DMS_id", "mutant", "DMS_score", "fitness_rank", "fold", "UniProt_ID",
                "cluster_representative", "selected_for_training_when_outer_train",
                *RANK_COLUMNS, *NUMERIC_FEATURES, *CATEGORY_COLUMNS]
    _require_columns(prepared_frame, required)
    data = prepared_frame.reset_index(drop=True)
    if not model_names or len(set(model_names)) != len(model_names) or any(m not in MODEL_NAMES for m in model_names):
        raise ValueError("Specify distinct supported Ridge models")
    alphas = tuple(float(a) for a in alphas)
    if not alphas or len(set(alphas)) != len(alphas) or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("Specify distinct positive alpha candidates")
    if set(data["fold"].unique()) != set(range(5)):
        raise ValueError("The benchmark protocol requires outer folds 0 through 4")
    for column in ["DMS_id", "UniProt_ID", "cluster_representative"]:
        if data[column].isna().any() or data.groupby(column)["fold"].nunique().max() != 1:
            raise ValueError(f"Missing or cross-fold {column} grouping")
    if data.duplicated(["DMS_id", "mutant"]).any():
        raise ValueError("Duplicate assay/mutation training rows")
    sample_column = data["selected_for_training_when_outer_train"]
    if not sample_column.isin([True, False]).all():
        raise ValueError("Training-sample indicators must be booleans")
    counts = data.assign(_selected=sample_column.astype(bool)).groupby("DMS_id")["_selected"].sum()
    if (counts < 1).any() or (counts > 300).any():
        raise ValueError("Each assay needs between 1 and 300 frozen training variants")
    ranks = data[RANK_COLUMNS].to_numpy(np.float32)
    folds = data["fold"].to_numpy(int)
    sampled = sample_column.to_numpy(bool)
    target = data["fitness_rank"].to_numpy(np.float64)
    oof = {model: np.full(len(data), np.nan) for model in model_names}
    models = {model: [] for model in model_names}
    selection = []
    for outer in range(5):
        scores = {model: {alpha: [] for alpha in alphas} for model in model_names}
        for inner in range(5):
            if inner == outer:
                continue
            fit_scope = (folds != outer) & (folds != inner)
            train, validation = fit_scope & sampled, folds == inner
            schema = fit_schema(data.loc[fit_scope])
            train_context = encode_context(data.loc[train], schema)
            validation_context = encode_context(data.loc[validation], schema)
            validation_frame = data.loc[validation].reset_index(drop=True)
            for model in model_names:
                matrix = design_matrix(ranks[train], train_context, model)
                for alpha in alphas:
                    coefficients = fit_ridge(matrix, target[train], alpha)
                    prediction = _predict_arrays(ranks[validation], validation_context, coefficients, model)
                    score = _assay_mean_spearman(validation_frame, prediction)
                    scores[model][alpha].append(score)
                    selection.append({"outer_fold": outer, "inner_validation_fold": inner,
                                      "model": model, "alpha": alpha, "assay_mean_spearman": score,
                                      "n_train": int(train.sum()), "n_validation": int(validation.sum()),
                                      "selected": False})
        fit_scope, test = folds != outer, folds == outer
        train = fit_scope & sampled
        schema = fit_schema(data.loc[fit_scope])
        context = encode_context(data.loc[train], schema)
        for model in model_names:
            means = {alpha: float(np.mean(values)) for alpha, values in scores[model].items()}
            chosen = max(alphas, key=lambda alpha: (means[alpha], -alpha))
            for alpha, value in means.items():
                selection.append({"outer_fold": outer, "inner_validation_fold": "MEAN",
                                  "model": model, "alpha": alpha, "assay_mean_spearman": value,
                                  "n_train": None, "n_validation": None, "selected": alpha == chosen})
            coefficients = fit_ridge(design_matrix(ranks[train], context, model), target[train], chosen)
            record = make_model(model, coefficients, schema, chosen, outer)
            models[model].append(record)
            oof[model][test] = predict(data.loc[test], record)
    if any(not np.isfinite(values).all() for values in oof.values()):
        raise ValueError("Missing or nonfinite out-of-fold predictions")
    predictions = data[["DMS_id", "mutant", "fold"]].copy()
    for model, values in oof.items():
        predictions[model] = values
    return {"models": models, "oof_predictions": predictions,
            "selection": pd.DataFrame(selection)}
