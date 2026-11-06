# COVER

Minimal code release accompanying **COVER: Context-conditioned stacking of
zero-shot protein fitness experts for variant ranking without labels from the
target assay**.

First public software release: **v1.0.0**.

[Download the v1.0.0 package](https://github.com/SIAT-code/COVER/releases/download/v1.0.0/COVER-v1.0.0-minimal.zip).

COVER learns a Ridge combination of six fixed expert scores from historical
DMS assays. The full model has 139 predictors: six expert ranks, 19 context
terms, and 114 score–context interactions, plus an unpenalized intercept.
It ranks a complete target library without its experimental fitness labels.

## Contents

- `cover.py`: feature construction, four Ridge specifications, nested training,
  and frozen-model prediction.
- `run.py`: training and prediction command line.
- `prepare.py`: reconstruct the frozen benchmark inputs from ProteinGym v1.3.
- `evaluate.py`: assay aggregation and paired sequence-cluster bootstrap.
- `models/frozen_ridge.json`: the original five outer-fold fits of each Ridge
  specification, including coefficients and category vocabularies.
- `data/`: frozen metadata, training identifiers, alternative folds, and the
  reported assay-level results and selected contrasts.
- `examples/ralbp1.csv`: the complete 1,332-variant RalBP1 example used in the paper.
- `tests/`: regression and protocol checks.

The package implements the four Ridge models. It uses precomputed expert scores;
it does not train the six upstream predictors. The reported results of the other
architectures are included for numerical comparison; their training pipelines
and manuscript figure-generation code are outside this minimal release.

## Install and check

Use Python 3.10. The dependency versions below were tested with Python 3.10.16;
no GPU is required.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Predict

Reproduce the paper's RalBP1 example with its held-out model:

```bash
python run.py predict --input examples/ralbp1.csv --fold oof --output outputs/ralbp1.csv
```

For a new scored library, average the five frozen outer-fold predictions:

```bash
python run.py predict --input new_library.csv --fold ensemble --output outputs/predictions.csv
```

Each input row represents one variant. Required columns are:

| Fields | Meaning |
| --- | --- |
| `DMS_id`, `mutant` | Library identifier and substitutions, e.g. `A2V:G8D` |
| `seq_len`, `MSA_Neff_L` | Target length and MSA depth; missing depth becomes zero |
| `coarse_selection_type`, `taxon`, `MSA_Neff_L_category` | Unlabeled assay/protein metadata |
| `ESM2_650M`, `GEMME`, `MSA_Transformer_ensemble`, `Tranception_L`, `SaProt_650M_AF2`, `VenusREM` | Six expert scores |

Expert scores must already have the larger-is-more-favorable direction used by
ProteinGym. All eligible variants in a library must be supplied together, since
fractional ranks are calculated within each `DMS_id`. Rows missing an expert
score are excluded. Prediction neither requires nor uses `DMS_score`.
Substitution positions refer to the supplied target construct. Unknown category
levels map to zero contrasts in the saved training vocabulary.

`--model` selects `full` (default), `fixed`, `no_interactions`, or
`no_context_main_effects`. The output includes the raw model score and its
within-library fractional rank. For benchmark assays, `--fold oof` uses the
corresponding held-out fit recorded in the selected model bundle; ensemble mode
rejects that bundle's training-cohort IDs.

## Recalculate the core results

```bash
python evaluate.py --bootstrap --output outputs/statistics
```

This recalculates Table 2 and the four Ridge rows of Table 1 from archived
assay-level correlations. It uses 10,000 paired resamples of the 172 sequence
clusters, seed 20260808, individual percentile intervals, and Holm correction
across four structural contrasts separately for each aggregation.

The official full-minus-fixed effect is approximately 0.00546007, with an
individual 95% interval of [0.00151543, 0.00970013] and Holm-adjusted p of
0.02799720. These are checks of the manuscript's saved results, not a new
independent experiment. The expected values are in
`data/results/paired_core_expected.csv`; they are not inputs to the computation.

Other Table 1 columns can be aggregated with, for example:

```bash
python evaluate.py --metrics LightGBM MLP "Softmax gate" --output outputs/architectures
```

## Refit the Ridge models

Download `zero_shot_substitutions_scores.zip` from
[ProteinGym v1.3](https://zenodo.org/records/15293562). The archive is external
because of its size. `prepare.py` verifies its SHA256 before reconstructing the
2,465,767 complete-case observations and joining the 64,236 frozen training IDs.
The included metadata preserve the training cohort and source row order.

```bash
python prepare.py --scores data/external/zero_shot_substitutions_scores.zip --output outputs/benchmark.csv.gz
OPENBLAS_NUM_THREADS=2 python run.py train --input outputs/benchmark.csv.gz --models full fixed no_interactions no_context_main_effects --output outputs/refit
```

The five outer folds keep both sequence clusters and UniProt targets intact.
Within each outer-training partition, four inner folds independently choose
alpha from {1, 10, 100} for each model. Selection averages the four inner-fold
assay-mean Spearman values equally and resolves exact ties toward the smaller
alpha. Each fit uses at most 300 frozen sampled variants per training assay;
validation and prediction use complete eligible libraries. Category vocabularies
are learned only from the corresponding training partition. No additional
feature standardization is fitted.

Frozen feature ranks and context coordinates use float32. The full and
no-interaction designs are formed in float64. The 120-column ablation retains
the original float32 interaction multiplication before conversion to float64.
The saved alpha is 100 in every released fit. New refits can differ slightly
with numerical libraries; the frozen fits are the manuscript inference reference.

To use an alternative allocation, replace `fold` by one of `fold_1` through
`fold_5` from `data/frozen/alternative_folds.csv` and rerun the same training
procedure. These allocations are sensitivity analyses of the same cohort.
The training command saves the actual assay-to-fold assignments in `models.json`.
Use `--models outputs/refit/models.json --fold oof` to apply those refitted models
to their held-out assays.

## Data and interpretation

| File | Scope |
| --- | --- |
| `data/results/per_assay_core.csv` | 217 assays: all reported expert, architecture and ablation Spearman values |
| `data/results/maintext_contrasts.csv` | Selected S6, S7, S12 and S16 contrasts; original Holm families are retained |
| `data/results/alternative_fold_effects.csv` | Five alternative assignments, all three Spearman aggregations |
| `data/external/assay_metrics_12.csv` | All 13 endpoints for the 12 external assays and two frozen models |
| `data/external/targets_12.csv` | External target identifiers, observation counts and MaveDB source links |

The four core result keys map to code models as follows:

| Result key | Code model | Predictors, excluding intercept |
| --- | --- | ---: |
| `hierarchical_cover` | `full` | 139 |
| `fixed_fusion` | `fixed` | 6 |
| `additive_context` | `no_interactions` | 25 |
| `ridge_cover` | `no_context_main_effects` | 120 |

External endpoints are averaged within each protein and then equally across
the nine proteins. All positive and negative differences are retained. These
descriptive results have no associated significance tests. The external files
support reaggregation of archived metrics; they do not supply a complete replay
of external upstream expert scoring. External target metadata were checked
against MaveDB snapshots obtained on 2026-10-29, rather than asserted to be the
original scoring-time snapshots.

`data/provenance.json` records the source artifacts, hashes and export operations.
`VALIDATION.json` records checks actually run for this release. The released
code and data contain neither manuscript drafts nor the historical development
and document-editing scripts. ProteinGym and MaveDB inputs remain attributable
to their original sources and terms.

Please cite the COVER manuscript and the
[ProteinGym benchmark](https://github.com/OATML-Markslab/ProteinGym), together
with the underlying expert methods and assay studies appropriate to your use.
