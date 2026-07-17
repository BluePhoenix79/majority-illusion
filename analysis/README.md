# Condition-level analysis

`run_all_analyses.py` implements RQ1-RQ4 from final condition CSVs. The unit is
one entity x ratio x model x strategy x layout condition. Three repeated calls
form that condition's modal outcome; they are never treated as independent
observations. Inferential models cluster by entity.

## Run

```powershell
pip install -r requirements.txt
python analysis/run_all_analyses.py `
  results/conditions_standard.csv `
  results/conditions_cot.csv `
  --output-dir results/analysis
```

The loader requires protocol `2026-07-17-rich-distribution-v3-balanced`, the SHA-256 of
the current `data/entities.json`, three samples per condition, and a complete
75 entities x 6 ratios x 3 model slots factorial for every supplied
strategy/layout. It also validates entity-level assignment, the expected arm
allocation, and that 4:0 never requests an inline distribution.
Every stored inline distribution is independently reparsed: the three
resolution probabilities must be integer percentages summing to 100, the
source-conflict probability is validated independently, and reported counts
and means must match values recomputed from the stored distributions.
Strict final analysis also requires every condition to contain all three valid
primary samples. Post-hoc completion must be either successful or skipped for a
genuine three-way modal tie; failure-driven skips and post-hoc errors are
rejected. `--allow-incomplete` remains available only for explicitly labeled
pilots and audits.

`--allow-incomplete` permits a partial but otherwise valid v3 pilot.
`--audit-override` permits an explicitly labeled non-final audit. Neither
override produces final evidence. The default expects the balanced design of
37 answer-only and 38 structured-elicitation entities.

## Outputs

- `rq1_outcome_rates.csv` and `rq1_majority_rates.csv`: arm-stratified outcome
  rates with Wilson intervals.
- `rq1_posthoc_subjective_confidence.csv`: truth-neutral post-hoc confidence.
- `rq1_inline_distribution_summaries.csv` and `rq1_self_consistency.csv`:
  probability components and stability kept separate from confidence.
- `rq2_domain_flag_rates.csv` and `rq2_numeric_only_sensitivity.csv`: domain
  FLAG contrasts, stratified and adjusted for treatment arm.
- `rq3_position_outcomes.csv` and `rq3_position_balance.csv`: MAJ, MIN, COM,
  FLAG, OTHER, and FLAG-or-COM by first/middle/last minority position, plus the
  randomized position counts. Primary MAJ/FLAG inference uses exact normalized
  position (0=first, 1=last) and is restricted to 2:1, 3:1, and 4:1.
- `rq4_confidence_request_rates.csv` and
  `rq4_confidence_request_effects.csv`: assignment-based complete-case estimates on
  conflict ratios. FLAG is primary; MAJ, COM, and FLAG-or-COM are secondary.
  Percentage-point intervals use Newcombe's hybrid score method for two
  independent proportions.
- `rq4_missing_outcome_bounds.csv`: worst-case assigned-arm effect bounds when
  TIE, UNSCORED, or partially scored conditions have no modal binary outcome.
- `rq4_quality_diagnostics.csv`, `rq4_distribution_compliance.csv`, and
  `rq4_collection_error_*.csv`: TIE, UNSCORED, OTHER, partial scoring,
  post-hoc failures/skips, API/format errors, distribution compliance, and
  missingness by arm.
- `rq4_retry_diagnostics.csv`: physical format retries by arm, model, ratio,
  strategy, and layout. Retries are never counted as independent samples.
- `rq4_retry_sensitivity.csv`: the RQ4 treatment effects both with all complete
  conditions and after conservatively excluding every condition that needed a
  primary format retry.
- `rq4_distribution_and_consistency.csv`: inline probability components and
  self-consistency by arm, model, ratio, strategy, and layout.
- `rq4_conflict_abstention_*.csv`: distinguishes mentioning source conflict
  from abstaining when those v3 condition fields are present.
- `secondary_standard_vs_cot.csv`: secondary strategy comparison adjusted for
  assignment arm.
- `model_coefficients.csv`, `model_summaries.txt`, and the JSON/text reports:
  clustered inference, provenance, warnings, and the output manifest.

## Interpretation

RQ4 tests the pilot observation that structured, conflict-aware uncertainty
elicitation may alter answers rather than passively measure them. Because the
treatment asks for answer probabilities, an indeterminate probability, and a
separate source-conflict probability, it does not isolate probability numbers
from every other uncertainty cue. Pilot percentages are never hard-coded.
The 38/37 allocation supplies nearly balanced independent entity clusters for
the RQ4 comparison. Report model- and ratio-specific intervals even with this
stronger design.

RQ4 uses randomized entity assignment, not successful distribution parsing.
The point estimates and clustered models are assignment-based complete-case
analyses; worst-case bounds make outcome missingness explicit. Assignment must
equal actual exposure on conflict ratios. Protocol v3 requests no distribution
from either arm at 4:0, so 4:0 is excluded from every RQ4 treatment-effect table
and model.

The position analysis does not claim perfectly balanced positions. The domain
analysis estimates an association with the banking/general stimulus families,
not a causal effect of real-world financial sensitivity. Subjective confidence
is not factual accuracy. No context-length or Cohen's-kappa workflow is part of
this package.
