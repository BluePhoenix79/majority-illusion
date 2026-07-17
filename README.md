# Majority Illusion in RAG

This repository studies how three language models resolve contradictory,
retrieved-style documents about 75 fictional entities. The independent
variable is the majority:minority document ratio; the primary behavioral
outcomes are `MAJ`, `MIN`, `COM`, `FLAG`, and `OTHER`.

## Current protocol

- 75 entities: 30 banking and 45 general-corporate.
- Six ratios: `4:0`, `3:1`, `2:2`, `4:1`, `2:1`, and `3:2`.
- Three fixed model slots: Gemini 3.5 Flash, DeepSeek V4 Flash, and Claude
  Haiku 4.5. DeepSeek and Claude use separate OpenRouter slots and cannot be
  silently swapped or pooled.
- Three valid, byte-identical primary calls per entity x ratio x model
  condition at temperature 1.0. A format-invalid response is retried at most
  twice with the identical prompt; every attempt remains in the raw log, while
  only the final trial record enters the modal outcome. Repeated calls are not
  independent observations.
- A separate post-hoc 0-100 judgment asks whether the modal answer is the best
  resolution of the supplied documents. It is subjective confidence, not a
  calibrated probability of factual correctness.
- On conflict ratios, 38 entities are assigned to a structured distribution
  request and 37 to a matched answer-only control. Every `4:0`
  prompt is answer-only so an absent minority value is never introduced.
- Inline distributions report `p_claim_a`, `p_claim_b`, `p_indeterminate`, and
  independent `p_sources_conflict`. Claim A/B are counterbalanced and stored
  canonically as majority/minority probabilities.
- No truth labels, Platt calibration, Brier/ECE analysis, Cohen's-kappa
  workflow, or context-length research question is part of this protocol.

The near-balanced 38/37 allocation provides substantially stronger independent
clusters for estimating whether structured uncertainty elicitation changes
answers. It does not increase the planned scientific calls: every entity still
gets three primary trials and, when a unique modal answer exists, one post-hoc
call. Physical API attempts can be slightly higher when format retries occur.

## Validate without API calls

```powershell
python -m unittest tests.test_new_approach visualizations.test_common `
  analysis.tests.test_run_all_analyses

python harness/run_experiment.py --mock `
  --entity-ids E003 E001 --ratios 3:1 4:0 `
  --models gemini deepseek claude
```

`E003` is in the distribution-request arm and `E001` is in the default
answer-only arm, so that mock exercises both arms and the `4:0` safeguard.

## Collect final data

After the required repository sync and a small live pilot, run each strategy
once. Each command plans at most 5,400 scientific calls (three primary trials
plus one post-hoc call across 1,350 conditions); modal ties can skip post-hoc
calls and logged format retries can add physical API attempts.

```powershell
python harness/run_experiment.py --strategy standard `
  --output results/run_v3_standard.csv `
  --condition-output results/conditions_v3_standard.csv

python harness/run_experiment.py --strategy cot `
  --output results/run_v3_cot.csv `
  --condition-output results/conditions_v3_cot.csv
```

When Standard and CoT are chained, use `&&`. The collector exits nonzero unless
every condition contains all required valid primary samples and either a valid
post-hoc result or a genuine modal-tie skip, so CoT will not start after a failed
Standard run. Row coverage alone is never reported as successful completion.

Do not combine interrupted, legacy, or different-protocol CSVs with final v3
data. The collector records protocol version, dataset hash, run seed, layout,
requested/returned model IDs, errors, distribution compliance, token usage,
conflict mention, abstention, and retry provenance. Permanent account/request
errors such as OpenRouter HTTP 402 stop collection immediately after the fatal
attempt is flushed to the raw CSV. Recoverable errors may finish writing
diagnostic rows, but the final completion gate still returns a nonzero status.

## Analyze

```powershell
python analysis/run_all_analyses.py `
  results/conditions_v3_standard.csv `
  results/conditions_v3_cot.csv `
  --output-dir results/analysis
```

The strict analysis rejects the wrong dataset, mixed protocol versions,
duplicate conditions, incomplete final factorials, model-slot drift, treatment
assignment/exposure mismatches, invalid stored probability sums or means, and
distributions in `4:0`. It produces
arm-aware RQ1-RQ4 tables, entity-clustered models, position and domain analyses,
quality/missingness diagnostics, worst-case missing-outcome bounds, and a
secondary Standard-versus-CoT analysis. It also reports retry rates and repeats
RQ4 after excluding conditions that needed a primary format retry.
See [analysis/README.md](analysis/README.md) and
[visualizations/README.md](visualizations/README.md) for output details.
