# Majority Illusion in RAG

This repository studies how language models resolve contradictory evidence in
retrieval-augmented-generation-style prompts. The experiment supplies models
with fictional documents that disagree about an entity, varies the number of
documents supporting each claim, and measures whether the models follow the
document majority, select the minority claim, combine the claims, or abstain.

The project simulates the **generation stage** of RAG by placing controlled
documents directly in the prompt. It does not implement a vector database or a
retrieval algorithm.

## Current repository status

- The final dataset contains **75 fictional entities**: 30 banking entities and
  45 general-corporate entities.
- The **Standard** run is complete and has passed the strict final validation
  gate: 1,350 conditions, 4,050 valid terminal primary samples, all six ratios,
  and all three model slots.
- The validated Standard condition data is in
  [`results/conditions_final_standard_salvaged.csv`](results/conditions_final_standard_salvaged.csv).
  The complete call-level audit log is in
  [`results/run_final_standard_salvaged_raw.csv`](results/run_final_standard_salvaged_raw.csv),
  and [`results/salvage_standard_manifest.json`](results/salvage_standard_manifest.json)
  records the provenance of the recovered conditions.
- Standard-run tables and model outputs are in
  [`results/analysis_final_standard_salvaged/`](results/analysis_final_standard_salvaged/).
- A complete final **CoT** run is not present. Standard-versus-CoT analysis is
  therefore pending and must not be inferred from the current results.
- [`Results_and_Discussion.pdf`](Results_and_Discussion.pdf) contains the current
  results discussion. Development history and superseded decisions remain in
  [`UPDATES.md`](UPDATES.md); the newest entries take precedence over older ones.

## Research design

### Evidence ratios

A ratio is the number of documents supporting the designated majority claim
versus the number supporting the competing minority claim.

| Ratio | Majority share | Documents | Role |
|---|---:|---:|---|
| `2:2` | 50% | 4 | tied evidence |
| `3:2` | 60% | 5 | small majority |
| `2:1` | 66.7% | 3 | moderate majority |
| `3:1` | 75% | 4 | strong majority |
| `4:1` | 80% | 5 | stronger majority |
| `4:0` | 100% | 4 | unanimous, no-conflict control |

The documents are stored under each entity's `documents` field in
[`data/entities.json`](data/entities.json). They are generated deterministically
by [`data/generate_dataset.py`](data/generate_dataset.py) from a fixed seed.
Documents use different fictional source styles, but intentionally repeat one
of the two competing claims at the requested ratio.

### Models

The final roster uses three fixed model slots:

- **Gemini 3.5 Flash**, called through the native Google GenAI SDK.
- **DeepSeek V4 Flash**, called through OpenRouter.
- **Claude Haiku 4.5**, called through a separate OpenRouter slot.

DeepSeek and Claude share an API gateway but not an experimental model slot.
The harness records the slot, requested model ID, and returned model ID so the
two models cannot be silently swapped or pooled.

### Conditions and repeated calls

The unit of analysis is one:

```text
entity x ratio x model x strategy x document layout
```

For each condition, the harness:

1. Sends the same primary prompt three times at temperature 1.0.
2. Classifies the three responses and records their modal outcome.
3. Measures self-consistency across the three responses.
4. When there is a unique modal answer, makes a separate post-hoc 0-100
   judgment asking whether that answer is the best resolution of the supplied
   documents.

The three calls are repeated measurements of one condition, not three
independent observations. A malformed response may receive up to two
byte-identical format retries. Every physical attempt remains in the raw log,
while the condition file contains one summarized condition record.

### Response categories

- `MAJ`: selects the majority-supported value.
- `MIN`: selects the minority-supported value.
- `COM`: includes both supplied values without strongly refusing to resolve
  them.
- `FLAG`: treats the answer as indeterminate or strongly refuses to choose.
- `OTHER`: matches neither supplied value without a strong refusal.
- `TIE` / `AMBIGUOUS`: the repeated calls do not produce a unique modal result.
- `UNSCORED`: an API error occurred or no valid answer could be extracted.

Mentioning that sources conflict is stored separately from abstaining. A model
can notice a conflict and still select one of the claims.

### Structured-elicitation experiment

On conflict ratios, 38 entities are assigned to a structured-distribution arm
and 37 to a matched answer-only arm. The structured arm reports:

- `p_claim_a`
- `p_claim_b`
- `p_indeterminate`
- `p_sources_conflict`

Claim A and Claim B are counterbalanced, then stored canonically as majority and
minority probabilities. Every `4:0` prompt remains answer-only so the prompt
never introduces a minority value that is absent from its documents.

This arm comparison tests whether asking for an uncertainty distribution
changes the answer itself; the distribution is not assumed to be a passive
measurement.

## Interpretation safeguards

- `majority_value` and `minority_value` are experimental labels, not truth
  labels. `MAJ` does not automatically mean factually correct, and `MIN` does
  not automatically mean wrong.
- The entities are fictional, so the experiment tests behavior under supplied
  context rather than override of known real-world facts.
- Post-hoc confidence is a subjective judgment about the best resolution of the
  supplied documents. It is not calibrated factual correctness.
- Document count varies across ratios, so ratio and context length are not fully
  separated.
- Banking-versus-general comparisons describe these synthetic stimulus
  families; they do not establish a causal effect of real financial risk.
- The current repository contains final Standard evidence only. Any
  Standard-versus-CoT claim requires a separately completed and validated CoT
  run.

No truth-label calibration, Platt scaling, Brier/ECE analysis, Cohen's-kappa
workflow, or standalone context-length research question is part of the final
protocol.

## Repository map

```text
data/
  generate_dataset.py       deterministic dataset generator
  entities.json             75 entities and all ratio-specific documents
harness/
  run_experiment.py         collection, retry, scoring, and completion gates
  salvage_standard_run.py   provenance-preserving Standard-run recovery
  token_report.py           token and supported cost accounting
analysis/
  run_all_analyses.py       strict RQ1-RQ4 condition-level analysis
  README.md                 analysis rules and output definitions
visualizations/
  common.py                 shared loading and scoring helpers
  plot_*.py                 figure generators
  figures/                  generated PNG figures
results/
  conditions_final_standard_salvaged.csv   validated analysis-unit data
  run_final_standard_salvaged_raw.csv      complete physical call log
  salvage_standard_manifest.json           recovery provenance
  analysis_final_standard_salvaged/        generated Standard analysis tables
tests/                      offline protocol and analysis tests
UPDATES.md                  append-only development history
Results_and_Discussion.pdf  current results write-up
```

## Setup

Use Python 3 from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The live collector reads credentials from environment variables or a local
`.env` file. `.env` and common Google service-account filenames are ignored by
Git and must never be committed.

Required live credentials depend on the Gemini authentication mode:

- AI Studio: `GEMINI_API_KEY`
- Vertex Express: `GEMINI_USE_VERTEX=1` and `GEMINI_API_KEY`
- Vertex project authentication: `GEMINI_USE_VERTEX=1`,
  `GEMINI_VERTEX_PROJECT`, optional `GEMINI_VERTEX_LOCATION` (default `global`),
  and Application Default Credentials or `GOOGLE_APPLICATION_CREDENTIALS`
- OpenRouter models: `OPENROUTER_API_KEY`

See the header of [`harness/run_experiment.py`](harness/run_experiment.py) for
the complete authentication behavior.

## Validate without API calls

After installing the dependencies, run the primary offline suites:

```bash
python3 -m unittest \
  tests.test_new_approach \
  visualizations.test_common \
  analysis.tests.test_run_all_analyses
```

Then run a small mock collection:

```bash
python3 harness/run_experiment.py --mock \
  --entity-ids E003 E001 \
  --ratios 3:1 4:0 \
  --models gemini deepseek claude
```

`E003` exercises the structured-distribution arm, while `E001` exercises the
answer-only arm. Including `4:0` also checks the unanimous-control safeguard.
Mock data validates the pipeline but is not scientific evidence.

## Collect data

The final Standard data is already present. Do not overwrite it. A fresh full
run has 1,350 conditions:

```text
75 entities x 6 ratios x 3 models = 1,350 conditions
```

It plans three primary calls per condition plus at most one post-hoc call, or at
most 5,400 scientific calls per strategy. Format retries can increase the
number of physical API attempts.

To collect the still-pending CoT strategy into new files:

```bash
python3 harness/run_experiment.py --strategy cot \
  --output results/run_final_cot_raw.csv \
  --condition-output results/conditions_final_cot.csv
```

Run a small live pilot and confirm credentials, model IDs, output formatting,
and budget before starting a full collection. The collector exits nonzero if
the final factorial is incomplete or a condition lacks the required primary
samples or valid post-hoc disposition.

## Analyze

Reproduce the existing Standard-only analysis:

```bash
python3 analysis/run_all_analyses.py \
  results/conditions_final_standard_salvaged.csv \
  --output-dir results/analysis_final_standard_reproduced
```

After a complete CoT run exists, analyze the two strategies together:

```bash
python3 analysis/run_all_analyses.py \
  results/conditions_final_standard_salvaged.csv \
  results/conditions_final_cot.csv \
  --output-dir results/analysis_standard_vs_cot
```

The strict loader checks the protocol version, dataset hash, model slots,
factorial completeness, treatment assignment, repeated-sample completeness,
post-hoc disposition, and stored probability distributions. See
[`analysis/README.md`](analysis/README.md) for every output table and its
interpretation.

The current Standard analysis report logs statistical fitting warnings and
explicitly notes that the secondary Standard-versus-CoT comparison was not run.
Do not discard those warnings when interpreting model coefficients.

## Generate figures

Condition-level CSVs should be used for figures because one row represents one
experimental condition. The two elicitation arms must be selected explicitly
for descriptive plots:

```bash
python3 visualizations/plot_majority_curve.py \
  --csv results/conditions_final_standard_salvaged.csv \
  --strategy standard \
  --arm answer-only

python3 visualizations/plot_majority_curve.py \
  --csv results/conditions_final_standard_salvaged.csv \
  --strategy standard \
  --arm distribution
```

Use the dedicated RQ4 scripts when comparing the two arms. See
[`visualizations/README.md`](visualizations/README.md) for scoring rules,
denominators, confidence intervals, and all figure commands.

## Dataset regeneration warning

Running the generator is deterministic:

```bash
python3 data/generate_dataset.py
```

However, changing the generator, seed, ratios, names, attributes, or source
templates changes the dataset hash and can invalidate compatibility with the
collected results. Do not change or regenerate the final dataset as part of an
analysis-only workflow.

## License

The code is available under the [MIT License](LICENSE).
