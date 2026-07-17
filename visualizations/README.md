# Visualizations

Plot scripts for the Research Brief figures. Each script reads experiment CSVs
from `results/` (default: the most recent `conditions_*.csv`, then the most
recent legacy/raw `run_*.csv`), applies the scoring rubric, prints the
underlying numbers to stdout, and saves a 300-dpi PNG into
`visualizations/figures/`. A condition file is preferred because its unit of
analysis is one modal result per model/entity/ratio, not three correlated
primary calls.

| Script | Figure | Research use |
|---|---|---|
| `plot_majority_curve.py` | Majority-follow rate versus evidence ratio | RQ1 |
| `plot_flag_rate.py` | Conflict-abstention rate versus ratio | RQ1 / RQ4 |
| `plot_outcome_breakdown.py` | 100% stacked outcome breakdown | descriptive, all RQs |
| `plot_agreement.py` | Subjective best-resolution confidence beside self-consistency | RQ1 / stability |
| `plot_rq4_treatment_effect.py` | Distribution-request minus answer-only outcome effects | RQ4 |

Run from the repository root or this directory:

```bash
python visualizations/plot_majority_curve.py
python visualizations/plot_majority_curve.py --csv results/conditions_<run>.csv --arm distribution
python visualizations/plot_majority_curve.py --output /tmp/fig1.png
python visualizations/plot_rq4_treatment_effect.py --csv results/conditions_<run>.csv --strategy standard
```

The 38 distribution-request entities and 37 matched answer-only entities are
different experimental arms. If a CSV contains both, every descriptive plot
requires `--arm distribution` or `--arm answer-only`; it will stop instead of
silently pooling them. `--arm all` is an explicit opt-in intended for a
treatment comparison. The dedicated RQ4 figure uses both arms intentionally,
restricts itself to conflict ratios, and reports distribution-request minus
answer-only complete-case effects with 95% Newcombe intervals. Use the
analysis runner's `rq4_missing_outcome_bounds.csv` alongside it.

## Scoring

Each primary response is classified as:

- **MAJ:** matches the majority-supported value.
- **MIN:** matches the minority-supported value.
- **COM:** includes both values without refusing to resolve them.
- **FLAG:** strongly refuses to select a value or says the resolution is
  indeterminate.
- **OTHER:** matches neither supplied value without a strong refusal.
- **TIE/AMBIGUOUS:** the three calls have no unique modal category.
- **UNSCORED:** an API error occurred or no answer was extracted.

Mentioning that sources disagree is stored as a separate diagnostic and does
not by itself make an answer `FLAG`. The classifier exposes `category`,
`mentions_conflict`, and `abstained` so conflict awareness and behavioral
abstention are not conflated.

Numeric claim values (`$12`, `3.5%`, `240`) are compared as numbers, so
`12.00` or `3.5 percent` match, while `2400` never matches `240`. Other values
use whole-token string matching.

MAJ- and FLAG-rate plots exclude `TIE`, `AMBIGUOUS`, `UNSCORED`, and conditions
with fewer than three scored primary calls from their analytic denominators.
The outcome-breakdown plot intentionally displays all categories. Proportion
error bars are 95% Wilson intervals.

## Confidence and repeated calls

New condition CSVs store the rich inline distribution as
`mean_p_majority`, `mean_p_minority`, `mean_p_indeterminate`, and
`mean_p_sources_conflict`. The separate post-hoc field
`confidence_best_resolution` is the model's subjective 0-100 estimate that
its modal answer best resolves the supplied documents; it is not an externally
calibrated probability.

Post-hoc confidence and observed self-consistency remain separate measures.
Self-consistency uses all three planned calls as its denominator, so one valid
call plus two failures is 1/3 stability rather than 100%. The two measures are
shown side-by-side only for comparison in `plot_agreement.py`; they are never
combined into one score.

If a CSV contains both prompting strategies, pass `--strategy standard` or
`--strategy cot`. Pooling them would mix two experiments. Claude and DeepSeek
share `model_provider=openrouter`, so plots group by `model_id`; grouping by
provider would incorrectly combine two models.
