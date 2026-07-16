# Visualizations

Plot scripts for the Research Brief figures. Each script reads experiment CSVs
from `results/` (default: the most recent `run_*.csv`, falling back to the real
pilot CSV), applies the Day 1 scoring rubric, prints the underlying numbers to
stdout, and saves a 300-dpi PNG into `visualizations/figures/`.

| Script | Figure | Hypothesis |
|---|---|---|
| `plot_majority_curve.py` | Majority-follow rate vs. evidence ratio (the majority-illusion curve) | H1 |
| `plot_flag_rate.py` | Conflict-flag rate vs. ratio (conflict blindness) | H2 |
| `plot_confidence.py` | Mean confidence by ratio, majority vs. minority answers | H3 |
| `plot_outcome_breakdown.py` | 100% stacked MAJ/MIN/COM/FLAG breakdown per ratio and model | all / H4 |

Run from the repo root or this directory:

```bash
python visualizations/plot_majority_curve.py                 # latest full run
python visualizations/plot_majority_curve.py --csv results/pilot_gemini_gpt5mini.csv
python visualizations/plot_majority_curve.py --output /tmp/fig1.png
```

Scoring (in `common.py`, per the research plan): each response is classified as
**MAJ** (matches majority claim), **MIN** (matches minority), **COM** (mentions
both without refusing), **FLAG** (notes the conflict / refuses to pick), plus
**OTHER** (matches neither) and **UNSCORED** (API error or unparseable JSON).
Numeric claim values (`$12`, `3.5%`, `240`) are compared as numbers, so
`12.00`/`3.5 percent` match but `2400` never matches `240`; other values use
whole-token string matching. Proportion error bars are 95% Wilson intervals.

If a CSV contains both prompting strategies (standard + CoT), pass
`--strategy standard` or `--strategy cot` — pooling them mixes two experiments.
The confidence figure auto-detects the elicitation scale (0-100 in pre-Jul-15
CSVs, 1-5 after); don't mix CSVs from both sides of that change in one figure.

The automatic FLAG/COM keyword matching is a first pass — the plan calls for
human double-scoring of a 15% sample (Cohen's kappa), so spot-check the
`category` assignments against `raw_response` before quoting numbers in the
brief.
