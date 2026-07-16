"""Sampling-based ("self-consistency") confidence from multi-trial runs.

Logprobs are unavailable on all three models (GPT-5 mini and Gemini 3.5 Flash
reject the parameter; OpenRouter returns none for DeepSeek). So instead of a
token-level probability we estimate confidence *behaviorally*: run each
(model, entity, ratio) condition several times and measure how consistent the
answers are. Two distinct measures, both in [0,1]:

  self_consistency  - share of trials that gave the MODAL answer. "How sure is
                      the model of *an* answer?" 1.0 = identical every trial.
  majority_follow   - share of trials categorized MAJ. "How reliably does it
                      follow the document majority here?"

These are kept SEPARATE from the model's self-reported 1-5 confidence
(`parsed_confidence`). The payoff is comparing them: if verbalized confidence
is high while self-consistency is low, the model's stated certainty is not
backed by its own behavior.

Requires a run with --trials > 1 (the trial_index column). Prints a per-
condition table and, per model x ratio, the mean of each measure alongside the
mean self-reported confidence for side-by-side comparison.

Usage:
    python visualizations/analyze_agreement.py --csv results/multitrial_smoke.csv
"""

import pandas as pd

from common import RATIO_ORDER, load_results, make_arg_parser, pretty_model_label


def _norm_ans(s):
    return str(s).lower().replace(",", "").strip().strip('."')


def condition_table(df):
    """One row per (model, entity, ratio): trial count + the two measures.

    self_consistency is measured over the answer CATEGORY (MAJ/MIN/COM/FLAG),
    not the raw answer string: a verbose model that says "$25, though one source
    notes $35" with slightly different wording each trial is giving the *same*
    answer, and raw-string matching would wrongly score it as inconsistent.
    """
    rows = []
    for (prov, eid, ratio), g in df.groupby(["model_provider", "entity_id", "ratio"]):
        n = len(g)
        cats = g["category"][g["category"] != "UNSCORED"]
        if len(cats):
            modal_cat = cats.value_counts().index[0]
            modal_share = cats.value_counts().iloc[0] / len(cats)
        else:
            modal_cat, modal_share = "", float("nan")
        maj_share = (g["category"] == "MAJ").mean()
        rows.append({
            "model_provider": prov, "entity_id": eid, "ratio": ratio,
            "n_trials": n, "self_consistency": modal_share,
            "majority_follow": maj_share, "modal_category": modal_cat,
            "mean_selfreport": g["confidence"].mean(),
        })
    return pd.DataFrame(rows)


def main():
    args = make_arg_parser(__doc__.splitlines()[0]).parse_args()
    df = load_results(args.csv, args.strategy, args.exclude)

    if "trial_index" not in df.columns or df["trial_index"].nunique() <= 1:
        raise SystemExit(
            "This analysis needs multiple trials per condition. Re-run the "
            "harness with --trials N (N>1) and point --csv at that output.")

    cond = condition_table(df)
    ratios_present = [r for r in RATIO_ORDER if r in set(cond["ratio"])]

    print("\n=== per-condition self-consistency (share of trials on modal answer) ===")
    for prov, g in cond.groupby("model_provider"):
        print(f"\n{pretty_model_label(g_model_id(df, prov))}")
        for _, r in g.sort_values(["ratio", "entity_id"]).iterrows():
            print(f"  {r.entity_id} {r.ratio:>4}: "
                  f"{int(r.n_trials)} trials, self-consistency "
                  f"{r.self_consistency:.0%}, majority-follow "
                  f"{r.majority_follow:.0%}, modal-cat={r.modal_category}, "
                  f"self-report~{r.mean_selfreport:.1f}")

    print("\n=== mean by model x ratio: behavioral vs. self-reported ===")
    print(f"{'model':22} {'ratio':>5} {'self-consist':>12} "
          f"{'maj-follow':>11} {'self-report':>12}")
    scale_hint = "(1-5)" if df["confidence"].dropna().max() <= 5 else "(0-100)"
    for prov, g in cond.groupby("model_provider"):
        label = pretty_model_label(g_model_id(df, prov))
        for ratio in ratios_present:
            sub = g[g["ratio"] == ratio]
            if not len(sub):
                continue
            print(f"{label[:22]:22} {ratio:>5} "
                  f"{sub.self_consistency.mean():>11.0%} "
                  f"{sub.majority_follow.mean():>10.0%} "
                  f"{sub.mean_selfreport.mean():>8.1f} {scale_hint}")
    print("\nRead: where self-consistency << self-report, the model's stated "
          "confidence is not backed by its own answer stability.")


def g_model_id(df, provider):
    ids = df[df["model_provider"] == provider]["model_id"]
    return ids.iloc[0] if len(ids) else provider


if __name__ == "__main__":
    main()
