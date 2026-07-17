"""Sampling-based self-consistency from three identical answer calls.

For every (model, entity, ratio), the harness measures how consistently the
model chooses the same answer category. Two distinct behavioral measures are
reported in [0,1]:

  self_consistency  - share of trials that gave the MODAL answer. "How sure is
                      the model of *an* answer?" 1.0 = identical every trial.
  majority_follow   - share of trials categorized MAJ. "How reliably does it
                      follow the document majority here?"

These are kept SEPARATE from the raw 0-100 post-hoc self-report.

Accepts either the new condition-level CSV directly or a legacy raw run with
--trials > 1. Prints a per-condition table and, per model x ratio, the mean of
each measure alongside raw confidence.

Usage:
    python visualizations/analyze_agreement.py --csv results/multitrial_smoke.csv
"""

import json

import pandas as pd

from common import RATIO_ORDER, load_results, make_arg_parser, pretty_model_label


def condition_table(df):
    """One row per (model, entity, ratio): trial count + the two measures.

    self_consistency is measured over the answer CATEGORY (MAJ/MIN/COM/FLAG),
    not the raw answer string: a verbose model that says "$25, though one source
    notes $35" with slightly different wording each trial is giving the *same*
    answer, and raw-string matching would wrongly score it as inconsistent.
    """
    if "modal_category" in df.columns and "self_consistency" in df.columns:
        cond = df.copy()
        cond["n_trials"] = pd.to_numeric(cond["n_samples"], errors="coerce")
        cond["self_consistency"] = pd.to_numeric(
            cond["self_consistency"], errors="coerce"
        )
        def majority_share(serialized):
            try:
                categories = json.loads(serialized)
            except (TypeError, json.JSONDecodeError):
                return float("nan")
            return (
                sum(category == "MAJ" for category in categories) / len(categories)
                if categories else float("nan")
            )

        cond["majority_follow"] = cond["response_categories"].map(majority_share)
        cond["mean_inline_selfreport"] = pd.to_numeric(
            cond.get("mean_inline_probability", ""), errors="coerce"
        )
        cond["raw_posthoc"] = pd.to_numeric(
            cond.get("posthoc_probability", ""), errors="coerce"
        )
        return cond

    rows = []
    for (model_id, prov, eid, ratio), g in df.groupby(
        ["model_id", "model_provider", "entity_id", "ratio"]
    ):
        n = len(g)
        cats = g["category"][g["category"] != "UNSCORED"]
        if len(cats):
            modal_cat = cats.value_counts().index[0]
            modal_share = cats.value_counts().iloc[0] / len(cats)
        else:
            modal_cat, modal_share = "", float("nan")
        maj_share = (g["category"] == "MAJ").mean()
        rows.append({
            "model_id": model_id, "model_provider": prov,
            "entity_id": eid, "ratio": ratio,
            "n_trials": n, "self_consistency": modal_share,
            "majority_follow": maj_share, "modal_category": modal_cat,
            "mean_inline_selfreport": g["confidence"].mean(),
            "raw_posthoc": float("nan"),
        })
    return pd.DataFrame(rows)


def main():
    args = make_arg_parser(__doc__.splitlines()[0]).parse_args()
    df = load_results(args.csv, args.strategy, args.exclude)

    if "modal_category" not in df.columns and (
        "trial_index" not in df.columns or df["trial_index"].nunique() <= 1
    ):
        raise SystemExit(
            "This analysis needs multiple trials per condition. Re-run the "
            "harness with --trials N (N>1) and point --csv at that output.")

    cond = condition_table(df)
    ratios_present = [r for r in RATIO_ORDER if r in set(cond["ratio"])]

    print("\n=== per-condition self-consistency (share of trials on modal answer) ===")
    for model_id, g in cond.groupby("model_id"):
        print(f"\n{pretty_model_label(model_id)}")
        for _, r in g.sort_values(["ratio", "entity_id"]).iterrows():
            print(f"  {r.entity_id} {r.ratio:>4}: "
                  f"{int(r.n_trials)} trials, self-consistency "
                  f"{r.self_consistency:.0%}, majority-follow "
                  f"{r.majority_follow:.0%}, modal-cat={r.modal_category}, "
                  f"inline~{r.mean_inline_selfreport:.1f}, "
                  f"posthoc~{r.raw_posthoc:.1f}")

    print("\n=== mean by model x ratio: behavioral vs. self-reported ===")
    print(f"{'model':22} {'ratio':>5} {'self-consist':>12} "
          f"{'maj-follow':>11} {'inline':>9} {'posthoc':>9}")
    for model_id, g in cond.groupby("model_id"):
        label = pretty_model_label(model_id)
        for ratio in ratios_present:
            sub = g[g["ratio"] == ratio]
            if not len(sub):
                continue
            print(f"{label[:22]:22} {ratio:>5} "
                  f"{sub.self_consistency.mean():>11.0%} "
                  f"{sub.majority_follow.mean():>10.0%} "
                  f"{sub.mean_inline_selfreport.mean():>8.1f} "
                  f"{sub.raw_posthoc.mean():>8.1f}")
    print("\nRead these as distinct quantities: agreement measures stability; "
          "post-hoc confidence estimates correctness.")


if __name__ == "__main__":
    main()
