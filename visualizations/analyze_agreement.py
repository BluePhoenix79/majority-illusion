"""Sampling-based self-consistency from three identical answer calls.

For every (model, entity, ratio), the harness measures how consistently the
model chooses the same answer category. Two distinct behavioral measures are
reported in [0,1]:

  self_consistency  - share of all planned calls that gave the MODAL answer.
                      1.0 means the same scored category on every call; failed
                      or unscored calls remain in the denominator.
  majority_follow   - share of trials categorized MAJ. "How reliably does it
                      follow the document majority here?"

These are kept SEPARATE from the model's subjective 0-100 post-hoc estimate
that its modal answer is the best resolution of the supplied documents. Rich
inline distribution fields are reported separately as well.

Accepts either the new condition-level CSV directly or a legacy raw run with
--trials > 1. Prints a per-condition table and, per model x ratio, the mean of
each measure alongside the subjective best-resolution estimate.

Usage:
    python visualizations/analyze_agreement.py --csv results/multitrial_smoke.csv
"""

import json

import pandas as pd

from common import (ANALYTIC_CATEGORIES, RATIO_ORDER, arm_display_label,
                    load_results, make_arg_parser, pretty_model_label)


def condition_table(df):
    """One row per (model, entity, ratio): trial count + the two measures.

    self_consistency is measured over the answer CATEGORY (MAJ/MIN/COM/FLAG),
    not the raw answer string, and uses all planned calls as its denominator.
    A verbose model that says "$25, though one source notes $35" with slightly
    different wording each trial is giving the *same* answer, while a failed
    call still lowers observed stability.
    """
    if "modal_category" in df.columns and "self_consistency" in df.columns:
        cond = df.copy()
        cond["n_trials"] = pd.to_numeric(
            cond.get("n_samples", 3), errors="coerce"
        )
        stability_column = (
            "self_consistency_all_samples"
            if "self_consistency_all_samples" in cond.columns
            else "self_consistency"
        )
        cond["self_consistency"] = pd.to_numeric(
            cond[stability_column], errors="coerce"
        )
        def majority_share(serialized):
            try:
                categories = json.loads(serialized)
            except (TypeError, json.JSONDecodeError):
                return float("nan")
            scored = [
                category for category in categories
                if category in ANALYTIC_CATEGORIES
            ]
            return (
                sum(category == "MAJ" for category in scored) / len(scored)
                if scored else float("nan")
            )

        cond["majority_follow"] = cond["response_categories"].map(majority_share)
        for column in (
            "mean_p_majority", "mean_p_minority", "mean_p_indeterminate",
            "mean_p_sources_conflict",
        ):
            cond[column] = pd.to_numeric(
                cond.get(column, float("nan")), errors="coerce"
            )
        confidence_source = next(
            (cond[column] for column in (
                "confidence_best_resolution",
                "subjective_best_resolution_confidence",
                "posthoc_subjective_confidence",
                "posthoc_probability",
            ) if column in cond.columns),
            pd.Series(float("nan"), index=cond.index),
        )
        cond["best_resolution_confidence"] = pd.to_numeric(
            confidence_source, errors="coerce"
        )
        return cond

    rows = []
    for (model_id, prov, eid, ratio), g in df.groupby(
        ["model_id", "model_provider", "entity_id", "ratio"]
    ):
        def numeric_mean(column):
            if column not in g.columns:
                return float("nan")
            return pd.to_numeric(g[column], errors="coerce").mean()

        n = len(g)
        cats = g["category"][g["category"].isin(ANALYTIC_CATEGORIES)]
        if len(cats):
            modal_cat = cats.value_counts().index[0]
            # Use every planned call as the denominator. A single valid call
            # plus two failures is 1/3 stability, not perfect stability.
            modal_share = cats.value_counts().iloc[0] / n
        else:
            modal_cat, modal_share = "", float("nan")
        maj_share = (cats == "MAJ").mean() if len(cats) else float("nan")
        rows.append({
            "model_id": model_id, "model_provider": prov,
            "entity_id": eid, "ratio": ratio,
            "n_trials": n, "self_consistency": modal_share,
            "majority_follow": maj_share, "modal_category": modal_cat,
            "mean_p_majority": numeric_mean("p_majority"),
            "mean_p_minority": numeric_mean("p_minority"),
            "mean_p_indeterminate": numeric_mean("p_indeterminate"),
            "mean_p_sources_conflict": numeric_mean("p_sources_conflict"),
            "best_resolution_confidence": g["confidence"].mean(),
        })
    return pd.DataFrame(rows)


def main():
    args = make_arg_parser(__doc__.splitlines()[0]).parse_args()
    df = load_results(args.csv, args.strategy, args.exclude, args.arm)
    print(f"Elicitation arm: {arm_display_label(df)}")

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
                  f"p-majority~{r.mean_p_majority:.1f}, "
                  f"p-indeterminate~{r.mean_p_indeterminate:.1f}, "
                  f"p-conflict~{r.mean_p_sources_conflict:.1f}, "
                  f"best-resolution~{r.best_resolution_confidence:.1f}")

    print("\n=== mean by model x ratio: stability and subjective reports ===")
    print(f"{'model':22} {'ratio':>5} {'self-consist':>12} "
          f"{'maj-follow':>11} {'p-majority':>11} {'p-indet':>9} "
          f"{'p-conflict':>11} {'best-res':>9}")
    for model_id, g in cond.groupby("model_id"):
        label = pretty_model_label(model_id)
        for ratio in ratios_present:
            sub = g[g["ratio"] == ratio]
            if not len(sub):
                continue
            print(f"{label[:22]:22} {ratio:>5} "
                  f"{sub.self_consistency.mean():>11.0%} "
                  f"{sub.majority_follow.mean():>10.0%} "
                  f"{sub.mean_p_majority.mean():>10.1f} "
                  f"{sub.mean_p_indeterminate.mean():>8.1f} "
                  f"{sub.mean_p_sources_conflict.mean():>10.1f} "
                  f"{sub.best_resolution_confidence.mean():>8.1f}")
    print("\nRead these as distinct quantities: self-consistency measures "
          "behavioral stability; inline distributions describe the model's "
          "reported uncertainty; post-hoc confidence is a subjective estimate "
          "that the modal answer best resolves the supplied documents.")


if __name__ == "__main__":
    main()
