"""Figure 5 — stated confidence vs. behavioral self-consistency.

Since logprobs are unavailable on all three models, confidence is measured
behaviorally from a multi-trial run: self-consistency = share of repeated trials
that land on the same answer category. This figure sets that behavioral measure
next to the model's own 1-5 self-report (normalized to 0-100%), one panel per
model, so the calibration gap is visible: a tall "stated" bar over a short
"self-consistency" bar means the model claims confidence its own answers don't
back up.

Needs a run with --trials > 1. Reuses condition_table() from analyze_agreement
so the numbers match that report exactly.

Usage:
    python visualizations/plot_agreement.py --csv results/multitrial_smoke.csv
"""

import matplotlib.pyplot as plt

from analyze_agreement import condition_table
from common import (RATIO_ORDER, SURFACE, apply_style, load_results,
                    make_arg_parser, pretty_model_label, save_figure)

STATED_COLOR = "#eda100"   # yellow — what the model SAYS
BEHAV_COLOR = "#2a78d6"    # blue — what the model DOES


def main():
    args = make_arg_parser(__doc__.splitlines()[0]).parse_args()
    df = load_results(args.csv, args.strategy, args.exclude)
    if "trial_index" not in df.columns or df["trial_index"].nunique() <= 1:
        raise SystemExit("Needs a multi-trial run (--trials N, N>1).")

    scale_max = 5 if df["confidence"].dropna().max() <= 5 else 100

    def norm(v):  # map self-report onto 0-1 to sit beside self-consistency
        return (v - 1) / 4 if scale_max == 5 else v / 100

    cond = condition_table(df)
    cond["selfreport_norm"] = cond["mean_selfreport"].map(norm)
    ratios = [r for r in RATIO_ORDER if r in set(cond["ratio"])]
    providers = sorted(cond["model_provider"].unique())

    apply_style()
    fig, axes = plt.subplots(1, max(len(providers), 1),
                             figsize=(4.6 * max(len(providers), 1), 4.5),
                             sharey=True, squeeze=False)
    width = 0.38

    for ax, prov in zip(axes[0], providers):
        g = cond[cond["model_provider"] == prov]
        xs = range(len(ratios))
        stated = [g[g["ratio"] == r]["selfreport_norm"].mean() for r in ratios]
        behav = [g[g["ratio"] == r]["self_consistency"].mean() for r in ratios]
        ns = [len(g[g["ratio"] == r]) for r in ratios]
        ax.bar([x - width / 2 for x in xs], stated, width, color=STATED_COLOR,
               label="Stated confidence (1-5, normalized)",
               edgecolor=SURFACE, linewidth=2)
        ax.bar([x + width / 2 for x in xs], behav, width, color=BEHAV_COLOR,
               label="Self-consistency (agreement across trials)",
               edgecolor=SURFACE, linewidth=2)
        for x, s, b, nn in zip(xs, stated, behav, ns):
            ax.annotate(f"{s:.0%}", (x - width / 2, s), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8,
                        color=STATED_COLOR)
            ax.annotate(f"{b:.0%}", (x + width / 2, b), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8,
                        color=BEHAV_COLOR)
            ax.annotate(f"n={nn}", (x, 0), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=7,
                        color="#898781")
        # model_id for this provider -> pretty label
        mid = df[df["model_provider"] == prov]["model_id"].iloc[0]
        ax.set_title(pretty_model_label(mid), fontsize=11)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(ratios)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("Evidence ratio")

    axes[0][0].set_ylabel("Confidence")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle("Stated confidence vs. behavioral self-consistency", fontsize=12)

    save_figure(fig, args, "fig5_agreement.png")


if __name__ == "__main__":
    main()
