"""Figure 4 — full outcome breakdown per ratio and model.

100% stacked bars showing how every response was categorized (MAJ / MIN / COM /
FLAG / OTHER / TIE / AMBIGUOUS / UNSCORED) at each evidence ratio, one panel
per model. This is
the honest "everything" view behind Figures 1-3: it shows where the majority
answers came at the expense of flagging, and surfaces parse failures.

Usage:
    python visualizations/plot_outcome_breakdown.py
    python visualizations/plot_outcome_breakdown.py --csv results/pilot_gemini_gpt5mini.csv
"""

import matplotlib.pyplot as plt

from common import (CATEGORY_COLORS, CATEGORY_ORDER, RATIO_ORDER, SURFACE,
                    apply_style, arm_display_label, load_results,
                    make_arg_parser, save_figure)


def main():
    args = make_arg_parser(__doc__.splitlines()[0]).parse_args()
    df = load_results(args.csv, args.strategy, args.exclude, args.arm)

    models = sorted(df["model_id"].unique())
    apply_style()
    fig, axes = plt.subplots(1, max(len(models), 1),
                             figsize=(5.5 * max(len(models), 1), 4.5),
                             sharey=True, squeeze=False)

    drawn_cats = set()  # every category drawn in ANY panel (for a complete legend)
    for ax, model_id in zip(axes[0], models):
        group = df[df["model_id"] == model_id]
        bottoms = [0.0] * len(RATIO_ORDER)
        for cat in CATEGORY_ORDER:
            shares = []
            for ratio in RATIO_ORDER:
                sub = group[group["ratio"] == ratio]
                share = (sub["category"] == cat).mean() if len(sub) else 0.0
                shares.append(share)
            if not any(shares):
                continue
            drawn_cats.add(cat)
            ax.bar(range(len(RATIO_ORDER)), shares, bottom=bottoms, width=0.6,
                   color=CATEGORY_COLORS[cat], label=cat,
                   edgecolor=SURFACE, linewidth=2)
            for i, (share, bottom) in enumerate(zip(shares, bottoms)):
                if share >= 0.08:
                    ax.annotate(f"{share:.0%}",
                                (i, bottom + share / 2), ha="center",
                                va="center", fontsize=8, color=SURFACE)
            bottoms = [b + s for b, s in zip(bottoms, shares)]
        counts = group.groupby("ratio")["category"].value_counts()
        print(f"\n{model_id} outcome counts:\n{counts.to_string()}")

        ax.set_xticks(range(len(RATIO_ORDER)))
        ax.set_xticklabels([r + ("\n(control)" if r == "4:0" else "")
                            for r in RATIO_ORDER])
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("Evidence ratio")
        ax.set_title(group["model_label"].iloc[0] if len(group) else model_id,
                     fontsize=11)

    axes[0][0].set_ylabel("Share of responses")
    # Build the legend from every category drawn across ALL panels, in a fixed
    # order -- not from one subplot's handles, which would omit categories that
    # only appear in other panels (e.g. FLAG/COM absent from the first model).
    from matplotlib.patches import Patch
    legend_cats = [c for c in CATEGORY_ORDER if c in drawn_cats]
    handles = [Patch(facecolor=CATEGORY_COLORS[c], label=c) for c in legend_cats]
    fig.legend(handles, legend_cats, loc="upper center", ncol=len(legend_cats),
               bbox_to_anchor=(0.5, -0.04), fontsize=9)
    # reserve headroom so the two-line suptitle clears the per-panel titles
    fig.subplots_adjust(top=0.82)
    fig.suptitle(
        f"Response categories by evidence ratio\n{arm_display_label(df)}",
        fontsize=12, y=1.0,
    )

    save_figure(fig, args, "fig4_outcome_breakdown.png")


if __name__ == "__main__":
    main()
