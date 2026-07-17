"""Figure 3 — self-reported confidence by evidence ratio and answer category.

The entities are fictional and have no external answer key. This figure
therefore treats 0-100 confidence as a model self-report and compares it across
MAJ/MIN/COM/FLAG outcomes. It never labels an answer factually correct or wrong.

Usage:
    python visualizations/plot_confidence.py
    python visualizations/plot_confidence.py --csv results/conditions_<run>.csv
"""

import matplotlib.pyplot as plt

from common import (CATEGORY_COLORS, RATIO_ORDER, SURFACE, apply_style,
                    load_results, make_arg_parser, save_figure)


GROUPS = [
    ("MAJ", "Majority claim", CATEGORY_COLORS["MAJ"]),
    ("MIN", "Minority claim", CATEGORY_COLORS["MIN"]),
    ("COM", "Compromise", CATEGORY_COLORS["COM"]),
    ("FLAG", "Conflict flagged", CATEGORY_COLORS["FLAG"]),
]


def main():
    args = make_arg_parser(__doc__.splitlines()[0]).parse_args()
    df = load_results(args.csv, args.strategy, args.exclude)
    scored = df[
        df["category"].isin([category for category, _, _ in GROUPS])
        & df["confidence"].notna()
    ].copy()
    if not len(scored):
        raise SystemExit("No categorized rows with usable confidence values.")

    # Current condition files store 0-100 self-reports. Historical raw files
    # may contain the original 1-5 field, which remains readable.
    scale_max = (
        100 if "modal_category" in df.columns
        else 5 if scored["confidence"].max() <= 5
        else 100
    )
    print(
        f"Detected confidence scale: 1-{scale_max}"
        if scale_max == 5 else "Detected confidence scale: 0-100"
    )

    models = sorted(scored["model_id"].unique())
    ratios = [ratio for ratio in RATIO_ORDER if ratio in set(scored["ratio"])]
    apply_style()
    fig, axes = plt.subplots(
        1, max(len(models), 1),
        figsize=(5.5 * max(len(models), 1), 4.5),
        sharey=True, squeeze=False,
    )
    width = 0.18
    offsets = [(-1.5 + index) * width for index in range(len(GROUPS))]

    for ax, model_id in zip(axes[0], models):
        model_rows = scored[scored["model_id"] == model_id]
        for offset, (category, label, color) in zip(offsets, GROUPS):
            xs, means, ns, present_ratios = [], [], [], []
            for ratio_index, ratio in enumerate(ratios):
                subset = model_rows[
                    (model_rows["ratio"] == ratio)
                    & (model_rows["category"] == category)
                ]
                if not len(subset):
                    continue
                xs.append(ratio_index + offset)
                means.append(subset["confidence"].mean())
                ns.append(len(subset))
                present_ratios.append(ratio)
            ax.bar(
                xs, means, width=width, color=color, label=label,
                edgecolor=SURFACE, linewidth=1.5,
            )
            for x, mean, n, ratio in zip(xs, means, ns, present_ratios):
                value = f"{mean:.1f}" if scale_max == 5 else f"{mean:.0f}"
                ax.annotate(
                    value, (x, mean), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7,
                    color=color,
                )
                print(
                    f"{model_id} ratio={ratio} category={category}: "
                    f"mean self-reported "
                    f"confidence {mean:.1f} (n={n})"
                )
        ax.set_xticks(range(len(ratios)))
        ax.set_xticklabels([
            ratio + ("\n(control)" if ratio == "4:0" else "")
            for ratio in ratios
        ])
        ax.set_ylim(0, scale_max * 1.08)
        ax.set_xlabel("Evidence ratio")
        ax.set_title(model_rows["model_label"].iloc[0], fontsize=11)

    axes[0][0].set_ylabel(
        f"Mean self-reported confidence ({'1-5' if scale_max == 5 else '0-100'})"
    )
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(labels),
        bbox_to_anchor=(0.5, -0.04), fontsize=9,
    )
    fig.suptitle(
        "Self-reported confidence by evidence ratio and response category",
        fontsize=12,
    )
    save_figure(fig, args, "fig3_confidence.png")


if __name__ == "__main__":
    main()
