"""Figure 5: subjective best-resolution confidence and self-consistency.

Self-consistency is the modal-category share across three identical answer
calls. It is displayed beside, but never combined with, the model's post-hoc
estimate that its modal answer best resolves the supplied documents.

Usage:
    python visualizations/plot_agreement.py --csv results/conditions_<run>.csv
"""

import matplotlib.pyplot as plt

from analyze_agreement import condition_table
from common import (RATIO_ORDER, SURFACE, apply_style, arm_display_label,
                    load_results, make_arg_parser, pretty_model_label,
                    save_figure)

SUBJECTIVE_COLOR = "#eda100"
BEHAV_COLOR = "#2a78d6"


def main():
    args = make_arg_parser(__doc__.splitlines()[0]).parse_args()
    df = load_results(args.csv, args.strategy, args.exclude, args.arm)
    if "modal_category" not in df.columns and (
        "trial_index" not in df.columns or df["trial_index"].nunique() <= 1
    ):
        raise SystemExit("Needs a condition CSV or multi-trial raw run.")

    cond = condition_table(df)
    cond["best_resolution_norm"] = cond["best_resolution_confidence"] / 100

    ratios = [ratio for ratio in RATIO_ORDER if ratio in set(cond["ratio"])]
    models = sorted(cond["model_id"].unique())

    apply_style()
    fig, axes = plt.subplots(
        1, max(len(models), 1),
        figsize=(4.8 * max(len(models), 1), 4.5),
        sharey=True, squeeze=False,
    )
    width = 0.25

    for ax, model_id in zip(axes[0], models):
        group = cond[cond["model_id"] == model_id]
        xs = list(range(len(ratios)))
        subjective = [
            group[group["ratio"] == ratio]["best_resolution_norm"].mean()
            for ratio in ratios
        ]

        behavioral = [
            group[group["ratio"] == ratio]["self_consistency"].mean()
            for ratio in ratios
        ]
        ns = [len(group[group["ratio"] == ratio]) for ratio in ratios]

        ax.bar([x - width/2 for x in xs], subjective, width,
               color=SUBJECTIVE_COLOR,
               label="Subjective best-resolution confidence",
               edgecolor=SURFACE, linewidth=2)
        ax.bar([x + width/2 for x in xs], behavioral, width, color=BEHAV_COLOR,
               label="Self-consistency (separate diagnostic)",
               edgecolor=SURFACE, linewidth=2)
        for x, n in zip(xs, ns):
            ax.annotate(f"n={n}", (x, 0), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=7,
                        color="#898781")
        ax.set_title(pretty_model_label(model_id), fontsize=11)
        ax.set_xticks(xs)
        ax.set_xticklabels(ratios)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("Evidence ratio")

    axes[0][0].set_ylabel("Reported estimate / observed stability")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle(
        "Subjective best-resolution confidence and behavioral stability\n"
        f"{arm_display_label(df)}",
        fontsize=12,
    )
    save_figure(fig, args, "fig5_agreement.png")


if __name__ == "__main__":
    main()
