"""Figure 3: reported probability distribution across evidence ratios (RQ1).

For the structured-elicitation arm, each answer reports subjective probabilities
p_majority + p_minority + p_indeterminate (summing to 100) plus an independent
p_sources_conflict. This figure shows how that distribution shifts as the
majority grows -- the graded form of the majority illusion. p_sources_conflict is
overlaid as a line because it is independent of the three-way split (the model
can commit to a claim while still signalling that the sources disagree).

Only the structured-elicitation arm reports distributions, so this figure uses
that arm regardless of --arm.

Usage:
    python visualizations/plot_confidence_distribution.py --csv results/conditions_<run>.csv
"""

import matplotlib.pyplot as plt

from common import (CONFLICT_RATIOS, MUTED, SURFACE, apply_style,
                    arm_display_label, load_results, make_arg_parser,
                    pretty_model_label, save_figure)

P_MAJ_COLOR = "#2a78d6"       # blue
P_IND_COLOR = "#c3c2b7"       # gray
P_MIN_COLOR = "#eb6834"       # orange
P_CONFLICT_COLOR = "#4a3aa7"  # violet


def main():
    args = make_arg_parser(__doc__.splitlines()[0]).parse_args()
    # Distributions exist only in the structured arm; select it explicitly so
    # the figure never depends on the caller passing --arm.
    df = load_results(args.csv, args.strategy, args.exclude, "distribution")
    df = df[df["mean_p_majority"].notna()]
    if not len(df):
        raise SystemExit("No rows with a reported probability distribution "
                         "(needs the structured-elicitation arm).")

    ratios = [r for r in CONFLICT_RATIOS if r in set(df["ratio"])]
    models = sorted(df["model_id"].unique())

    apply_style()
    fig, axes = plt.subplots(1, max(len(models), 1),
                             figsize=(4.8 * max(len(models), 1), 4.5),
                             sharey=True, squeeze=False)

    for ax, model_id in zip(axes[0], models):
        g = df[df["model_id"] == model_id]
        xs = list(range(len(ratios)))
        p_maj = [g[g["ratio"] == r]["mean_p_majority"].mean() for r in ratios]
        p_ind = [g[g["ratio"] == r]["mean_p_indeterminate"].mean() for r in ratios]
        p_min = [g[g["ratio"] == r]["mean_p_minority"].mean() for r in ratios]
        p_conf = [g[g["ratio"] == r]["mean_p_sources_conflict"].mean()
                  for r in ratios]
        ns = [len(g[g["ratio"] == r]) for r in ratios]

        # Stacked three-way split (sums to ~100).
        ax.bar(xs, p_maj, width=0.6, color=P_MAJ_COLOR,
               label="P(majority claim)", edgecolor=SURFACE, linewidth=2)
        ax.bar(xs, p_ind, width=0.6, bottom=p_maj, color=P_IND_COLOR,
               label="P(indeterminate)", edgecolor=SURFACE, linewidth=2)
        base2 = [a + b for a, b in zip(p_maj, p_ind)]
        ax.bar(xs, p_min, width=0.6, bottom=base2, color=P_MIN_COLOR,
               label="P(minority claim)", edgecolor=SURFACE, linewidth=2)
        # Independent conflict estimate as an overlaid line.
        ax.plot(xs, p_conf, color=P_CONFLICT_COLOR, marker="o", markersize=6,
                linewidth=2, label="P(sources conflict) - independent")

        for x, n in zip(xs, ns):
            ax.annotate(f"n={n}", (x, 0), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=7,
                        color=MUTED)
        ax.set_title(pretty_model_label(model_id), fontsize=11)
        ax.set_xticks(xs)
        ax.set_xticklabels(ratios)
        ax.set_ylim(0, 108)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Evidence ratio")

    axes[0][0].set_ylabel("Mean reported probability (0-100)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=8)
    # reserve headroom so the two-line suptitle clears the per-panel titles
    fig.subplots_adjust(top=0.80)
    fig.suptitle("Reported probability distribution across evidence ratios\n"
                 f"{arm_display_label(df)}", fontsize=12, y=1.0)
    save_figure(fig, args, "fig3_confidence_distribution.png")


if __name__ == "__main__":
    main()
