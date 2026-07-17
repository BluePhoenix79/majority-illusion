"""RQ4: effect of requesting a rich probability distribution on outcomes.

The figure uses complete, condition-level modal outcomes from conflict ratios
only. Each point is the percentage-point difference between the structured
conflict-aware uncertainty-elicitation arm and the matched answer-only arm.
Error bars are 95% Newcombe hybrid-score intervals for two independent
proportions. Missing-outcome bounds are produced by the analysis runner rather
than this descriptive figure.

Usage:
    python visualizations/plot_rq4_treatment_effect.py \
        --csv results/conditions_<run>.csv --strategy standard
"""

import math

import matplotlib.pyplot as plt

from common import (
    ARM_COLUMN,
    CONFLICT_RATIOS,
    SURFACE,
    analytic_rate_rows,
    apply_style,
    load_results,
    make_arg_parser,
    model_color,
    newcombe_difference_ci,
    pretty_model_label,
    save_figure,
)


OUTCOMES = (
    ("FLAG", "Primary: conflict abstention (FLAG)", {"FLAG"}),
    ("MAJ", "Secondary: majority selection (MAJ)", {"MAJ"}),
    ("COM", "Secondary: compromise (COM)", {"COM"}),
    ("FLAG|COM", "Secondary: caution (FLAG or COM)", {"FLAG", "COM"}),
)


def main():
    args = make_arg_parser(__doc__.splitlines()[0]).parse_args()
    if args.arm not in (None, "all"):
        raise SystemExit(
            "The RQ4 treatment-effect figure requires both arms; omit --arm "
            "or pass --arm all."
        )

    # ``all`` is intentional here: unlike the descriptive plots, this script
    # explicitly contrasts rather than pools the two arms.
    df = load_results(
        args.csv, args.strategy, args.exclude, arm="all"
    )
    if "modal_category" not in df.columns:
        raise SystemExit(
            "RQ4 requires a condition-level CSV with one modal outcome per "
            "model/entity/ratio; raw repeated-call rows are not independent."
        )

    scored = analytic_rate_rows(df)
    scored = scored[scored["ratio"].isin(CONFLICT_RATIOS)].copy()
    present_arms = set(scored[ARM_COLUMN].astype(int))
    if present_arms != {0, 1}:
        raise SystemExit(
            "RQ4 needs both distribution-request (1) and answer-only (0) "
            f"conditions; found {sorted(present_arms)}."
        )

    models = sorted(scored["model_id"].unique())
    if not models:
        raise SystemExit("No scored conflict-ratio conditions remain.")

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    offsets = {
        model_id: (index - (len(models) - 1) / 2) * 0.055
        for index, model_id in enumerate(models)
    }

    for ax, (_, title, positive_categories) in zip(axes.flat, OUTCOMES):
        for model_id in models:
            model = scored[scored["model_id"].eq(model_id)]
            xs, differences, errors_lo, errors_hi = [], [], [], []
            for ratio_index, ratio in enumerate(CONFLICT_RATIOS):
                cell = model[model["ratio"].eq(ratio)]
                treatment = cell[cell[ARM_COLUMN].eq(1)]
                control = cell[cell[ARM_COLUMN].eq(0)]
                n_t, n_c = len(treatment), len(control)
                k_t = int(treatment["category"].isin(positive_categories).sum())
                k_c = int(control["category"].isin(positive_categories).sum())
                difference, lower, upper = newcombe_difference_ci(
                    k_t, n_t, k_c, n_c
                )
                xs.append(ratio_index + offsets[model_id])
                differences.append(difference)
                errors_lo.append(
                    difference - lower if not math.isnan(difference) else math.nan
                )
                errors_hi.append(
                    upper - difference if not math.isnan(difference) else math.nan
                )
                outcome_label = "/".join(sorted(positive_categories))
                if n_t and n_c:
                    print(
                        f"{model_id} {ratio} {outcome_label}: "
                        f"distribution {k_t}/{n_t} ({k_t/n_t:.1%}), "
                        f"answer-only {k_c}/{n_c} ({k_c/n_c:.1%}), "
                        f"difference {difference:+.1%} "
                        f"[95% CI {lower:+.1%}, {upper:+.1%}]"
                    )
                else:
                    print(
                        f"{model_id} {ratio} {outcome_label}: missing one arm "
                        f"(distribution n={n_t}, answer-only n={n_c})"
                    )

            ax.errorbar(
                xs,
                differences,
                yerr=[errors_lo, errors_hi],
                color=model_color(model_id),
                marker="o",
                markersize=5,
                linewidth=1.7,
                elinewidth=1,
                capsize=3,
                label=pretty_model_label(model_id),
            )
        ax.axhline(0, color="#898781", linewidth=1, linestyle="--")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(CONFLICT_RATIOS)))
        ax.set_xticklabels(CONFLICT_RATIOS)
        ax.set_ylim(-1.05, 1.05)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticklabels(["-100 pp", "-50 pp", "0 pp", "+50 pp", "+100 pp"])
        ax.set_xlabel("Evidence ratio")

    axes[0][0].set_ylabel("Distribution requested − Answer only")
    axes[1][0].set_ylabel("Distribution requested − Answer only")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=max(len(labels), 1),
        bbox_to_anchor=(0.5, 0.965), fontsize=9,
    )
    fig.suptitle(
        "RQ4: effect of structured conflict-aware uncertainty elicitation",
        fontsize=13,
        y=1.0,
    )
    fig.text(
        0.5, 0.01,
        "Complete-case estimates. Positive values mean the structured-elicitation "
        "arm produced the outcome more often. Error bars: 95% Newcombe intervals.",
        ha="center", fontsize=9, color="#52514e",
    )
    fig.subplots_adjust(top=0.88, bottom=0.1, hspace=0.32, wspace=0.18)
    save_figure(fig, args, "fig6_rq4_treatment_effect.png")


if __name__ == "__main__":
    main()
