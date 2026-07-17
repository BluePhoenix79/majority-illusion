"""Figure 2 — conflict-flag rate vs. evidence ratio (H2, conflict blindness).

How often each model explicitly says "the sources disagree" instead of picking
a side, across the conflict conditions. H2 predicts this safety behavior fades
as the majority grows more lopsided.

Grouped bars, one bar per model at each ratio. (At small n the 0% bars still
draw wide Wilson error bars, which looks busy; at the full n=75 those whiskers
shrink and the chart reads cleanly.)

Usage:
    python visualizations/plot_flag_rate.py
    python visualizations/plot_flag_rate.py --csv results/pilot_gemini_gpt5mini.csv
"""

import matplotlib.pyplot as plt

from common import (CONFLICT_RATIOS, MUTED, SURFACE, analytic_rate_rows,
                    apply_style, arm_display_label, load_results,
                    make_arg_parser, model_color, save_figure, wilson_ci)


def main():
    args = make_arg_parser(__doc__.splitlines()[0]).parse_args()
    df = load_results(args.csv, args.strategy, args.exclude, args.arm)
    scored = analytic_rate_rows(df)
    scored = scored[scored["ratio"].isin(CONFLICT_RATIOS)]

    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    models = sorted(scored["model_id"].unique())
    # Fit all model bars inside the unit category spacing with a gap between
    # groups; a fixed width overflows once there are >2 models and adjacent
    # ratios overlap.
    width = 0.8 / max(len(models), 1)

    for mi, model_id in enumerate(models):
        group = scored[scored["model_id"] == model_id]
        color = model_color(model_id)
        offsets = [i + (mi - (len(models) - 1) / 2) * width
                   for i in range(len(CONFLICT_RATIOS))]
        rates, errs_lo, errs_hi = [], [], []
        for ratio in CONFLICT_RATIOS:
            sub = group[group["ratio"] == ratio]
            n = len(sub)
            k = (sub["category"] == "FLAG").sum()
            rate = k / n if n else 0.0
            lo, hi = wilson_ci(k, n)
            rates.append(rate)
            # clamp: Wilson bounds can land a float-hair inside the rate at 0%/100%
            errs_lo.append(max(0.0, rate - lo))
            errs_hi.append(max(0.0, hi - rate))
            print(f"{model_id} {ratio}: FLAG {k}/{n} = {rate:.0%}" if n
                  else f"{model_id} {ratio}: no data")
        label = group["model_label"].iloc[0] if len(group) else model_id
        ax.bar(offsets, rates, width=width, color=color, label=label,
               edgecolor=SURFACE, linewidth=2)
        ax.errorbar(offsets, rates, yerr=[errs_lo, errs_hi], fmt="none",
                    ecolor=MUTED, elinewidth=1, capsize=3)
        for x, rate, hi in zip(offsets, rates, errs_hi):
            if rate > 0:  # a 0% bar is absent; labeling the baseline just clutters
                ax.annotate(f"{rate:.0%}", (x, rate + hi), xytext=(0, 4),
                            textcoords="offset points", ha="center", fontsize=8,
                            color=color)

    ax.set_xticks(range(len(CONFLICT_RATIOS)))
    ax.set_xticklabels(CONFLICT_RATIOS)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Evidence ratio (majority : minority documents)")
    ax.set_ylabel("Conflict-flag rate")
    ax.set_title(
        "Conflict abstention by evidence ratio\n"
        f"{arm_display_label(df)}"
    )
    ax.legend(loc="upper right", fontsize=9)

    save_figure(fig, args, "fig2_flag_rate.png")


if __name__ == "__main__":
    main()
