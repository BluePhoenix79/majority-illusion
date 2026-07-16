"""Figure 2 — conflict-flag rate vs. evidence ratio (H2, conflict blindness).

How often each model explicitly says "the sources disagree" instead of picking
a side, across the three conflict conditions. H2 predicts this safety behavior
fades as the majority grows more lopsided (2:2 -> 3:1 -> 4:1).

Usage:
    python visualizations/plot_flag_rate.py
    python visualizations/plot_flag_rate.py --csv results/pilot_gemini_gpt5mini.csv
"""

import matplotlib.pyplot as plt

from common import (CONFLICT_RATIOS, MODEL_COLORS, MUTED, SURFACE, apply_style,
                    load_results, make_arg_parser, save_figure, wilson_ci)


def main():
    args = make_arg_parser(__doc__.splitlines()[0]).parse_args()
    df = load_results(args.csv)
    scored = df[(df["category"] != "UNSCORED")
                & (df["ratio"].isin(CONFLICT_RATIOS))]

    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    providers = sorted(scored["model_provider"].unique())
    width = 0.36

    for mi, provider in enumerate(providers):
        group = scored[scored["model_provider"] == provider]
        color = MODEL_COLORS.get(provider, MUTED)
        offsets = [i + (mi - (len(providers) - 1) / 2) * width
                   for i in range(len(CONFLICT_RATIOS))]
        rates, errs_lo, errs_hi = [], [], []
        for ratio in CONFLICT_RATIOS:
            sub = group[group["ratio"] == ratio]
            n = len(sub)
            k = (sub["category"] == "FLAG").sum()
            rate = k / n if n else 0.0
            lo, hi = wilson_ci(k, n)
            rates.append(rate)
            errs_lo.append(rate - lo)
            errs_hi.append(hi - rate)
            print(f"{provider:8s} {ratio}: FLAG {k}/{n} = {rate:.0%}" if n
                  else f"{provider:8s} {ratio}: no data")
        label = group["model_label"].iloc[0] if len(group) else provider
        ax.bar(offsets, rates, width=width, color=color, label=label,
               edgecolor=SURFACE, linewidth=2)
        ax.errorbar(offsets, rates, yerr=[errs_lo, errs_hi], fmt="none",
                    ecolor=MUTED, elinewidth=1, capsize=3)
        for x, rate in zip(offsets, rates):
            ax.annotate(f"{rate:.0%}", (x, rate), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=9,
                        color=color)

    ax.set_xticks(range(len(CONFLICT_RATIOS)))
    ax.set_xticklabels(CONFLICT_RATIOS)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Evidence ratio (majority : minority documents)")
    ax.set_ylabel("Conflict-flag rate")
    ax.set_title("Conflict blindness: flagging fades as the majority grows")
    ax.legend(loc="upper right", fontsize=9)

    save_figure(fig, args, "fig2_flag_rate.png")


if __name__ == "__main__":
    main()
