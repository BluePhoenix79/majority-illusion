"""Figure 1 — the majority-illusion curve (H1).

Majority-follow rate vs. evidence ratio, one line per model, with 95% Wilson
confidence intervals. The 4:0 control shows context compliance when all
documents agree; a rising curve as majority share grows is the majority
illusion. It is not a factual-accuracy measure.

Usage:
    python visualizations/plot_majority_curve.py            # latest run CSV
    python visualizations/plot_majority_curve.py --csv results/pilot_gemini_gpt5mini.csv
"""

import matplotlib.pyplot as plt

from common import (RATIO_ORDER, apply_style, load_results, make_arg_parser,
                    model_color, save_figure, wilson_ci)


def main():
    args = make_arg_parser(__doc__.splitlines()[0]).parse_args()
    df = load_results(args.csv, args.strategy, args.exclude)
    scored = df[df["category"] != "UNSCORED"]

    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    models = list(scored.groupby("model_id"))
    for mi, (model_id, group) in enumerate(models):
        # Small x-offset per series so identical values don't occlude.
        x_pos = [i + (mi - (len(models) - 1) / 2) * 0.06
                 for i in range(len(RATIO_ORDER))]
        color = model_color(model_id)
        rates, los, his = [], [], []
        for ratio in RATIO_ORDER:
            sub = group[group["ratio"] == ratio]
            n = len(sub)
            k = (sub["category"] == "MAJ").sum()
            rate = k / n if n else float("nan")
            lo, hi = wilson_ci(k, n)
            rates.append(rate)
            # clamp: Wilson bounds can land a float-hair inside the rate at 0%/100%
            los.append(max(0.0, rate - lo))
            his.append(max(0.0, hi - rate))
            print(f"{model_id} {ratio}: MAJ {k}/{n} = {rate:.0%}" if n
                  else f"{model_id} {ratio}: no data")
        label = group["model_label"].iloc[0]
        ax.errorbar(x_pos, rates, yerr=[los, his], color=color, linewidth=2,
                    marker="o", markersize=6, capsize=3, elinewidth=1,
                    label=label)

    ax.set_xticks(range(len(RATIO_ORDER)))
    ax.set_xticklabels([r + ("\n(control)" if r == "4:0" else "")
                        for r in RATIO_ORDER])
    ax.set_xlim(-0.3, len(RATIO_ORDER) - 0.3)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Evidence ratio (majority : minority documents)")
    ax.set_ylabel("Majority-follow rate")
    ax.set_title("Majority-follow rate by evidence ratio")
    ax.legend(loc="lower right", fontsize=9)

    save_figure(fig, args, "fig1_majority_curve.png")


if __name__ == "__main__":
    main()
