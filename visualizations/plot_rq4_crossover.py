"""Figure 9: the ratio-dependent crossover of structured elicitation (RQ4).

Isolates the novel finding that structured confidence-and-conflict elicitation
has *directionally different* effects depending on the evidence ratio. Each
point is the treatment effect -- the percentage-point difference between the
structured-elicitation arm and the matched answer-only control -- for one model
at one conflict ratio. A dashed line marks zero (no effect).

Left panel (majority selection): the effect crosses zero -- elicitation reduces
majority-following at the 2:2 tie but increases it at the slight-majority ratios.
Right panel (conflict flagging): elicitation adds abstention overwhelmingly at
the tie. Together they show the dominant effect shifting from added caution (at
the tie) to added majority-adherence (at slight majorities), with large
heterogeneity across models.

Usage:
    python visualizations/plot_rq4_crossover.py --csv results/conditions_<run>.csv --strategy standard
"""

import matplotlib.pyplot as plt

from common import (CONFLICT_RATIOS, MUTED, apply_style, load_results,
                    make_arg_parser, model_color, pretty_model_label,
                    save_figure)
import pandas as pd

ANALYTIC = {"MAJ", "MIN", "COM", "FLAG", "OTHER"}


def main():
    args = make_arg_parser(__doc__.splitlines()[0]).parse_args()
    df = load_results(args.csv, args.strategy, args.exclude, "all")
    df["arm"] = pd.to_numeric(df.get("distribution_request_assigned"),
                              errors="coerce")
    a = df[df["category"].isin(ANALYTIC)]
    ratios = [r for r in CONFLICT_RATIOS if r in set(a["ratio"])]
    models = sorted(a["model_id"].unique())
    x = list(range(len(ratios)))

    def delta(model_id, ratio, cat):
        ao = a[(a.model_id == model_id) & (a.ratio == ratio) & (a.arm == 0)]
        di = a[(a.model_id == model_id) & (a.ratio == ratio) & (a.arm == 1)]
        if not len(ao) or not len(di):
            return float("nan")
        return 100 * ((di.category == cat).mean() - (ao.category == cat).mean())

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    panels = [("MAJ", "Effect on majority-following"),
              ("FLAG", "Effect on conflict-flagging")]

    for ax, (cat, title) in zip(axes, panels):
        ax.axhline(0, color=MUTED, linestyle="--", linewidth=1, zorder=1)
        for model_id in models:
            ys = [delta(model_id, r, cat) for r in ratios]
            ax.plot(x, ys, color=model_color(model_id), marker="o",
                    markersize=6, linewidth=2, label=pretty_model_label(model_id),
                    zorder=3)
            print(f"{cat} {pretty_model_label(model_id)[:16]:16} "
                  + "  ".join(f"{r}:{y:+.0f}" for r, y in zip(ratios, ys)))
        ax.set_xticks(x)
        ax.set_xticklabels(ratios)
        ax.set_xlim(-0.3, len(ratios) - 0.7)
        ax.set_xlabel("Evidence ratio")
        ax.set_title(title, fontsize=11)

    axes[0].set_ylabel("Treatment effect (pp):\nstructured $-$ answer-only")
    # a light annotation of the interpretation on the MAJ panel
    axes[0].annotate("more caution\n(follows majority less)", (0, -8),
                     fontsize=8, color=MUTED, ha="left", va="top")
    axes[0].annotate("more majority-\nadherence", (len(ratios) - 1.4, 22),
                     fontsize=8, color=MUTED, ha="left")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle("Ratio-dependent crossover: structured elicitation adds caution "
                 "at the tie,\nbut majority-adherence at slight majorities",
                 fontsize=12)
    fig.subplots_adjust(top=0.82)
    save_figure(fig, args, "fig9_rq4_crossover.png")


if __name__ == "__main__":
    main()
