"""Figure 3 — confident wrongness (H3).

Mean self-reported confidence per ratio, split by whether the answer followed
the majority claim (ground truth) or the minority claim (wrong). H3 predicts
models are MORE confident at lopsided ratios (3:1, 4:1) even when wrong,
meaning confidence can't filter bad answers out of a RAG pipeline.

Bars are annotated with n; a group with no data is simply absent.

Usage:
    python visualizations/plot_confidence.py
    python visualizations/plot_confidence.py --csv results/pilot_gemini_gpt5mini.csv
"""

import matplotlib.pyplot as plt

from common import (CATEGORY_COLORS, MUTED, RATIO_ORDER, SURFACE, apply_style,
                    load_results, make_arg_parser, save_figure)

GROUPS = [("MAJ", "Followed majority (correct)", CATEGORY_COLORS["MAJ"]),
          ("MIN", "Followed minority (wrong)", CATEGORY_COLORS["MIN"])]


def main():
    args = make_arg_parser(__doc__.splitlines()[0]).parse_args()
    df = load_results(args.csv)
    scored = df[df["category"].isin(["MAJ", "MIN"])
                & df["confidence"].notna()]

    providers = sorted(scored["model_provider"].unique())
    apply_style()
    fig, axes = plt.subplots(1, max(len(providers), 1),
                             figsize=(5.5 * max(len(providers), 1), 4.5),
                             sharey=True, squeeze=False)

    for ax, provider in zip(axes[0], providers):
        group = scored[scored["model_provider"] == provider]
        width = 0.36
        for gi, (cat, label, color) in enumerate(GROUPS):
            xs, means, ns, ratios = [], [], [], []
            for ri, ratio in enumerate(RATIO_ORDER):
                sub = group[(group["ratio"] == ratio)
                            & (group["category"] == cat)]
                if not len(sub):
                    continue
                xs.append(ri + (gi - 0.5) * width)
                means.append(sub["confidence"].mean())
                ns.append(len(sub))
                ratios.append(ratio)
            ax.bar(xs, means, width=width, color=color, label=label,
                   edgecolor=SURFACE, linewidth=2)
            for x, mean, n in zip(xs, means, ns):
                ax.annotate(f"{mean:.0f}", (x, mean), xytext=(0, 4),
                            textcoords="offset points", ha="center",
                            fontsize=9, color=color)
                ax.annotate(f"n={n}", (x, 0), xytext=(0, 4),
                            textcoords="offset points", ha="center",
                            fontsize=7.5, color=MUTED)
            for ratio, mean, n in zip(ratios, means, ns):
                print(f"{provider:8s} {ratio} {cat}: mean confidence "
                      f"{mean:.1f} (n={n})")
        ax.set_xticks(range(len(RATIO_ORDER)))
        ax.set_xticklabels([r + ("\n(control)" if r == "4:0" else "")
                            for r in RATIO_ORDER])
        ax.set_ylim(0, 108)
        ax.set_xlabel("Evidence ratio")
        ax.set_title(group["model_label"].iloc[0] if len(group) else provider,
                     fontsize=11)

    axes[0][0].set_ylabel("Mean self-reported confidence (0-100)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.04), fontsize=9)
    fig.suptitle("Confident wrongness: confidence by ratio and answer side",
                 fontsize=12)

    save_figure(fig, args, "fig3_confidence.png")


if __name__ == "__main__":
    main()
