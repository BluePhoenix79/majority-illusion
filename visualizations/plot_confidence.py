"""Figure 3 — confident wrongness (H3).

Mean confidence per ratio, split by whether the selected claim matches the
independent true_value label. For new condition-level files this uses the
model-specific calibrated confidence; older/raw files fall back to the stored
self-report. H3 asks whether models remain highly confident when wrong.

Bars are annotated with n; a group with no data is simply absent.

Usage:
    python visualizations/plot_confidence.py
    python visualizations/plot_confidence.py --csv results/pilot_gemini_gpt5mini.csv
"""

import matplotlib.pyplot as plt

from common import (CATEGORY_COLORS, MUTED, RATIO_ORDER, SURFACE, apply_style,
                    load_results, make_arg_parser, save_figure)

GROUPS = [(True, "Correct value", CATEGORY_COLORS["MAJ"]),
          (False, "Wrong value", CATEGORY_COLORS["MIN"])]


def main():
    args = make_arg_parser(__doc__.splitlines()[0]).parse_args()
    df = load_results(args.csv, args.strategy, args.exclude)
    if "answer_correct" not in df.columns:
        raise SystemExit("Confidence analysis requires true_side/true_value labels.")
    scored = df[df["category"].isin(["MAJ", "MIN"])
                & df["confidence"].notna()].copy()
    if not len(scored):
        raise SystemExit("No truth-labeled rows with usable confidence values.")

    # New condition-level scores always use 0-100. Only legacy raw files need
    # scale detection for their historical 1-5 field.
    scale_max = (
        100 if "modal_category" in df.columns
        else 5 if scored["confidence"].max() <= 5
        else 100
    )
    print(f"Detected confidence scale: 1-{scale_max}"
          if scale_max == 5 else "Detected confidence scale: 0-100")

    models = sorted(scored["model_id"].unique())
    apply_style()
    fig, axes = plt.subplots(1, max(len(models), 1),
                             figsize=(5.5 * max(len(models), 1), 4.5),
                             sharey=True, squeeze=False)

    for ax, model_id in zip(axes[0], models):
        group = scored[scored["model_id"] == model_id]
        width = 0.36
        for gi, (is_correct, label, color) in enumerate(GROUPS):
            xs, means, ns, ratios = [], [], [], []
            for ri, ratio in enumerate(RATIO_ORDER):
                sub = group[(group["ratio"] == ratio)
                            & (group["answer_correct"] == is_correct)]
                if not len(sub):
                    continue
                xs.append(ri + (gi - 0.5) * width)
                means.append(sub["confidence"].mean())
                ns.append(len(sub))
                ratios.append(ratio)
            ax.bar(xs, means, width=width, color=color, label=label,
                   edgecolor=SURFACE, linewidth=2)
            for x, mean, n in zip(xs, means, ns):
                ax.annotate(f"{mean:.1f}" if scale_max == 5 else f"{mean:.0f}",
                            (x, mean), xytext=(0, 4),
                            textcoords="offset points", ha="center",
                            fontsize=9, color=color)
                ax.annotate(f"n={n}", (x, 0), xytext=(0, 4),
                            textcoords="offset points", ha="center",
                            fontsize=7.5, color=MUTED)
            for ratio, mean, n in zip(ratios, means, ns):
                print(f"{model_id} {ratio} correct={is_correct}: mean confidence "
                      f"{mean:.1f} (n={n})")
        ax.set_xticks(range(len(RATIO_ORDER)))
        ax.set_xticklabels([r + ("\n(control)" if r == "4:0" else "")
                            for r in RATIO_ORDER])
        ax.set_ylim(0, scale_max * 1.08)
        ax.set_xlabel("Evidence ratio")
        ax.set_title(group["model_label"].iloc[0] if len(group) else model_id,
                     fontsize=11)

    axes[0][0].set_ylabel(
        f"Mean {'calibrated' if 'modal_category' in df.columns else 'self-reported'} confidence "
        f"({'1-5' if scale_max == 5 else '0-100'})")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.04), fontsize=9)
    fig.suptitle("Confident wrongness: confidence by ratio and correctness",
                 fontsize=12)

    save_figure(fig, args, "fig3_confidence.png")


if __name__ == "__main__":
    main()
