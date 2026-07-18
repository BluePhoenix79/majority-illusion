"""Figure 8: the awareness-action gap (headline safety finding).

In the structured-elicitation arm the model reports, independently of its
answer, a 0-100 probability that the sources conflict (p_sources_conflict). This
figure places that *awareness* signal next to the model's protective *action* --
the rate at which it actually flags/abstains (FLAG) -- across evidence ratios.
The large, persistent gap between the two is the core result: models recognize
the conflict but rarely act on it, instead following the majority.

Pooled across models (the gap is near-universal). Structured arm only, since the
awareness measure exists only there.

Usage:
    python visualizations/plot_awareness_gap.py --csv results/conditions_<run>.csv --strategy standard
"""

import matplotlib.pyplot as plt

from common import (CONFLICT_RATIOS, MUTED, apply_style, load_results,
                    make_arg_parser, save_figure)

AWARE_COLOR = "#4a3aa7"   # violet  -- what the model KNOWS
ACTION_COLOR = "#e34948"  # red     -- what the model DOES (flags)
MAJ_COLOR = "#2a78d6"     # blue    -- what it does instead (follow majority)


def main():
    args = make_arg_parser(__doc__.splitlines()[0]).parse_args()
    df = load_results(args.csv, args.strategy, args.exclude, "distribution")
    df = df[df["mean_p_sources_conflict"].notna()]
    scored = df[df["category"].isin({"MAJ", "MIN", "COM", "FLAG", "OTHER"})]
    ratios = [r for r in CONFLICT_RATIOS if r in set(scored["ratio"])]
    x = list(range(len(ratios)))

    aware, flag, maj, ns = [], [], [], []
    for r in ratios:
        sub = scored[scored["ratio"] == r]
        n = len(sub)
        aware.append(df[df["ratio"] == r]["mean_p_sources_conflict"].mean())
        flag.append(100 * (sub["category"] == "FLAG").mean() if n else float("nan"))
        maj.append(100 * (sub["category"] == "MAJ").mean() if n else float("nan"))
        ns.append(n)
        print(f"{r}: awareness P(conflict)={aware[-1]:.0f}  FLAG={flag[-1]:.0f}%  "
              f"MAJ={maj[-1]:.0f}%  n={n}")

    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    # shade the gap between awareness and protective action
    ax.fill_between(x, flag, aware, color=AWARE_COLOR, alpha=0.10, zorder=1)
    ax.plot(x, aware, color=AWARE_COLOR, marker="o", markersize=7, linewidth=2.5,
            label="Reported P(sources conflict)  —  awareness", zorder=3)
    ax.plot(x, maj, color=MAJ_COLOR, marker="s", markersize=6, linewidth=2,
            linestyle="--", label="Majority-follow rate  —  what it does instead",
            zorder=2)
    ax.plot(x, flag, color=ACTION_COLOR, marker="D", markersize=6, linewidth=2.5,
            label="Conflict-flag rate  —  protective action", zorder=3)

    # annotate the gap at the most lopsided ratio
    if aware and flag:
        xi = len(x) - 1
        ax.annotate("", xy=(xi + 0.12, aware[xi]), xytext=(xi + 0.12, flag[xi]),
                    arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.2))
        ax.annotate("awareness–\naction gap", (xi + 0.16, (aware[xi] + flag[xi]) / 2),
                    va="center", fontsize=8.5, color=MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels(ratios)
    ax.set_xlim(-0.3, len(ratios) - 0.4)
    ax.set_ylim(-5, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Evidence ratio (majority : minority documents)")
    ax.set_ylabel("Rate / reported probability")
    ax.set_title("The awareness–action gap: models report the conflict "
                 "but rarely flag it")
    ax.legend(loc="center left", fontsize=8.5)

    save_figure(fig, args, "fig8_awareness_gap.png")


if __name__ == "__main__":
    main()
