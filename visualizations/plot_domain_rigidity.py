"""Figure 7: domain rigidity (RQ2).

Conflict-acknowledgement rate (FLAG or COM) by domain -- banking (sensitive
numeric financial parameters) versus general corporate facts -- across evidence
ratios. RQ2 asks whether the sensitivity of the attribute makes models more
cautious, i.e. more likely to flag or compromise rather than silently commit to
the majority.

One line per domain, with 95% Wilson intervals. Pass --arm to pick an
elicitation arm if the CSV contains more than one.

Usage:
    python visualizations/plot_domain_rigidity.py --csv results/conditions_<run>.csv --arm all
"""

import matplotlib.pyplot as plt

from common import (CONFLICT_RATIOS, MUTED, apply_style, arm_display_label,
                    load_results, make_arg_parser, save_figure, wilson_ci)

DOMAIN_COLORS = {"banking": "#eb6834", "general": "#2a78d6"}   # orange / blue
DOMAIN_LABELS = {"banking": "Banking (sensitive)",
                 "general": "General corporate"}


def main():
    args = make_arg_parser(__doc__.splitlines()[0]).parse_args()
    df = load_results(args.csv, args.strategy, args.exclude, args.arm)
    scored = df[(df["category"] != "UNSCORED")
                & df["ratio"].isin(CONFLICT_RATIOS)]
    if not len(scored):
        raise SystemExit("No scored conflict-ratio rows to plot.")

    ratios = [r for r in CONFLICT_RATIOS if r in set(scored["ratio"])]
    domains = [d for d in ("banking", "general")
               if d in set(scored["domain"])]

    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for di, dom in enumerate(domains):
        g = scored[scored["domain"] == dom]
        # small x-offset so overlapping points/error bars stay legible
        xs = [i + (di - (len(domains) - 1) / 2) * 0.06
              for i in range(len(ratios))]
        rates, los, his = [], [], []
        for r in ratios:
            sub = g[g["ratio"] == r]
            n = len(sub)
            k = sub["category"].isin(["FLAG", "COM"]).sum()
            rate = k / n if n else float("nan")
            lo, hi = wilson_ci(k, n)
            rates.append(rate)
            los.append(max(0.0, rate - lo) if n else 0.0)
            his.append(max(0.0, hi - rate) if n else 0.0)
            print(f"{dom:8s} {r}: FLAG|COM {k}/{n} = "
                  f"{rate:.0%}" if n else f"{dom:8s} {r}: no data")
        ax.errorbar(xs, rates, yerr=[los, his], color=DOMAIN_COLORS[dom],
                    marker="o", markersize=6, linewidth=2, capsize=3,
                    label=DOMAIN_LABELS[dom])

    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels(ratios)
    ax.set_xlim(-0.3, len(ratios) - 0.7)
    ax.set_ylim(-0.05, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Evidence ratio (majority : minority documents)")
    ax.set_ylabel("Conflict-acknowledgement rate (FLAG or COM)")
    ax.set_title("Domain rigidity: conflict acknowledgement by domain\n"
                 f"{arm_display_label(df)}", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)

    save_figure(fig, args, "fig7_domain_rigidity.png")


if __name__ == "__main__":
    main()
