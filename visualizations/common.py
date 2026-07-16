"""Shared loading, scoring, and figure styling for the visualization scripts.

Every plot script reads experiment CSVs from results/ (default: the most
recent run_*.csv, falling back to the real pilot), classifies each response
with the Day 1 scoring rubric, and saves a 300-dpi PNG into
visualizations/figures/.

Scoring rubric (from the research plan):
  MAJ  - answer matches the majority claim
  MIN  - answer matches the minority claim
  COM  - answer blends/mentions both claims without refusing to choose
  FLAG - answer notes the conflict and refuses to pick one claim
Plus two bookkeeping buckets the rubric implies but doesn't name:
  OTHER    - a parsed answer matching neither claim and no conflict language
  UNSCORED - API error or no JSON answer extracted
"""

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

# Conflict conditions in order of increasing majority share, then the
# no-conflict control.
# NOTE: every ratio in data/entities.json must have a MAJORITY_SHARE entry --
# load_results() maps this column, and an unmapped ratio becomes NaN and is
# silently dropped from the plots rather than erroring.
RATIO_ORDER = ["2:2", "3:2", "2:1", "3:1", "4:1", "4:0"]
CONFLICT_RATIOS = ["2:2", "3:2", "2:1", "3:1", "4:1"]
MAJORITY_SHARE = {
    "2:2": 0.50,   # 2 of 4
    "3:2": 0.60,   # 3 of 5
    "2:1": 2 / 3,  # 2 of 3  (~0.67)
    "3:1": 0.75,   # 3 of 4
    "4:1": 0.80,   # 4 of 5
    "4:0": 1.00,   # 4 of 4 (control, no conflict)
}

# Labels derive from model_id (not provider) since the team has swapped models
# several times; unknown ids fall back to the raw id string.
MODEL_LABEL_PREFIXES = [
    ("gemini-3.5-flash", "Gemini 3.5 Flash"),
    ("gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite"),
    ("gpt-5-mini", "GPT-5 mini (Azure)"),
    ("gpt-4o-mini", "GPT-4o mini (Azure)"),
    ("anthropic/claude-haiku-4.5", "Claude Haiku 4.5"),  # OpenRouter slug
    ("claude-haiku-4-5", "Claude Haiku 4.5"),             # direct-API id
    ("deepseek/deepseek-v4-flash", "DeepSeek V4 Flash"),
    ("deepseek", "DeepSeek"),  # fallback for other deepseek ids
]


def pretty_model_label(model_id):
    mid = str(model_id).lower()
    for prefix, label in MODEL_LABEL_PREFIXES:
        if mid.startswith(prefix):
            return label + (" [mock]" if mid.endswith("-mock") else "")
    return str(model_id)

# Validated categorical palette (dataviz reference palette, light mode).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
# Keyed by the model_provider column. Current study has two providers: gemini
# (native) and openrouter (the gateway, which serves DeepSeek or Claude -- the
# specific model is in model_id/model_label, not the provider). Plot scripts
# fall back to MUTED (gray) for an unknown provider.
MODEL_COLORS = {"gemini": "#2a78d6", "openrouter": "#1baf7a"}  # blue / aqua
CATEGORY_COLORS = {
    "MAJ": "#2a78d6",       # blue
    "MIN": "#eb6834",       # orange
    "COM": "#eda100",       # yellow
    "FLAG": "#4a3aa7",      # violet
    "OTHER": "#c3c2b7",     # gray
    "UNSCORED": "#e1e0d9",  # light gray
}
CATEGORY_ORDER = ["MAJ", "MIN", "COM", "FLAG", "OTHER", "UNSCORED"]

FLAG_PATTERNS = [
    r"\bconflict", r"\bdisagree", r"\bcontradict", r"\binconsistent",
    r"sources\s+(differ|vary)", r"cannot\s+(be\s+)?determin", r"\bunclear\b",
    r"\bambiguous\b", r"\buncertain\b", r"not\s+(possible|able)\s+to\s+determine",
    r"no\s+definitive", r"documents?\s+(differ|vary|are\s+split)",
]


def _norm(value):
    return str(value).lower().replace(",", "").strip()


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _value_in(value, text):
    """Does the answer text assert this claim value?

    Numeric values (including '$12', '3.5%') compare as numbers, so '$12'
    matches '12.00' or '12 dollars' but never '120'. Non-numeric values use
    whole-token string matching so '240' can't hit inside '2400'.
    """
    text = _norm(text)
    v = _norm(value).strip("$%")
    try:
        target = float(v)
    except ValueError:
        return re.search(
            rf"(?<![\w.]){re.escape(_norm(value))}(?![\w.])", text) is not None
    return any(float(n) == target for n in _NUM_RE.findall(text))


def classify(row):
    if row["error"] or not row["parsed_answer"]:
        return "UNSCORED"
    ans = _norm(row["parsed_answer"])
    has_maj = _value_in(row["majority_value"], ans)
    has_min = _value_in(row["minority_value"], ans)
    flagged = any(re.search(p, ans) for p in FLAG_PATTERNS)
    if has_maj and has_min:
        return "FLAG" if flagged else "COM"
    if has_maj:
        return "MAJ"
    if has_min:
        return "MIN"
    return "FLAG" if flagged else "OTHER"


def load_results(csv_paths=None, strategy=None, exclude=None):
    """Load one or more result CSVs into a scored DataFrame.

    `strategy` filters to one prompting strategy (e.g. "standard" or "cot")
    when the harness ran with several; mixing strategies in one figure would
    silently pool two different experiments.
    """
    if not csv_paths:
        runs = sorted(RESULTS_DIR.glob("run_*.csv"))
        pilot = RESULTS_DIR / "pilot_gemini_gpt5mini.csv"
        if runs:
            csv_paths = [runs[-1]]
        elif pilot.exists():
            csv_paths = [pilot]
            print(f"NOTE: no run_*.csv found; using pilot data ({pilot.name}). "
                  "Pilot n is small — treat these figures as smoke tests.")
        else:
            raise SystemExit("No result CSVs found in results/.")
    frames = [pd.read_csv(p, dtype=str).fillna("") for p in map(Path, csv_paths)]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} rows from: {', '.join(Path(p).name for p in csv_paths)}")

    if "strategy" in df.columns:
        present = sorted(set(df["strategy"]) - {""})
        if strategy:
            df = df[df["strategy"] == strategy]
            print(f"Filtered to strategy={strategy}: {len(df)} rows")
        elif len(present) > 1:
            print(f"WARNING: CSV mixes prompting strategies {present}; these "
                  "figures pool them. Re-run with --strategy <name> to split.")

    df["category"] = df.apply(classify, axis=1)
    if exclude:
        before = len(df)
        df = df[~df["category"].isin(exclude)]
        print(f"WARNING: excluded categories {list(exclude)} — dropped "
              f"{before - len(df)} rows. Preview only; do not quote these "
              "figures in the brief.")
    df["confidence"] = pd.to_numeric(df["parsed_confidence"], errors="coerce")
    df["majority_share"] = df["ratio"].map(MAJORITY_SHARE)
    df["model_label"] = df["model_id"].map(pretty_model_label)
    return df


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a proportion; returns (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "font.size": 10,
    })


def make_arg_parser(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--csv", nargs="*", default=None,
                    help="result CSV path(s); default: latest results/run_*.csv, "
                         "else the real pilot CSV")
    ap.add_argument("--output", default=None,
                    help="output PNG path (default: visualizations/figures/<name>.png)")
    ap.add_argument("--strategy", default=None,
                    help="filter to one prompting strategy (standard/cot) when "
                         "the CSV contains several")
    ap.add_argument("--exclude", nargs="*", default=None, metavar="CAT",
                    help="drop rows scored as these categories before plotting "
                         "(e.g. --exclude COM); for previews/sensitivity checks")
    return ap


def save_figure(fig, args, default_name):
    FIGURES_DIR.mkdir(exist_ok=True)
    out = Path(args.output) if args.output else FIGURES_DIR / default_name
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
