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

Condition-level modal ties are represented as TIE (or AMBIGUOUS in sanitized
legacy files). They are displayed in the complete outcome breakdown but are
not valid observations for MAJ- or FLAG-rate denominators.
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
# Claude and DeepSeek share the OpenRouter gateway, so model series must be
# keyed by model_id rather than model_provider or the two models get pooled.
MODEL_COLOR_PREFIXES = [
    ("gemini-3.5-flash", "#2a78d6"),
    ("deepseek/deepseek-v4-flash", "#1baf7a"),
    ("deepseek", "#1baf7a"),
    ("anthropic/claude-haiku-4.5", "#7b4bb7"),
    ("claude", "#7b4bb7"),
]
# Retained for backward compatibility with historical/custom scripts.
MODEL_COLORS = {"gemini": "#2a78d6", "openrouter": "#1baf7a"}  # blue / aqua
CATEGORY_COLORS = {
    "MAJ": "#2a78d6",       # blue
    "MIN": "#eb6834",       # orange
    "COM": "#eda100",       # yellow
    "FLAG": "#4a3aa7",      # violet
    "OTHER": "#c3c2b7",     # gray
    "TIE": "#6f6d68",       # dark gray (new-protocol modal tie)
    "AMBIGUOUS": "#6f6d68", # legacy/sanitized modal tie
    "UNSCORED": "#e1e0d9",  # light gray
}
CATEGORY_ORDER = [
    "MAJ", "MIN", "COM", "FLAG", "OTHER", "TIE", "AMBIGUOUS", "UNSCORED"
]
ANALYTIC_CATEGORIES = {"MAJ", "MIN", "COM", "FLAG", "OTHER"}
NONANALYTIC_CATEGORIES = {"TIE", "AMBIGUOUS", "UNSCORED"}
ARM_COLUMN = "distribution_request_assigned"
ARM_LABELS = {1: "Distribution requested", 0: "Answer only"}


def model_color(model_id):
    mid = str(model_id).lower().removesuffix("-mock")
    for prefix, color in MODEL_COLOR_PREFIXES:
        if mid.startswith(prefix):
            return color
    return MUTED

# Mentioning a contradiction and refusing to resolve it are deliberately
# separate constructs. A model may accurately mention disagreement and still
# choose one supplied value; that is not conflict abstention.
CONFLICT_MENTION_PATTERNS = (
    r"\bconflict(?:ing|s|ed)?\b",
    r"\bdisagree(?:ment|ments|s|d)?\b",
    r"\bcontradict(?:ion|ions|ory|s|ed)?\b",
    r"\binconsisten(?:t|cy|cies)\b",
    r"sources?\s+(?:differ|vary|are\s+split)",
    r"documents?\s+(?:differ|vary|are\s+split)",
    r"reports?\s+(?:differ|vary|are\s+split)",
)

ABSTENTION_PATTERNS = (
    r"\bcannot\s+(?:be\s+)?(?:determine|establish|resolve|verify|conclude)",
    r"\bcan(?:not|'t)\s+(?:determine|choose|select|say\s+which|tell\s+which)",
    r"\bunable\s+to\s+(?:determine|choose|select|resolve|verify)",
    r"\bnot\s+(?:possible|able)\s+to\s+(?:determine|choose|select|resolve)",
    r"\b(?:insufficient|not\s+enough)\s+(?:evidence|information)(?:\s+to\s+(?:determine|choose|select))?",
    r"\bno\s+(?:definitive|reliable|single|unique|conclusive)\s+(?:answer|value|conclusion)",
    r"\b(?:answer|result|value)\s+is\s+(?:indeterminate|undetermined|unresolved)",
    r"\brefus(?:e|es|ed|ing)\s+to\s+(?:choose|select|provide)",
    r"\bneither\s+(?:claim|value|answer)\s+can\s+be\s+(?:determined|selected)",
)

# Models occasionally put a terse non-answer in the JSON ``answer`` field
# instead of writing a full refusal sentence. These are behavioral
# abstentions, not novel factual answers. Keep this deliberately narrow so a
# response that merely *mentions* conflict and still supplies a claim value is
# classified MAJ/MIN/COM rather than FLAG.
TERSE_ABSTENTION_PATTERNS = (
    r"^(?:unknown|unclear|undetermined|unresolved|indeterminate)"
    r"(?:\s+(?:answer|value|result|resolution))?[.!]?$",
    r"^(?:the\s+)?(?:answer|value|result|resolution)\s+is\s+"
    r"(?:unknown|unclear|undetermined|unresolved|indeterminate)[.!]?$",
    r"^(?:conflict|conflicting|source\s+conflict|document\s+conflict|"
    r"conflicting\s+(?:sources?|information|evidence|documents?|reports?))"
    r"(?:\s+in\s+(?:the\s+)?(?:sources?|documents?|reports?))?[.!]?$",
    r"^(?:the\s+)?(?:sources?|documents?|reports?)\s+"
    r"(?:are\s+)?(?:in\s+)?conflict(?:ing)?[.!]?$",
)


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
    if not v:
        return False
    try:
        target = float(v)
    except ValueError:
        return re.search(
            rf"(?<!\w){re.escape(_norm(value))}(?!\w)", text
        ) is not None
    return any(float(n) == target for n in _NUM_RE.findall(text))


def _mentions_claim_label(answer, letter):
    """True if a normalized answer denotes counterbalanced Claim A/B by label.

    Matches "claim a" / "claima" anywhere, or a bare "a" / "(a)" answer -- the
    forms models use when they echo the label instead of the value.
    """
    return bool(
        re.search(rf"\bclaim\s*{letter}\b", answer)
        or re.fullmatch(rf"\(?{letter}\)?", answer.strip())
    )


def _row_value(row, key, default=""):
    """Read a key from a dict or Series without treating missing data as true."""
    try:
        value = row.get(key, default)
    except AttributeError:
        try:
            value = row[key]
        except (KeyError, TypeError):
            value = default
    return default if pd.isna(value) else value


def _row_value_at(df, idx, key, default=""):
    """Safe cell read: returns default when the column is absent."""
    return df.at[idx, key] if key in df.columns else default


def score_response(row):
    """Return category plus conflict-mention and abstention diagnostics.

    ``FLAG`` is reserved for a strong refusal/indeterminate resolution. Merely
    noting a conflict is not enough: an answer that mentions both supplied
    values without refusing is ``COM``; an answer that explicitly supplies
    only one value remains ``MAJ`` or ``MIN``. This keeps source-conflict
    awareness separate from behavioral abstention, as required by RQ4.
    """
    if _row_value(row, "error") or not _row_value(row, "parsed_answer"):
        return {
            "category": "UNSCORED",
            "mentions_conflict": 0,
            "abstained": 0,
        }

    answer = _norm(_row_value(row, "parsed_answer"))
    maj_value = _row_value(row, "majority_value")
    min_value = _row_value(row, "minority_value")
    has_maj = _value_in(maj_value, answer)
    has_min = _value_in(min_value, answer)

    # Resolve counterbalanced claim labels. Models sometimes answer with the
    # label ("Claim A"/"Claim B") instead of the value it denotes; without this
    # a label-only answer falls through to OTHER even though it is really a MAJ
    # or MIN response. Additive and guarded: it only fires when the claim-value
    # columns are present and the answer references a label, so answers that
    # already contain the value (and legacy rows without claim columns) are
    # unaffected.
    claim_a = _norm(_row_value(row, "claim_a_value"))
    claim_b = _norm(_row_value(row, "claim_b_value"))
    if claim_a and _mentions_claim_label(answer, "a"):
        has_maj = has_maj or _value_in(maj_value, claim_a)
        has_min = has_min or _value_in(min_value, claim_a)
    if claim_b and _mentions_claim_label(answer, "b"):
        has_maj = has_maj or _value_in(maj_value, claim_b)
        has_min = has_min or _value_in(min_value, claim_b)

    mentions_conflict = int(any(
        re.search(pattern, answer) for pattern in CONFLICT_MENTION_PATTERNS
    ))
    # A terse non-answer is the model choosing the indeterminate resolution
    # offered by the prompt -- a deliberate abstention, not an OTHER.
    abstained = int(
        bool(re.match(r"^\s*indeterminate\b", answer))
        or any(re.search(pattern, answer) for pattern in TERSE_ABSTENTION_PATTERNS)
        or any(re.search(pattern, answer) for pattern in ABSTENTION_PATTERNS)
    )

    if abstained:
        category = "FLAG"
    elif has_maj and has_min:
        category = "COM"
    elif has_maj:
        category = "MAJ"
    elif has_min:
        category = "MIN"
    else:
        category = "OTHER"
    return {
        "category": category,
        "mentions_conflict": mentions_conflict,
        "abstained": abstained,
    }


def classify(row):
    """Backward-compatible category-only wrapper around :func:`score_response`."""
    return score_response(row)["category"]


def analytic_rate_rows(df):
    """Return rows eligible for MAJ/FLAG rate denominators.

    Complete outcome plots intentionally do *not* call this helper: TIE,
    AMBIGUOUS, and UNSCORED remain visible there as data-quality outcomes.
    """
    eligible = df["category"].isin(ANALYTIC_CATEGORIES)
    if {"n_scored", "n_samples"}.issubset(df.columns):
        n_scored = pd.to_numeric(df["n_scored"], errors="coerce")
        n_samples = pd.to_numeric(df["n_samples"], errors="coerce")
        eligible &= n_scored.eq(n_samples)
    return df[eligible].copy()


def arm_display_label(df):
    """Human-readable treatment-arm provenance for figure titles."""
    if "elicitation_arm" not in df.columns:
        return "Elicitation arm unavailable"
    labels = sorted(set(df["elicitation_arm"].dropna().astype(str)) - {""})
    if len(labels) == 1:
        return f"{labels[0]} arm"
    if len(labels) > 1:
        return "Both elicitation arms (explicitly combined)"
    return "Elicitation arm unavailable"


def _assignment_value(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "distribution", "requested"}:
        return 1
    if text in {"0", "false", "no", "n", "answer-only", "answer_only"}:
        return 0
    return float("nan")


def _filter_elicitation_arm(df, arm):
    """Apply an explicit RQ4 arm filter and prevent accidental pooling."""
    if ARM_COLUMN not in df.columns:
        if arm in {"distribution", "answer-only"}:
            raise SystemExit(
                f"--arm {arm} requires the {ARM_COLUMN} column; this appears "
                "to be a legacy CSV."
            )
        df = df.copy()
        df["elicitation_arm"] = "Unknown (legacy schema)"
        print(
            f"WARNING: {ARM_COLUMN} is absent; treatment-arm filtering is "
            "unavailable for this legacy CSV."
        )
        return df

    assignment = df[ARM_COLUMN].map(_assignment_value)
    nonblank = df[ARM_COLUMN].astype(str).str.strip().ne("")
    invalid = nonblank & assignment.isna()
    if invalid.any():
        examples = sorted(set(df.loc[invalid, ARM_COLUMN].astype(str)))[:5]
        raise SystemExit(
            f"Unrecognized {ARM_COLUMN} values: {examples}. Expected 0/1."
        )
    if assignment.isna().any():
        raise SystemExit(
            f"{ARM_COLUMN} is blank for {int(assignment.isna().sum())} rows; "
            "cannot identify the RQ4 treatment arm safely."
        )

    present = set(assignment.astype(int))
    if arm is None and present == {0, 1}:
        raise SystemExit(
            "This CSV contains both confidence-elicitation arms. Choose "
            "--arm distribution or --arm answer-only for a single-arm plot; "
            "use --arm all only when intentional pooling/comparison is the "
            "purpose of the figure."
        )

    df = df.copy()
    df[ARM_COLUMN] = assignment.astype(int)
    df["elicitation_arm"] = df[ARM_COLUMN].map(ARM_LABELS)
    if arm in {"distribution", "answer-only"}:
        wanted = 1 if arm == "distribution" else 0
        df = df[df[ARM_COLUMN].eq(wanted)].copy()
        print(f"Filtered to arm={arm}: {len(df)} rows")
        if df.empty:
            raise SystemExit(f"No rows remain after --arm {arm}.")
    elif arm == "all":
        labels = [ARM_LABELS[value] for value in sorted(present, reverse=True)]
        print(f"Using all elicitation arms by explicit request: {labels}")
    elif arm is None and len(present) == 1:
        only = next(iter(present))
        print(f"CSV contains one elicitation arm: {ARM_LABELS[only]}")
    return df


def load_results(csv_paths=None, strategy=None, exclude=None, arm=None):
    """Load one or more result CSVs into a scored DataFrame.

    `strategy` filters to one prompting strategy (e.g. "standard" or "cot")
    when the harness ran with several; mixing strategies in one figure would
    silently pool two different experiments.
    """
    if not csv_paths:
        # New runs produce one condition-level row per model/entity/ratio. Use
        # that by default so the three repeated calls are not treated as three
        # independent experimental units in confidence intervals.
        conditions = sorted(RESULTS_DIR.glob("conditions_*.csv"))
        runs = sorted(RESULTS_DIR.glob("run_*.csv"))
        pilot = RESULTS_DIR / "pilot_gemini_gpt5mini.csv"
        if conditions:
            csv_paths = [conditions[-1]]
        elif runs:
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
            raise SystemExit(
                f"This CSV contains multiple prompting strategies {present}. "
                "Pass --strategy standard or --strategy cot; pooling would mix "
                "two experiments."
            )

    df = _filter_elicitation_arm(df, arm)

    is_condition_level = "modal_category" in df.columns
    if is_condition_level:
        df["category"] = (
            df["modal_category"].astype(str).str.strip().str.upper()
        )
        df.loc[df["category"].eq(""), "category"] = "UNSCORED"
        # Interrupted legacy runs sometimes recorded a chosen modal category
        # even when modal_tie=1. Never let that arbitrary choice enter a rate
        # denominator.
        if "modal_tie" in df.columns:
            tied = pd.to_numeric(df["modal_tie"], errors="coerce").eq(1)
            df.loc[tied & ~df["category"].isin({"TIE", "AMBIGUOUS"}),
                   "category"] = "AMBIGUOUS"
        if "parsed_answer" not in df.columns:
            df["parsed_answer"] = df.get("modal_answer", "")
        if "error" not in df.columns:
            df["error"] = df.get("posthoc_error", "")
        # Repair condition rows whose stored modal_category predates the
        # claim-label scorer fix: re-score OTHER rows through the current scorer
        # using the modal answer. Targeted at OTHER only, so correctly stored
        # MAJ/MIN/COM/FLAG/TIE/AMBIGUOUS rows are never disturbed; a genuinely
        # novel modal answer stays OTHER. Consistent no-op on correctly scored
        # future runs.
        if "modal_answer" in df.columns:
            for idx in df.index[df["category"].eq("OTHER")]:
                df.at[idx, "category"] = score_response({
                    "parsed_answer": df.at[idx, "modal_answer"],
                    "majority_value": _row_value_at(df, idx, "majority_value"),
                    "minority_value": _row_value_at(df, idx, "minority_value"),
                    "claim_a_value": _row_value_at(df, idx, "claim_a_value"),
                    "claim_b_value": _row_value_at(df, idx, "claim_b_value"),
                    "error": "",
                })["category"]
    else:
        # Raw logs contain a fourth, post-hoc API call. Outcome analyses use
        # only primary answer calls; token_report.py still counts both phases.
        if "call_phase" in df.columns:
            df = df[df["call_phase"].isin(["", "primary"])].copy()
        # Format-retry attempts remain in the raw log for transparency and
        # token accounting, but only the final record for each planned trial
        # is an experimental observation. Legacy CSVs lack this column.
        if "trial_record" in df.columns:
            trial_record = pd.to_numeric(df["trial_record"], errors="coerce")
            df = df[trial_record.eq(1)].copy()
        df["category"] = df.apply(classify, axis=1)
    if exclude:
        before = len(df)
        df = df[~df["category"].isin(exclude)]
        print(f"WARNING: excluded categories {list(exclude)} — dropped "
              f"{before - len(df)} rows. Preview only; do not quote these "
              "figures in the brief.")
    if is_condition_level:
        # New-protocol schema. Legacy column names remain read-only fallbacks
        # so an interrupted pilot can be inspected without describing its
        # value as calibrated or objectively correct.
        confidence_source = None
        for column in (
            "confidence_best_resolution",
            "posthoc_subjective_confidence",
            "posthoc_probability",
        ):
            if column in df.columns:
                confidence_source = df[column]
                break
        if confidence_source is None:
            confidence_source = pd.Series(float("nan"), index=df.index)
        df["subjective_best_resolution_confidence"] = pd.to_numeric(
            confidence_source, errors="coerce"
        )
        # Retained as a plotting alias; this is explicitly a subjective
        # best-resolution estimate, not an empirical probability of truth.
        df["confidence"] = df["subjective_best_resolution_confidence"]

        for column in (
            "mean_p_majority",
            "mean_p_minority",
            "mean_p_indeterminate",
            "mean_p_sources_conflict",
        ):
            if column not in df.columns:
                df[column] = float("nan")
            else:
                df[column] = pd.to_numeric(df[column], errors="coerce")
    else:
        confidence_source = (
            df["confidence_best_resolution"]
            if "confidence_best_resolution" in df.columns
            else df.get("parsed_confidence", "")
        )
        df["subjective_best_resolution_confidence"] = pd.to_numeric(
            confidence_source, errors="coerce"
        )
        df["confidence"] = df["subjective_best_resolution_confidence"]
    df["majority_share"] = df["ratio"].map(MAJORITY_SHARE)
    df["model_label"] = df["model_id"].map(pretty_model_label)
    df["model_key"] = df["model_id"]
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


def newcombe_difference_ci(k_treatment, n_treatment, k_control, n_control,
                           z=1.96):
    """Newcombe hybrid-score CI for two independent proportion differences.

    Returns ``(difference, lower, upper)`` for treatment minus control. The
    construction combines the two Wilson score intervals without a pooled
    variance assumption.
    """
    if n_treatment == 0 or n_control == 0:
        return (float("nan"), float("nan"), float("nan"))
    p_t = k_treatment / n_treatment
    p_c = k_control / n_control
    lo_t, hi_t = wilson_ci(k_treatment, n_treatment, z=z)
    lo_c, hi_c = wilson_ci(k_control, n_control, z=z)
    difference = p_t - p_c
    lower = difference - (
        (p_t - lo_t) ** 2 + (hi_c - p_c) ** 2
    ) ** 0.5
    upper = difference + (
        (hi_t - p_t) ** 2 + (p_c - lo_c) ** 2
    ) ** 0.5
    return difference, max(-1.0, lower), min(1.0, upper)


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
    ap.add_argument(
        "--arm", choices=("distribution", "answer-only", "all"), default=None,
        help=(
            "confidence-elicitation arm. Required when both arms are present; "
            "'all' is an explicit opt-in to pooling or treatment comparison"
        ),
    )
    ap.add_argument("--exclude", nargs="*", default=None, metavar="CAT",
                    help="drop rows scored as these categories before plotting "
                         "(e.g. --exclude COM); for previews/sensitivity checks")
    return ap


def save_figure(fig, args, default_name):
    FIGURES_DIR.mkdir(exist_ok=True)
    out = Path(args.output) if args.output else FIGURES_DIR / default_name
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
