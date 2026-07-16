"""Tally token usage and estimated cost across saved result CSVs.

Reads the prompt_tokens / completion_tokens columns the harness already records,
so this is a pure accounting pass -- it makes no API calls and costs nothing.
Mock rows (model_id ending in -MOCK) never hit an API and are excluded.

Pricing comes from PRICING in run_experiment.py; some rates there are UNVERIFIED
and marked with * in the output. Token counts are exact regardless.

Usage:
    python harness/token_report.py                  # every results/*.csv
    python harness/token_report.py results/run_full_standard.csv
    python harness/token_report.py --by-file        # per-file breakdown too
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment import summarize_usage  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def read_rows(path):
    """Read a results CSV, tolerating pre-fix files written in cp1252."""
    try:
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except UnicodeDecodeError:
        print(f"  NOTE: {Path(path).name} is not UTF-8 (pre-encoding-fix file); "
              f"reading as cp1252.", file=sys.stderr)
        with open(path, encoding="cp1252", newline="") as f:
            return list(csv.DictReader(f))


def tally(paths):
    """Return {model_id: [calls, input_tokens, output_tokens]} across paths."""
    usage = defaultdict(lambda: [0, 0, 0])
    for path in paths:
        for row in read_rows(path):
            model_id = row.get("model_id", "")
            if not model_id or model_id.upper().endswith("-MOCK"):
                continue  # mock rows never billed
            try:
                tin = int(row.get("prompt_tokens") or 0)
                tout = int(row.get("completion_tokens") or 0)
            except ValueError:
                tin = tout = 0
            u = usage[model_id]
            u[0] += 1
            u[1] += tin
            u[2] += tout
    return usage


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csvs", nargs="*", help="result CSVs (default: results/*.csv)")
    ap.add_argument("--by-file", action="store_true",
                    help="also print a per-file breakdown")
    args = ap.parse_args()

    paths = [Path(p) for p in args.csvs] or sorted(RESULTS_DIR.glob("*.csv"))
    if not paths:
        sys.exit("No result CSVs found.")

    if args.by_file:
        for p in paths:
            per = tally([p])
            if per:
                print(f"\n### {p.name}")
                summarize_usage(per)

    print(f"\n### TOTAL across {len(paths)} file(s)")
    summarize_usage(tally(paths))


if __name__ == "__main__":
    main()
