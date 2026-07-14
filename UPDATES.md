# Team Updates Log

Running log of what's been committed to this repo, so everyone can see progress
without digging through git history. Newest entries go at the bottom.

---

## [Jul 13] — Team
Repo created. Locked project: Majority Illusion in RAG. Roles assigned —
Sohan (data), Pranav (harness), Ishaan (analysis), Kartigan (integrator/writing).
Next: generate the ~50 synthetic entities + conflicting documents.

## [Jul 14, 11:50 AM] — Kartigan
Committed: Added UPDATES.md changelog and .gitignore (ignores local CLAUDE.md).
Files: UPDATES.md, .gitignore
Status: Project scaffolding in place. Next: generate synthetic entities + conflicting documents.

## [Jul 14, 12:05 PM] — Kartigan
Correction: no fixed roles on this team. All four of us (Kartigan, Sohan, Pranav, Ishaan)
work across data, the query harness, analysis, and the Research Brief as needed — the
"Sohan/Pranav/Ishaan/Kartigan = data/harness/analysis/writing" split in the Jul 13 entry
above was never a hard assignment. Not touching that entry per the append-only rule; noting
the correction here instead.
Files: none (local CLAUDE.md config updated, not pushed)
Status: Next: generate the ~50 synthetic entities + conflicting documents.

## [Jul 14, 12:45 PM] — Kartigan
Committed: Dataset generator (50 fictional entities, 18 banking-themed, docs at all
4 ratios) + query harness (both models, retry/backoff, per-call error capture,
confidence elicitation, CSV output). Mock pilot passed: 24/24 rows clean, all fields
populated, error-recovery tested.
IMPORTANT: Claude 3.5 Haiku was RETIRED by Anthropic on Feb 19, 2026 (API returns 404).
Harness defaults to claude-haiku-4-5, the official drop-in replacement — note this in
the Research Brief. Model IDs are overridable via CLI flags.
Files: data/generate_dataset.py, data/entities.json, harness/run_experiment.py,
requirements.txt, results/pilot_mock.csv, .gitignore
Status: Pipeline ready. BLOCKED on API keys — export OPENAI_API_KEY + ANTHROPIC_API_KEY,
then run the real pilot: `python harness/run_experiment.py --entities 3`.
Full-run cost estimate: <$0.50 total (~$0.02 gpt-4o-mini + ~$0.15 haiku-4-5).
