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

## [Jul 14, 11:17 PM] — Kartigan
Committed: Switched Model A from OpenAI gpt-4o-mini to Gemini 3.5 Flash (gemini-3.5-flash,
current frontier Flash, GA 2026-05-19) via the google-genai SDK, key from GEMINI_API_KEY.
Model B unchanged (Claude Haiku 4.5). Added exponential-backoff+jitter retry for the
Gemini path (Anthropic SDK already retries internally); optional .env auto-load; .env
gitignored. Offline mock pilot still 24/24 clean, all fields populated.
Files: harness/run_experiment.py, .gitignore, UPDATES.md
Status: WIP commit — real pilot NOT yet run (blocked on GEMINI_API_KEY + ANTHROPIC_API_KEY).
Next: with keys in a local .env, run `python harness/run_experiment.py --entities 3` to
confirm both model strings resolve (no 404) and capture real token usage. Rough full-run
cost ~$0.30 (~$0.18 gemini-3.5-flash + ~$0.11 haiku-4-5); to be refined from the pilot.

## [Jul 15, 1:31 PM] — Kartigan
Committed: Model B switched from Anthropic (Claude Haiku 4.5) to OpenAI via Azure OpenAI
Service, to use Azure for Students credits. Harness now uses the AzureOpenAI client;
config read from AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY / AZURE_OPENAI_DEPLOYMENT
(default gpt-4o-mini) / AZURE_OPENAI_API_VERSION (default 2024-10-21). On Azure the API
`model` arg is the DEPLOYMENT name, not the base model name. OpenAI SDK retries transient
errors internally (max_retries); Gemini path keeps its backoff wrapper. requirements.txt
updated (dropped anthropic; added google-genai + python-dotenv, which were missing).
Offline mock pilot 24/24 clean; missing-config guard now checks the Azure vars.
Files: harness/run_experiment.py, requirements.txt, UPDATES.md
Status: Study is now Gemini 3.5 Flash vs. OpenAI (gpt-4o-mini on Azure). Real pilot still
NOT run — BLOCKED on GEMINI_API_KEY + Azure config (endpoint, key, deployment name) in a
local .env. Next: `python harness/run_experiment.py --entities 3`, then refine cost from
real usage.
