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

## [Jul 15, 5:16 PM] — Kartigan
Committed: REAL PILOT PASSED — 24/24 calls clean, 0 errors, every CSV field populated,
confidence captured for both models. Final model pair:
  Model A: gemini-3.1-flash-lite   Model B: gpt-5-mini (Azure OpenAI)
Three bugs found and fixed via the pilot:
 1. gpt-5-mini is a GPT-5 reasoning model — requires max_completion_tokens (rejects
    max_tokens) and needs headroom for hidden reasoning tokens.
 2. Gemini is also a thinking model; the old 300-token cap was consumed by thinking
    tokens and truncated the JSON mid-object (parse failures on the conflicting
    ratios, which reason more). Both models now get ~2K output budget.
 3. Backoff now honors the server-supplied retryDelay on 429s (Gemini's per-minute
    quota needs ~28s, far longer than the old ~8s exponential ceiling).
MODEL CHOICE — document in the Research Brief: gemini-3.5-flash is capped at 20 req/day
on this project's free quota, unusable for a 200-call run, so we dropped to
gemini-3.1-flash-lite (500 req/day, 15 req/min). Flash-LITE is Google's smallest/cheapest
tier — it is NOT a frontier model, so do not describe it as one (this matters for the AI
Use Transparency Statement). Confirmed the exact model ID against live models.list().
Measured usage (per call avg): gemini 286 in / 26 out; gpt-5-mini 280 in / 229 out
(the high output is reasoning tokens).
Full 50-entity run projection (200 calls/model): gemini ~57K in / ~5K out — $0, fits the
500/day free tier; gpt-5-mini ~56K in / ~46K out — well under $1 on Azure credits.
Runtime is bound by Gemini's 15 req/min: ~15+ min for the full run.
Files: harness/run_experiment.py, results/pilot_gemini_gpt5mini.csv, UPDATES.md
Status: Harness VERIFIED and ready to scale to the full 50-entity run
(`python harness/run_experiment.py`). Next: full run, then analysis + Research Brief.

## [Jul 15, 6:30 PM] — Pranav
Committed: Merged remote updates (Gemini/Azure OpenAI integration, reasoning tokens, rate limit retry) with local modifications. Preserved both upstream model providers (Gemini 3.1 Flash-Lite + Azure GPT-5 Mini) and local Anthropic support, and merged local improvements including prompting strategy configuration (standard vs. CoT), multiple trial repetitions with deterministic doc shuffling, and banking-themed entity attributes (interest rate, monthly fee, lending cap, overdraft limit).
Files: data/entities.json, data/generate_dataset.py, harness/run_experiment.py, results/pilot_mock.csv
Status: Pipeline ready and fully merged. Next: run the full experiment with standard and/or CoT prompting strategies.

## [Jul 15, 7:40 PM] — Pranav
Committed: Documented the core hypotheses, variables, experimental design, and pipeline details in research_brief.md.
Files: research_brief.md, UPDATES.md
Status: Design and hypotheses fully documented. Ready to execute the full 50-entity run next.

## [Jul 15, 10:40 PM] — Kartigan
Committed: Model A switched BACK to gemini-3.5-flash (frontier Flash tier) now that a key
with unrestricted 3.5 Flash access is available. This reverses the quota-driven downgrade
to gemini-3.1-flash-lite from the previous entry — that entry's "Flash-Lite is NOT a
frontier model, don't call it one" caveat NO LONGER APPLIES. Both models (gemini-3.5-flash
+ gpt-5-mini) are current-generation, so the Research Brief / AI Use Transparency Statement
can describe the pair accordingly. (Not editing the earlier entry, per the append-only
rule — noting the reversal here instead.)
HEADS-UP — NOT YET RUNNABLE: the 2-call fire test currently FAILS on the Gemini side with
403 PERMISSION_DENIED, reason API_KEY_SERVICE_BLOCKED. This is a KEY-CONFIG issue, not a
code issue: the key in the local .env is blocked from generativelanguage.googleapis.com
entirely (even ListModels fails), so no model string would work with it. Fix in Google
Cloud Console: enable the Generative Language API on the project and/or clear the key's
API restrictions. The gemini-3.5-flash model string itself was confirmed working earlier
with a previous key. The gpt-5-mini/Azure side is green (fire test clean).
Files: harness/run_experiment.py, UPDATES.md
Status: Code ready for the full 50-entity run, PENDING the Gemini key fix. Re-run the fire
test first: `python harness/run_experiment.py --entities 1 --ratios 3:1`.
