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
Model A: gemini-3.1-flash-lite Model B: gpt-5-mini (Azure OpenAI)
Three bugs found and fixed via the pilot:

1.  gpt-5-mini is a GPT-5 reasoning model — requires max_completion_tokens (rejects
    max_tokens) and needs headroom for hidden reasoning tokens.
2.  Gemini is also a thinking model; the old 300-token cap was consumed by thinking
    tokens and truncated the JSON mid-object (parse failures on the conflicting
    ratios, which reason more). Both models now get ~2K output budget.
3.  Backoff now honors the server-supplied retryDelay on 429s (Gemini's per-minute
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

- gpt-5-mini) are current-generation, so the Research Brief / AI Use Transparency Statement
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

## [Jul 15, 10:52 PM] — Kartigan

Committed: ROOT CAUSE of the 403 API_KEY_SERVICE_BLOCKED found and fixed. The Gemini key
was provisioned for VERTEX AI EXPRESS MODE (Google Cloud credits) -- scoped to
aiplatform.googleapis.com, not AI Studio's generativelanguage.googleapis.com. That's why
even ListModels was blocked: the key was never valid for the AI Studio endpoint at all,
regardless of model name, quota, or restrictions.
Fix: harness now supports a Vertex Express Mode client path, toggled by a new env var
GEMINI_USE_VERTEX=1 (set in the local .env). Express Mode client shape is
genai.Client(vertexai=True, api_key=...) -- NO project/location; the installed SDK
(google-genai 2.11.0) rejects project/location alongside api_key as mutually exclusive.
Default (var unset) stays the plain AI Studio client, so a future AI-Studio-scoped key
still works with zero code changes -- just don't set the var.
Fire test via the harness itself (not a standalone probe) now PASSES both models:
gemini-3.5-flash: ans='Vantry Heights' conf=90 in=270 out=20
gpt-5-mini: ans='Vantry Heights' conf=80 in=271 out=152
Files: harness/run_experiment.py, UPDATES.md
Status: Both models verified live. Full pilot (--entities 3) not yet re-run under this
config -- do that next, then the full 50-entity run on go-ahead.

## [Jul 16, 12:31 AM] — Kartigan

Committed: FULL 50-entity STANDARD-strategy run complete (400 calls, 0 hard exceptions,
3 parse_failures = 0.75%). CoT-strategy run was IN PROGRESS (~377/400) at commit time and
NOT included -- rerun it: `python harness/run_experiment.py --strategy cot --output
results/run_full_cot.csv`. It was running in a background shell tied to this session, which
will not survive a session/machine switch, so treat it as not done.

BUG FOUND + FIXED: harness wrote CSV output without encoding="utf-8". On Windows, open()
without an explicit encoding defaults to the system locale (cp1252), so any em-dash or other
non-ASCII punctuation in a model response corrupted the file for UTF-8 readers (pandas threw
UnicodeDecodeError on byte 0x97). Converted the already-collected run_full_standard.csv from
cp1252 to utf-8 in place (0 data loss, verified 400 rows before/after). New runs are correct
by default now. If you have ANY older CSV in results/ that throws UnicodeDecodeError when
loaded, it needs the same cp1252->utf-8 conversion.

REAL FINDINGS (using visualizations/common.py's actual classifier, not a naive substring
check -- see that file for the MAJ/MIN/COM/FLAG/OTHER/UNSCORED rubric):
ratio majority% gpt-5-mini MAJ% gemini-3.5-flash MAJ% gemini COM%
2:2 50% 32% 30% 8%
3:1 75% 100% 54% 22%
4:1 80% 100% 58% 34%
4:0 100% 100% 96% 0%
gpt-5-mini is essentially a STEP FUNCTION: saturates to 100% majority-following the instant
majority share crosses ~75% and stays flat -- 3:1 and 4:1 are indistinguishable for this
model. gemini-3.5-flash does NOT saturate -- it climbs gradually and has NOT converged even
at 80% majority share (4:1), instead increasingly citing both values (COM) rather than
picking one outright. This is a genuine, reportable per-model difference for Hypothesis 1.
CORRECTION to my earlier note in this log: I said the majority effect "saturates at 3:1" --
that was from a 3-entity pilot and is WRONG for Gemini at proper n=50. It only holds for
gpt-5-mini. Don't repeat the n=3 conclusion in the Research Brief.

RECOMMENDATION -- add intermediate ratios 2:1 (67%, 3 docs) and 3:2 (60%, 5 docs) to
data/generate_dataset.py's RATIOS dict. Verified this is PURELY ADDITIVE: generated a test
dataset with both ratios added and diffed against the current data/entities.json -- 0
mismatches across all 50 entities' names/values/questions/documents for the 4 existing
ratios (the per-ratio document RNG is seeded independently per ratio, so adding new ratios
can't perturb old ones). NOT applied -- pending review, since it changes the default full-run
scope from 400 to 600 calls/strategy. Do NOT add 5:1/6:1 -- the doc-style pool caps at 5
templates (4:1 already uses all 5), so a 6-doc ratio raises `ValueError: sample larger than
population` in make_documents(); it would also land past where gpt-5-mini already saturates
and gemini is already near-saturated, so it wouldn't add signal anyway.
Files: harness/run_experiment.py, results/run_full_standard.csv, UPDATES.md
Status: Standard-strategy data (n=50, both models) is clean and ready for analysis. CoT run
needs to be started fresh on the Mac. entities.json ratio addition needs a decision -- if
approved, run `python data/generate_dataset.py` then regenerate both full runs.

## [Jul 16, 12:44 AM] — Kartigan
Committed: Added ratios 2:1 + 3:2, CoT run results, and three bug fixes. Did NOT run any
experiment (no new API calls) -- a third model is being added first.

RATIOS ADDED (2:1 = 0.67 share / 3 docs, 3:2 = 0.60 / 5 docs). These fill the 0.50->0.75
gap where behavior actually transitions. entities.json regenerated and diffed against the
committed baseline: 0 mismatches across all 50 entities for the 4 pre-existing ratios, so
run_full_standard.csv stays valid and comparable. Default run scope is now 600 calls per
strategy (6 ratios x 50 x 2 models), up from 400.
DO NOT ADD 5:1/6:1: make_documents assigns one distinct DOC_STYLE per doc and there are only
5 styles (4:1 already uses all 5). A 6-doc ratio now raises a clear ValueError (guard added)
instead of a bare IndexError. Adding DOC_STYLES would shift the main RNG stream and
regenerate EVERY entity differently -- invalidating all collected data. Don't.

BUG FIXES:
 1. call_with_backoff only caught google.genai exception types. Network-layer failures
    (ConnectTimeout etc.) come from the underlying httpx transport and bypassed retry
    entirely, failing on first hit. This cost 40 Gemini calls during the Jul 15 WiFi
    outage. Now retries httpx.TimeoutException / NetworkError / RemoteProtocolError.
 2. visualizations/common.py MAJORITY_SHARE had no entry for 2:1 / 3:2. load_results()
    maps that column, so those ratios would have become NaN and been SILENTLY DROPPED from
    every figure -- no error, just missing data. Added, plus RATIO_ORDER / CONFLICT_RATIOS.
 3. MODEL_LABELS said "Gemini 3.1 Flash-Lite" -- stale from the quota-downgrade era. We are
    on gemini-3.5-flash; every figure would have carried the wrong model name into the
    Research Brief. Fixed. Also added anthropic label + a distinct color (plot scripts fall
    back to gray, which collides with the OTHER category) for the incoming third model.

COT RUN (results/run_full_cot.csv) -- COMPLETED, committed, but READ THIS BEFORE USING:
72/400 calls failed as network timeouts during the WiFi outage. The loss is NOT random: it
hit a contiguous block E029-E038, and ALL 72 failures are domain=general (0 banking). That
shifts the CoT sample to 49% banking vs the standard run's 40%, which CONFOUNDS any direct
std-vs-cot comparison and would badly bias H3 (domain rigidity). Re-run CoT before using it
for H3. Fix #1 above should prevent a recurrence.

H2 FINDING (computed on a MATCHED sample -- 322 cells scored in BOTH runs, which removes the
domain confound above; the effect holds):
  ratio  gemini std->cot MAJ%      gpt-5-mini std->cot MAJ%
  2:2    28% -> 33%  (+5)          32% -> 24%  (-7)
  3:1    57% -> 70%  (+13)         100% -> 95% (-5)
  4:1    55% -> 80%  (+25)         100% -> 93% (-7)
H2 predicted CoT would INCREASE flagging/compromise and reduce majority-following. That is
weakly true for gpt-5-mini (-5 to -7) but REVERSED for Gemini: CoT makes Gemini follow the
majority MORE (+25 at 4:1). Mechanism: under standard prompting Gemini hedges (34% COM at
4:1); CoT makes it stop hedging and commit -- and it commits to the majority. Worth writing
up as a per-model result rather than a single H2 verdict.
Files: data/generate_dataset.py, data/entities.json, harness/run_experiment.py,
visualizations/common.py, results/run_full_cot.csv, UPDATES.md
Status: No experiment re-run yet (intentional -- third model pending). Once the third model
is wired in: add it to MODEL_LABELS/MODEL_COLORS in visualizations/common.py, then re-run
both strategies over the full 6-ratio dataset (600 calls/strategy).

## [Jul 16, 1:20 AM] — Kartigan

Committed: Token counter. Every run now prints a per-model token + cost summary when it
finishes, and `python harness/token_report.py` tallies any/all saved result CSVs after the
fact (pure accounting, makes no API calls). `--by-file` gives a per-file breakdown. Mock rows
are excluded/unpriced -- they never hit an API and their token counts are chars/4 guesses.

USAGE TO DATE (measured from the prompt_tokens/completion_tokens the harness already
recorded -- these counts are exact):
  model                  calls    input    output     total
  gemini-3.5-flash         400  119,607    60,285   179,892
  gpt-5-mini               412  120,491   204,001   324,492
  gemini-3.1-flash-lite     12    3,427       310     3,737
  TOTAL                    824  243,525   264,596   508,121
Estimated cost ~$1.73 total (~$0.72 gemini-3.5-flash + ~$1.01 gpt-5-mini).

PRICING CAVEAT: only gemini-3.5-flash ($1.50/$9.00 per 1M) and claude-haiku-4-5 are verified
rates. I could NOT confirm a published rate for the gpt-5-mini model id specifically, so its
entry in PRICING is the nearest comparable small GPT-5-class tier ($0.75/$4.50) and is marked
UNVERIFIED (flagged with * in the counter output). Do not quote gpt-5-mini cost in the brief
without checking the Azure pricing page. Token counts are unaffected by this.

COST DRIVER -- OUTPUT tokens dominate, because both models reason before answering. gpt-5-mini
emits more output than input (204K out vs 120K in). CoT is far more expensive than standard:
gpt-5-mini output was 161,908 (CoT) vs 39,350 (standard) = 4.1x; gemini 46,178 vs 14,107 =
3.3x. Budget accordingly.
BUDGET FOR THE PLANNED RE-RUN: 6 ratios x 50 entities x 2 strategies x 3 models = 1,800 calls,
vs the 800 that cost ~$1.73 -> roughly $4-5. Still trivial, but CoT is the expensive half.
Files: harness/run_experiment.py, harness/token_report.py, UPDATES.md
Status: Counter is live. Third model still pending; no experiment re-run yet.

## [Jul 16, 12:51 PM] — Kartigan

Committed: Per-model reasoning-depth controls. Not run -- code only, per instruction.

VERIFIED THE OPENROUTER SLUG before using it (same discipline as the earlier Gemini 3.5
Flash mistake): Claude Haiku 4.5's OpenRouter model id is `anthropic/claude-haiku-4.5`
($1/$5 per 1M, matches the already-verified rate in PRICING -- good cross-check).

CONFIRMED: DeepSeek V4 Flash and Claude Haiku 4.5 both go through the SAME OpenRouter slot
(OPENROUTER_API_KEY, OpenAI-compatible client at base_url=openrouter.ai/api/v1) -- there is
NO separate Anthropic key involved when routing Claude through OpenRouter. Swapping
OPENROUTER_MODEL between the two is a pure config change.

Added `openrouter_reasoning_param(model_id)`: OpenRouter exposes one unified `reasoning`
field it translates server-side per provider -- reasoning.max_tokens for Anthropic models
(-> Claude's native budget_tokens), reasoning.effort for everything else (DeepSeek, etc).
Auto-detects "claude"/"anthropic" in the model id, so switching OPENROUTER_MODEL needs no
other code change. Passed via extra_body since `reasoning` isn't part of the OpenAI SDK's
typed schema.
  DeepSeek V4 Flash (via OpenRouter):     reasoning: {"effort": "high"}
  Claude Haiku 4.5 (via OpenRouter):      reasoning: {"max_tokens": 2048}

Added native thinking_level to call_gemini() (separate mechanism -- Gemini runs on its own
direct google-genai SDK path, not through OpenRouter). VERIFIED against the installed SDK's
pydantic model_fields (not assumed) that GenerateContentConfig.thinking_config accepts a
ThinkingConfig with a thinking_level field, and that ThinkingLevel.MEDIUM is a real enum
member -- confirmed by constructing the actual object, not just reading source.
  Gemini 3.5 Flash: thinking_config=ThinkingConfig(thinking_level=ThinkingLevel.MEDIUM)

New constants: GEMINI_THINKING_LEVEL="MEDIUM", OPENROUTER_ANTHROPIC_BUDGET_TOKENS=2048,
OPENROUTER_DEFAULT_EFFORT="high" -- all near the top of the file, easy to tune.
Files: harness/run_experiment.py, UPDATES.md
Status: Code verified to construct correctly (checked openrouter_reasoning_param() output
for both models + built a real Gemini ThinkingConfig object) but NOT run against a live API
yet -- do a small fire test before the next full run to confirm these params are actually
accepted server-side, not just constructed client-side without error.
