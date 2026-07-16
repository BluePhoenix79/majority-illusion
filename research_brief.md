# Research Brief: Majority Illusion in Retrieval-Augmented Generation (RAG)

This document outlines the core ideas, hypotheses, experimental design, and methodologies for the Majority Illusion study.

---

## 1. Core Concept: The "Majority Illusion" in RAG

In network science, the **majority illusion** describes a phenomenon where a globally rare behavior or opinion appears to be the prevailing norm because it is overrepresented in the local connections of individuals.

In **Retrieval-Augmented Generation (RAG)**, a similar illusion occurs within the retrieved context. If a RAG pipeline retrieves multiple documents containing conflicting information, the LLM must synthesize the correct answer. When the retrieved set is skewed—containing several documents stating fact A (the majority consensus) and only one document stating fact B (the minority fact)—the LLM may succumb to the "majority illusion." It might confidently output fact A simply because of its frequency in the context, even if fact B represents the true, updated, or correct information.

This study systematically measures how document ratios, prompting strategies, model sizes, and document ordering influence this illusion.

---

## 2. Experimental Hypotheses

*   **Hypothesis 1 (Majority Dominance):** Under standard prompting, as the ratio of majority-to-minority documents increases (e.g., from `2:2` to `3:1` and `4:1`), models are significantly more likely to select the majority claim (`MAJ`) and report higher confidence.
*   **Hypothesis 2 (Chain-of-Thought Mitigation):** Chain-of-Thought (`cot`) prompting will significantly increase the rate of conflict flagging (`FLAG`) or compromise answers (`COM`), and lower the stated confidence compared to standard prompting. The explicit reasoning trace forces the model to evaluate discrepancies rather than rely on consensus frequency.
*   **Hypothesis 3 (Domain Rigidity):** Conflicting facts in the banking domain (e.g., numeric interest rates, fees, caps) will yield higher conflict flagging (`FLAG`) rates and lower confidence than general corporate facts (e.g., founding years, employee counts), as LLMs exhibit greater caution with sensitive financial parameters.
*   **Hypothesis 4 (Position Bias / Primacy-Recency):** The position of the minority document in the retrieved context will skew the outcome. Placement at the beginning (Primacy) or end (Recency) of the document block will result in higher rates of minority selection (`MIN`) compared to when it is positioned in the middle.

---

## 3. Experimental Variables

### Independent Variables (IVs)
1.  **Document Ratio:** The ratio of majority-to-minority documents in the prompt:
    *   `4:0` (Control: 4 majority documents, 0 minority documents)
    *   `3:1` (Moderate conflict: 3 majority, 1 minority)
    *   `2:2` (Tied conflict: 2 majority, 2 minority)
    *   `4:1` (High noise conflict: 4 majority, 1 minority)
2.  **Prompting Strategy:**
    *   `standard`: Directly requests a structured JSON answer.
    *   `cot`: Directs the model to think step-by-step and write down its reasoning process before outputting the JSON block.
3.  **Model Architectures:**
    *   **Model A: Google Gemini 3.1 Flash-Lite** (Small, thinking-capable model, run on Google's free tier). *Note: Upgraded from 3.5 Flash due to strict 20 req/day quota limitations.*
    *   **Model B: OpenAI GPT-5 Mini** (Azure OpenAI deployment, representing advanced reasoning capabilities).
    *   **Model C: Anthropic Claude 4.5 Haiku** (Drop-in replacement for the retired 3.5 Haiku).
4.  **Domain:**
    *   `banking`: Attributes like `interest_rate`, `monthly_fee`, `lending_cap`, and `overdraft_limit`.
    *   `general`: Attributes like `founding_year`, `headquarters`, `ceo`, and `employee_count`.
5.  **Document Position:** Randomly shuffled in each trial to counter position bias (recorded in the `doc_positions` column).

### Dependent Variables (DVs)
1.  **Parsed Answer Category (`auto_category`):**
    *   `MAJ`: Answer matches the majority value.
    *   `MIN`: Answer matches the minority value.
    *   `COM`: Compromise or mention of both values.
    *   `FLAG`: Model explicitly identifies the conflict/discrepancy and declines to make a single choice.
    *   `OTHER`: Parse failure or unrelated response.
2.  **Stated Confidence:** Model self-reported confidence score on a scale of 1–5 (elicited in the prompt).

---

## 4. Methodology & Implementation

1.  **Dataset Generation ([generate_dataset.py](file:///c:/Users/Pranav%20Sai/majority-illusion/data/generate_dataset.py)):**
    Generates 50 synthetic fictional entities (20 banking, 30 general) with deterministic seeds, creating unique questions and a pool of conflicting documents at all ratios.
2.  **Query Harness ([run_experiment.py](file:///c:/Users/Pranav%20Sai/majority-illusion/harness/run_experiment.py)):**
    Reads the entities, shuffles documents for each trial to neutralize position bias, queries selected models, and records detailed token usage, raw outputs, and metadata to a CSV.
3.  **Scoring & Evaluation ([score_results.py](file:///c:/Users/Pranav%20Sai/majority-illusion/analysis/score_results.py)):**
    Uses a rule-based rubric classifier to categorize answers and generate statistics.
