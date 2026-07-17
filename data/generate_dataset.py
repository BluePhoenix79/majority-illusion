"""Generate the synthetic entity dataset for the Majority Illusion experiment.

Produces data/entities.json: 75 fictional entities (30 banking-themed and 45
general), each with a factual question, two conflicting answer values
(majority vs minority), and per-ratio document sets:

  4:0  -> 4 docs, all supporting the majority value (control)   majority share 1.00
  3:1  -> 3 majority docs, 1 minority doc                       majority share 0.75
  2:2  -> 2 majority docs, 2 minority docs                      majority share 0.50
  4:1  -> 4 majority docs, 1 minority doc                       majority share 0.80
  2:1  -> 2 majority docs, 1 minority doc                       majority share 0.67
  3:2  -> 3 majority docs, 2 minority docs                      majority share 0.60

2:1 and 3:2 fill the gap between 2:2 (0.50) and 3:1 (0.75) -- the region where
model behavior actually transitions from "flags the conflict" to "follows the
majority". Adding them is purely additive: the per-ratio document order is
seeded independently (order_rng below), and the style/city draws happen before
the ratio loop, so existing ratios regenerate byte-identically.

CAP: total docs per ratio must be <= len(DOC_STYLES) (5), since make_documents
assigns one distinct style per doc. That rules out 5:1 (6 docs) and beyond
without adding more DOC_STYLES entries -- and adding styles WOULD shift the
main RNG stream and regenerate every entity differently.

Fully deterministic (fixed seed) so the whole team regenerates identical data.

Usage:
    python data/generate_dataset.py            # writes data/entities.json
"""

import json
import random
from pathlib import Path

SEED = 20260714
N_BANKING = 30
N_GENERAL = 45

# Preserve the original 50 entities exactly, then generate the 25-entity
# expansion from a separate deterministic RNG stream. Simply increasing the
# original loop counts would advance the shared RNG before entity details are
# generated and silently change every existing question, value, and document.
BASE_N_BANKING = 20
BASE_N_GENERAL = 30
EXPANSION_SEED = f"{SEED}|entity-expansion|{N_BANKING + N_GENERAL}"

RATIOS = {
    "4:0": (4, 0),
    "3:1": (3, 1),
    "2:2": (2, 2),
    "4:1": (4, 1),
    "2:1": (2, 1),
    "3:2": (3, 2),
}

# ---------------------------------------------------------------------------
# Fictional entity name parts (chosen to avoid real-world collisions)

BANKING_FIRST = ["Meridian Vale", "Corvane", "Ashfell", "Northgale", "Tessamir",
                 "Ferrowick", "Quillbrook", "Solyndra Bay", "Marridge", "Veltra"]
BANKING_SECOND = ["Savings Bank", "Credit Union", "Trust Company", "Mutual Bank",
                  "Financial Group", "Bancorp"]

GENERAL_FIRST = ["Auralith", "Brindlewood", "Cassoveer", "Drummelin", "Eskarion",
                 "Fennmarrow", "Glimwater", "Halverstone", "Ithracel", "Jorvander",
                 "Kelvarris", "Lumeport", "Mossgrave", "Nerrivane", "Ostelbrook"]
GENERAL_SECOND = ["Industries", "Laboratories", "Logistics", "Observatory",
                  "Institute", "Foods", "Robotics", "Textiles", "Energy",
                  "Aerospace", "Publishing", "Analytics"]

CITIES = ["Dunmore Falls", "Kettlewick", "Port Halloran", "Sarnbridge",
          "New Aldery", "Wrenfield", "Coldbarrow", "Marlow Junction",
          "Estabrook", "Vantry Heights", "Redgate Hollow", "Silvermoor"]

FOUNDER_FIRST = ["Elara", "Tobias", "Marisol", "Hendrick", "Priya", "Casimir",
                 "Odette", "Rowan", "Sylvie", "Barnaby", "Ingrid", "Theodric"]
FOUNDER_LAST = ["Vantrell", "Okonkwo-Reyes", "Halloway", "Brandsgard", "Chen-Marlowe",
                "Fitzwarren", "Delacroix-Nunn", "Yamaguchi-Pryce", "Sorenfeld", "Aberlane"]

DOC_STYLES = [
    ("regional_news", "Regional Business Journal",
     "{city_line} {name} {claim_sentence} A spokesperson declined to comment further on expansion plans."),
    ("encyclopedia", "Open Encyclopedia Project",
     "{name} is a {domain_desc}. {claim_sentence} The organization remains privately held."),
    ("industry_report", "Sector Analysis Quarterly",
     "In our latest review of the sector, {name} stood out among mid-size players. {claim_sentence} Analysts rate its outlook as stable."),
    ("press_release", "Company Newswire",
     "{name} today announced continued growth across its core operations. {claim_sentence} Leadership credited steady demand for the results."),
    ("forum_post", "IndustryTalk Forums",
     "Was reading up on {name} for a project. {claim_sentence} Figured others here might find that useful."),
]

# (attribute, question template, claim template, value generator a/b)
def year_pair(rng):
    a = rng.randint(1948, 2011)
    b = a + rng.choice([-14, -9, 9, 13, 17])
    return str(a), str(b)

def city_pair(rng):
    a, b = rng.sample(CITIES, 2)
    return a, b

def founder_pair(rng):
    fa, fb = rng.sample(FOUNDER_FIRST, 2)
    la, lb = rng.sample(FOUNDER_LAST, 2)
    return f"{fa} {la}", f"{fb} {lb}"

def employees_pair(rng):
    a = rng.choice([240, 380, 520, 760, 1150, 1900, 2400, 3200])
    b = int(a * rng.choice([0.4, 0.55, 1.8, 2.5]))
    return str(a), str(b)

GENERAL_ATTRIBUTES = [
    ("founding_year",
     "In what year was {name} founded?",
     "{name} was founded in {value}.",
     year_pair),
    ("headquarters_city",
     "In which city is {name} headquartered?",
     "{name} is headquartered in {value}.",
     city_pair),
    ("founder_name",
     "Who founded {name}?",
     "{name} was founded by {value}.",
     founder_pair),
    ("employee_count",
     "Approximately how many employees does {name} have?",
     "{name} employs approximately {value} people.",
     employees_pair),
]

ACCOUNT_TYPES = [
    "FlexPrime Savings", "GoldShield Checking", "SecureYield Money Market",
    "ApexGrowth Savings", "EcoSaver Checking"
]

ACT_NAMES = [
    "Orcana Reserve Act", "Veltra Lending Policy", "Corvane Compliance Directive",
    "Ashfell Charter", "Tessamir Capital Accord"
]

def interest_rate_pair(rng):
    a = rng.choice([1.5, 2.25, 3.5, 4.25, 5.75, 6.5])
    b = rng.choice([x for x in [1.5, 2.25, 3.5, 4.25, 5.75, 6.5] if x != a])
    return f"{a}%", f"{b}%"

def monthly_fee_pair(rng):
    a = rng.choice([5, 12, 15, 25, 35])
    b = rng.choice([x for x in [5, 12, 15, 25, 35] if x != a])
    return f"${a}", f"${b}"

def lending_cap_pair(rng):
    a = rng.choice([10, 15, 20, 25, 30])
    b = rng.choice([x for x in [10, 15, 20, 25, 30] if x != a])
    return f"{a}%", f"{b}%"

def overdraft_pair(rng):
    a = rng.choice([100, 250, 500, 1000])
    b = rng.choice([x for x in [100, 250, 500, 1000] if x != a])
    return f"${a}", f"${b}"

BANKING_ATTRIBUTES = [
    ("interest_rate",
     "What is the annual interest rate for the {account} account at {name}?",
     "The annual interest rate for the {account} account at {name} is {value}.",
     interest_rate_pair),
    ("monthly_fee",
     "What is the monthly maintenance fee for the {account} account at {name}?",
     "The monthly maintenance fee for the {account} account at {name} is {value}.",
     monthly_fee_pair),
    ("lending_cap",
     "What is the maximum lending cap specified by the {act} at {name}?",
     "The {act} at {name} caps institutional lending at {value}.",
     lending_cap_pair),
    ("overdraft_limit",
     "What is the default overdraft protection limit for the {account} account at {name}?",
     "The default overdraft protection limit for the {account} account at {name} is {value}.",
     overdraft_pair),
]


def make_documents(rng, name, domain_desc, claim_template, maj_value, min_value):
    """Build 5 distinct document templates; return functions of value assignment."""
    # Each doc in a ratio gets its own distinct style, so no ratio may need more
    # docs than there are styles. Checked up front: otherwise this fails later as
    # a bare IndexError from styles[n_maj + i], which is hard to trace back here.
    _worst = max(n_maj + n_min for n_maj, n_min in RATIOS.values())
    if _worst > len(DOC_STYLES):
        raise ValueError(
            f"RATIOS needs {_worst} docs for its largest ratio but only "
            f"{len(DOC_STYLES)} DOC_STYLES exist. Add more DOC_STYLES entries -- "
            f"but note that changes the main RNG stream, so every entity's name, "
            f"values, and documents will regenerate differently."
        )
    styles = rng.sample(DOC_STYLES, 5)
    city = rng.choice(CITIES)

    def render(style, value):
        style_id, source, template = style
        claim = claim_template.format(name=name, value=value)
        text = template.format(
            name=name,
            claim_sentence=claim,
            city_line=f"{city.upper()} —",
            domain_desc=domain_desc,
        )
        return {"source": source, "style": style_id, "text": text}

    docs_by_ratio = {}
    for ratio, (n_maj, n_min) in RATIOS.items():
        docs = [render(styles[i], maj_value) for i in range(n_maj)]
        docs += [render(styles[n_maj + i], min_value) for i in range(n_min)]
        # Shuffle so the minority doc isn't always last (position bias control),
        # deterministically per entity+ratio.
        order_rng = random.Random(f"{SEED}-{name}-{ratio}")
        order_rng.shuffle(docs)
        docs_by_ratio[ratio] = docs
    return docs_by_ratio


def make_specs(rng, n_banking, n_general, used_names):
    """Create unique fictional entity specifications with a supplied RNG."""
    specs = []
    for _ in range(n_banking):
        while True:
            name = f"{rng.choice(BANKING_FIRST)} {rng.choice(BANKING_SECOND)}"
            if name not in used_names:
                used_names.add(name)
                break
        specs.append((name, "banking", "regional financial institution"))
    for _ in range(n_general):
        while True:
            name = f"{rng.choice(GENERAL_FIRST)} {rng.choice(GENERAL_SECOND)}"
            if name not in used_names:
                used_names.add(name)
                break
        specs.append((name, "general", "mid-size company"))
    return specs


def make_entity(rng, idx, spec):
    """Render one entity and all ratio-specific documents."""
    name, domain, domain_desc = spec
    if domain == "banking":
        attribute, q_tmpl, claim_tmpl, value_gen = BANKING_ATTRIBUTES[
            idx % len(BANKING_ATTRIBUTES)
        ]
        account = rng.choice(ACCOUNT_TYPES)
        act = rng.choice(ACT_NAMES)
        q_formatted = q_tmpl.format(name=name, account=account, act=act)
        claim_tmpl_formatted = claim_tmpl.format(
            name="{name}", value="{value}", account=account, act=act
        )
    else:
        attribute, q_tmpl, claim_tmpl, value_gen = GENERAL_ATTRIBUTES[
            idx % len(GENERAL_ATTRIBUTES)
        ]
        q_formatted = q_tmpl.format(name=name)
        claim_tmpl_formatted = claim_tmpl

    maj_value, min_value = value_gen(rng)
    return {
        "entity_id": f"E{idx:03d}",
        "entity_name": name,
        "domain": domain,
        "attribute": attribute,
        "question": q_formatted,
        "majority_value": maj_value,
        "minority_value": min_value,
        "documents": make_documents(
            rng, name, domain_desc, claim_tmpl_formatted,
            maj_value, min_value,
        ),
    }


def main():
    entities = []
    used_names = set()

    # Legacy stream: matches the original 20-banking/30-general generator.
    base_rng = random.Random(SEED)
    base_specs = make_specs(
        base_rng, BASE_N_BANKING, BASE_N_GENERAL, used_names
    )
    for idx, spec in enumerate(base_specs, start=1):
        entities.append(make_entity(base_rng, idx, spec))

    # Independent expansion stream: appends 10 banking + 15 general entities
    # without perturbing the original 50.
    expansion_rng = random.Random(EXPANSION_SEED)
    expansion_specs = make_specs(
        expansion_rng,
        N_BANKING - BASE_N_BANKING,
        N_GENERAL - BASE_N_GENERAL,
        used_names,
    )
    for idx, spec in enumerate(expansion_specs, start=len(entities) + 1):
        entities.append(make_entity(expansion_rng, idx, spec))

    if len(entities) != N_BANKING + N_GENERAL:
        raise RuntimeError("entity count does not match configured target")
    if len({entity["entity_name"] for entity in entities}) != len(entities):
        raise RuntimeError("duplicate fictional entity names generated")

    out_path = Path(__file__).parent / "entities.json"
    out_path.write_text(json.dumps({"seed": SEED, "ratios": list(RATIOS),
                                    "entities": entities}, indent=2))
    print(f"Wrote {len(entities)} entities to {out_path}")


if __name__ == "__main__":
    main()
