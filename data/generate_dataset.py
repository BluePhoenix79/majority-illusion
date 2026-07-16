"""Generate the synthetic entity dataset for the Majority Illusion experiment.

Produces data/entities.json: ~50 fictional entities (subset banking-themed),
each with a factual question, two conflicting answer values (majority vs
minority), and per-ratio document sets:

  4:0  -> 4 docs, all supporting the majority value (control)
  3:1  -> 3 majority docs, 1 minority doc
  2:2  -> 2 majority docs, 2 minority docs
  4:1  -> 4 majority docs, 1 minority doc

Fully deterministic (fixed seed) so the whole team regenerates identical data.

Usage:
    python data/generate_dataset.py            # writes data/entities.json
"""

import json
import random
from pathlib import Path

SEED = 20260714
N_BANKING = 20
N_GENERAL = 30

RATIOS = {
    "4:0": (4, 0),
    "3:1": (3, 1),
    "2:2": (2, 2),
    "4:1": (4, 1),
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


def main():
    rng = random.Random(SEED)
    entities = []

    specs = []
    used_names = set()
    for i in range(N_BANKING):
        while True:
            name = f"{rng.choice(BANKING_FIRST)} {rng.choice(BANKING_SECOND)}"
            if name not in used_names:
                used_names.add(name)
                break
        specs.append((name, "banking", "regional financial institution"))
    for i in range(N_GENERAL):
        while True:
            name = f"{rng.choice(GENERAL_FIRST)} {rng.choice(GENERAL_SECOND)}"
            if name not in used_names:
                used_names.add(name)
                break
        specs.append((name, "general", "mid-size company"))

    for idx, (name, domain, domain_desc) in enumerate(specs, start=1):
        if domain == "banking":
            attribute, q_tmpl, claim_tmpl, value_gen = BANKING_ATTRIBUTES[idx % len(BANKING_ATTRIBUTES)]
            account = rng.choice(ACCOUNT_TYPES)
            act = rng.choice(ACT_NAMES)
            q_formatted = q_tmpl.format(name=name, account=account, act=act)
            claim_tmpl_formatted = claim_tmpl.format(name="{name}", value="{value}", account=account, act=act)
        else:
            attribute, q_tmpl, claim_tmpl, value_gen = GENERAL_ATTRIBUTES[idx % len(GENERAL_ATTRIBUTES)]
            q_formatted = q_tmpl.format(name=name)
            claim_tmpl_formatted = claim_tmpl

        maj_value, min_value = value_gen(rng)
        entities.append({
            "entity_id": f"E{idx:03d}",
            "entity_name": name,
            "domain": domain,
            "attribute": attribute,
            "question": q_formatted,
            "majority_value": maj_value,
            "minority_value": min_value,
            "documents": make_documents(rng, name, domain_desc, claim_tmpl_formatted,
                                        maj_value, min_value),
        })

    out_path = Path(__file__).parent / "entities.json"
    out_path.write_text(json.dumps({"seed": SEED, "ratios": list(RATIOS),
                                    "entities": entities}, indent=2))
    print(f"Wrote {len(entities)} entities to {out_path}")


if __name__ == "__main__":
    main()
