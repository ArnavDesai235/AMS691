"""corpus_gen.repair: deterministic name expansion and canary de-duplication."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from corpus_gen.repair import dedupe_codes, expand_names

PERSON = {"Vadesis Gummenbel": ["Vadesis", "Gummenbel"]}
TEAM = {"Talhuor Integration Team": ["Talhuor"]}
PROJECT = {"Project Kistos": ["Kistos"]}


def test_first_name_and_surname_expanded():
    t, log = expand_names("Vadesis Gummenbel leads. Vadesis said Gummenbel's plan works.", PERSON)
    assert t == "Vadesis Gummenbel leads. Vadesis Gummenbel said Vadesis Gummenbel's plan works."
    assert len(log) == 2


def test_full_names_untouched():
    s = "Vadesis Gummenbel met Vadesis Gummenbel."
    assert expand_names(s, PERSON) == (s, [])


def test_partial_name_absorbed_not_duplicated():
    t, _ = expand_names("The Talhuor Integration team and the Talhuor team met.", TEAM)
    assert t == "The Talhuor Integration Team and the Talhuor Integration Team met."
    t, _ = expand_names("The project Kistos and Kistos.", PROJECT)
    assert t == "The Project Kistos and Project Kistos."


def test_other_words_not_touched():
    t, log = expand_names("Vadesisian notes stay.", PERSON)
    assert t == "Vadesisian notes stay." and log == []


CODES = {"SP-9062-V": "Pethisa Hebroren"}


def test_extra_codes_removed_anchor_kept():
    body = ("Pethisa Hebroren holds budget code SP-9062-V. Spend is tracked.\n\n"
            "ACTION ITEMS:\n- Finance to confirm budget code SP-9062-V (SP-9062-V).")
    title, out, log = dedupe_codes("Budget review", body, CODES)
    assert out.count("SP-9062-V") == 1
    assert "Pethisa Hebroren holds budget code SP-9062-V." in out
    assert "budget code noted earlier" in out and "()" not in out
    assert log == ["code SP-9062-V: kept 1 of 3 occurrences"]


def test_code_in_title_removed_when_body_anchored():
    title, body, _ = dedupe_codes("Review of SP-9062-V", "Pethisa Hebroren owns SP-9062-V.", CODES)
    assert "SP-9062-V" not in title and body.count("SP-9062-V") == 1


def test_no_anchor_left_for_llm():
    body = "The code SP-9062-V is set. Use SP-9062-V."
    assert dedupe_codes("T", body, CODES) == ("T", body, [])


def test_single_anchored_code_untouched():
    body = "Pethisa Hebroren owns SP-9062-V."
    assert dedupe_codes("T", body, CODES) == ("T", body, [])


from corpus_gen.repair import insert_facts


def test_fact_inserted_after_anchor_paragraph():
    body = "Intro text.\n\nThe Dankenor Analytics Unit runs reports.\n\nClosing."
    out, log = insert_facts(body, [("The Dankenor Analytics Unit has 38 people.", "Dankenor Analytics Unit")])
    assert out == ("Intro text.\n\nThe Dankenor Analytics Unit runs reports.\n\n"
                   "The Dankenor Analytics Unit has 38 people.\n\nClosing.")
    assert log == ["inserted fact: The Dankenor Analytics Unit has 38 people."]


def test_fact_appended_when_anchor_absent():
    out, _ = insert_facts("Only text.", [("X has 16 people.", "X Group")])
    assert out == "Only text.\n\nX has 16 people."


def test_fix_prompt_gives_exact_length_shortfall():
    from corpus_gen.verbalize import build_fix_prompt
    p = build_fix_prompt("BASE", "DRAFT", ["length 524 words; target about 762 (allowed 571-1028)"])
    assert "524 words" in p and "at least 571" in p and "about 238 words" in p
    p = build_fix_prompt("BASE", "DRAFT", ["length 1100 words; target about 762 (allowed 571-1028)"])
    assert "at most 1028" in p


def test_repair_gold_expands_names_and_appends_missing_fact():
    from types import SimpleNamespace as NS
    from corpus_gen.queries import repair_gold
    annex = NS(entity_id="e1", canonical_name="Tegurine Annex", type="site")
    ft = NS(entities=[annex])
    fact = NS(mentions=["e1"], object={"literal": "480 seats"},
              sentence="The Tegurine Annex has capacity for 480 seats.")
    out = repair_gold(ft, [fact], "Tegurine holds a 480-seat capacity.")
    assert out == ("Tegurine Annex holds a 480-seat capacity. "
                   "The Tegurine Annex has capacity for 480 seats.")
    ok = "The Tegurine Annex has capacity for 480 seats."
    assert repair_gold(ft, [fact], ok) == ok


from corpus_gen.repair import anchor_codes

ANCHOR = {"EP-8854-N": ("Maneor Tismean", "Maneor Tismean's vault key ID is EP-8854-N.")}


def test_unanchored_code_replaced_by_planned_sentence():
    body = ("Maneor Tismean leads the review.\n\nATTENDEES: finance lead (key EP-8854-N)\n\n"
            "Next steps follow.")
    title, out, log = anchor_codes("Review", body, ANCHOR)
    assert out.count("EP-8854-N") == 1
    assert "Maneor Tismean leads the review.\n\nManeor Tismean's vault key ID is EP-8854-N." in out
    assert log == ["code EP-8854-N: re-anchored with planned sentence"]


def test_missing_code_inserted():
    _, out, _ = anchor_codes("T", "Maneor Tismean leads.", ANCHOR)
    assert out == "Maneor Tismean leads.\n\nManeor Tismean's vault key ID is EP-8854-N."


def test_anchored_code_untouched():
    body = "Maneor Tismean holds key EP-8854-N."
    assert anchor_codes("T", body, ANCHOR) == ("T", body, [])


def test_len_bounds_are_whole_words():
    from types import SimpleNamespace as NS
    from corpus_gen.verbalize import _len_bounds
    cfg = NS(docs=NS(words_min=300, words_max=1200))
    assert _len_bounds(cfg, NS(target_words=591)) == (443, 797)   # 443.25 -> 443
