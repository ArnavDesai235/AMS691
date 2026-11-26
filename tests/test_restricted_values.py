from types import SimpleNamespace as NS

from corpus_gen.verbalize import _restricted_value_violations, _tier_leaks


def _ft():
    e = NS(entity_id="E195", canonical_name="Project Barkil")
    f_status = NS(fact_id="F1", subject="E195", tier="team_b", is_canary=False, object={"literal": "on track"})
    f_qtr = NS(fact_id="F2", subject="E195", tier="team_a", is_canary=False, object={"literal": "Q2 FY28"})
    f_pub = NS(fact_id="F3", subject="E195", tier="public", is_canary=False, object={"literal": "in pilot"})
    return NS(entities=[e], facts=[f_status, f_qtr, f_pub])


def _doc(tier):
    return NS(tier=tier, entities=["E195"], fact_ids=["F2"])


def test_tier_leaks():
    assert _tier_leaks("team_b", "team_a") and _tier_leaks("team_a", "public")
    assert not _tier_leaks("team_a", "team_a") and not _tier_leaks("team_a", "exec")
    assert not _tier_leaks("public", "public")


def test_flags_filler_restatement_in_lower_tier_doc():
    t = "Project Barkil is currently on track for a Q2 FY28 completion."
    assert len(_restricted_value_violations(_ft(), _doc("team_a"), t)) == 1


def test_clean_text_and_home_tier_and_planned_value_pass():
    t = "Project Barkil is targeting completion in Q2 FY28."
    assert _restricted_value_violations(_ft(), _doc("team_a"), t) == []
    assert _restricted_value_violations(_ft(), _doc("team_b"), "Project Barkil is on track.") == []
    d = NS(tier="team_a", entities=["E195"], fact_ids=["F1"])   # planned itself: coincidence, allowed
    assert _restricted_value_violations(_ft(), d, "Project Barkil is on track.") == []
