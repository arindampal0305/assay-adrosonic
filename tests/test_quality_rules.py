"""Every SRS 9.2 rule, driven one at a time against a column built to trigger it.

Two things each test asserts, because only the first is usually tested and the
second is where the real risk lives:

1. the rule fires on the thing it is for, and
2. the rule does **not** fire on the lookalike it must tolerate.

The second half is not padding. `FRAME_CONSTRUCTION_TERMS` once contained the
substring "combustible", which matches "Masonry Non-Combustible (ISO 4)" — a
phrase appearing 154 times in `SOV_Q8B3` — and would have produced a few hundred
confident false DQ-14 findings about masonry buildings being wood-frame towers. No
amount of positive testing would have caught that. So every rule here is also
shown a value it must leave alone.

Where a frame is constructed rather than read from a real workbook it is built by
`frame_from_columns`, which routes through the same cleaner and typed-frame
builder as production. DQ-16 can only be tested this way, and the reason is
recorded in that test rather than left implicit.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from backend.agents.quality import completeness, recommend, rules, schema, structural
from backend.agents.quality.findings import Finding
from backend.agents.quality.frame import frame_from_columns
from backend.agents.quality.models import OP_WHITELIST, RULE_CATALOGUE
from backend.agents.quality.vocab import FRAME_EXCLUSIONS


def _rows(findings: list[Finding], rule: str) -> list[int]:
    return sorted(f.row for f in findings if f.rule == rule)


def _only(findings: list[Finding], rule: str) -> Finding:
    matching = [f for f in findings if f.rule == rule]
    assert len(matching) == 1, f"expected one {rule}, got {[f.row for f in matching]}"
    return matching[0]


# --------------------------------------------------------------------------
# DQ-01 — missing values
# --------------------------------------------------------------------------
def test_dq01_flags_empty_and_reported_absences_but_not_values():
    frame = frame_from_columns(
        {"Building Value": [100.0, None, "", "N/A", 250.0, "  "]}
    )
    found = rules.dq01_missing_values(frame)
    assert _rows(found, "DQ-01") == [3, 4, 5, 7]
    assert all(f.op is None for f in found), "C-02 forbids inventing a missing value"


def test_dq01_severity_is_higher_on_a_value_field_than_on_county():
    value = rules.dq01_missing_values(frame_from_columns({"Building Value": [None]}))
    county = rules.dq01_missing_values(frame_from_columns({"County": [None]}))
    assert value[0].severity == "High"
    assert county[0].severity != "High"


# --------------------------------------------------------------------------
# DQ-07 — zip against state
# --------------------------------------------------------------------------
def test_dq07_flags_a_zip_allocated_to_another_state():
    frame = frame_from_columns({"Zip": ["90210"], "State": ["MA"]})
    finding = _only(rules.dq07_zip_state_mismatch(frame), "DQ-07")
    assert finding.op is None, "correcting the wrong half would move the risk"
    assert "90210" in finding.before


def test_dq07_accepts_a_zip_in_an_outlying_range_of_its_state():
    """Texas holds 733 and 885 outside its main 750-799 block, and New York holds
    00501 and 06390. A rule built on one contiguous range per state would report
    every one of those as a mismatch."""
    for zip_code, state in (("73301", "TX"), ("88510", "TX"), ("00501", "NY"),
                            ("06390", "NY"), ("71601", "AR"), ("75502", "AR")):
        frame = frame_from_columns({"Zip": [zip_code], "State": [state]})
        assert rules.dq07_zip_state_mismatch(frame) == [], f"{zip_code} is in {state}"


def test_dq07_does_not_fire_on_a_four_digit_zip_in_a_leading_zero_state():
    """'2110' in Massachusetts is a lost leading zero, which is DQ-08's job and a
    fixable one. Reporting it as a state mismatch would send a mechanical fix to a
    human instead."""
    frame = frame_from_columns({"Zip": ["2110"], "State": ["MA"]})
    assert rules.dq07_zip_state_mismatch(frame) == []
    assert _rows(rules.dq08_zip_format(frame), "DQ-08") == [2]


def test_dq07_fires_on_a_four_digit_zip_where_no_zip_starts_with_zero():
    """The same four digits in Texas cannot be a lost leading zero, so it is a
    contradiction rather than a formatting slip — and DQ-08 must stay silent so
    the two rules never both claim the cell."""
    frame = frame_from_columns({"Zip": ["7520"], "State": ["TX"]})
    finding = _only(rules.dq07_zip_state_mismatch(frame), "DQ-07")
    assert finding.op is None
    assert rules.dq08_zip_format(frame) == []


def test_dq07_says_nothing_when_the_state_itself_is_unreadable():
    """An unknown state gives nothing to compare against, and `_prefix_fits`
    returning None must not be read as a mismatch."""
    frame = frame_from_columns({"Zip": ["90210"], "State": ["Atlantis"]})
    assert rules.dq07_zip_state_mismatch(frame) == []


# --------------------------------------------------------------------------
# DQ-08 — zip format
# --------------------------------------------------------------------------
def test_dq08_restores_a_lost_leading_zero_and_computes_the_after():
    frame = frame_from_columns({"Zip": ["2110"], "State": ["MA"]})
    finding = _only(rules.dq08_zip_format(frame), "DQ-08")
    assert finding.op == "zip5"
    assert finding.after == "02110", "FR-DQ-09 requires an engine-computed after"


def test_dq08_truncates_a_zip_plus_four():
    frame = frame_from_columns({"Zip": ["02110-1234"], "State": ["MA"]})
    finding = _only(rules.dq08_zip_format(frame), "DQ-08")
    assert finding.op == "zip5"
    assert finding.after == "02110"


def test_dq08_leaves_a_correct_five_digit_zip_alone():
    frame = frame_from_columns({"Zip": ["02110", "90210"], "State": ["MA", "CA"]})
    assert rules.dq08_zip_format(frame) == []


def test_dq08_reads_the_raw_cell_not_the_typed_one():
    """`'00802'` types to the integer 802. A rule reading the typed frame would
    report a three-digit zip and 'correct' a cell that was already right — which
    is exactly what SOV_B4ID's Virgin Islands zips would have triggered."""
    frame = frame_from_columns({"Zip": ["00802"], "State": ["VI"]})
    assert frame.typed["Zip"].iloc[0] == 802
    assert rules.dq08_zip_format(frame) == []


def test_dq08_proposes_nothing_when_no_five_digit_code_can_be_read():
    frame = frame_from_columns({"Zip": ["see attached"], "State": ["MA"]})
    finding = _only(rules.dq08_zip_format(frame), "DQ-08")
    assert finding.op is None and finding.after is None
    assert finding.evidence["digits_found"] == 0


# --------------------------------------------------------------------------
# DQ-09 — state format
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "written,expected",
    [("Massachusetts", "MA"), ("massachusetts", "MA"), ("V.I.", "VI"),
     ("New York", "NY"), ("Calif.", "CA"), ("tx", "TX")],
)
def test_dq09_converts_a_full_or_dotted_state_name(written, expected):
    frame = frame_from_columns({"State": [written]})
    finding = _only(rules.dq09_state_format(frame), "DQ-09")
    assert finding.op == "state_to_abbrev"
    assert finding.after == expected


def test_dq09_leaves_a_usps_abbreviation_alone():
    frame = frame_from_columns({"State": ["MA", "TX", "VI", "DC"]})
    assert rules.dq09_state_format(frame) == []


def test_dq09_reports_an_unknown_state_without_guessing_one():
    frame = frame_from_columns({"State": ["Atlantis"]})
    finding = _only(rules.dq09_state_format(frame), "DQ-09")
    assert finding.op is None
    assert finding.after is None


# --------------------------------------------------------------------------
# DQ-10 — currency and separator formatting
# --------------------------------------------------------------------------
def test_dq10_names_the_characters_it_removed_and_the_number_underneath():
    frame = frame_from_columns({"Building Value": ["$2,100,000.00"]})
    finding = _only(rules.dq10_currency_formatting(frame), "DQ-10")
    assert finding.op == "strip_currency_to_float"
    assert finding.after == "2100000"
    assert finding.evidence["removed"] == "$,", "the reviewer is shown what went"


def test_dq10_treats_an_accounting_bracket_as_formatting_and_keeps_the_sign():
    """`(45,000)` is a negative written for a human. The brackets are notation, so
    DQ-10 owns them; the negative itself is DQ-04's question and is preserved
    rather than quietly made positive."""
    frame = frame_from_columns({"Building Value": ["(45,000)"]})
    finding = _only(rules.dq10_currency_formatting(frame), "DQ-10")
    assert finding.after == "-45000"
    assert "()" in finding.evidence["removed"]


def test_dq10_ignores_a_bare_number():
    frame = frame_from_columns({"Building Value": [1250000, 980000.5]})
    assert rules.dq10_currency_formatting(frame) == []


def test_dq10_stays_silent_on_a_cell_stripping_cannot_rescue():
    """Offering `strip_currency_to_float` on prose would propose an op that must
    fail. That cell is DQ-02's, and only DQ-02's."""
    frame = frame_from_columns({"Building Value": ["see attached schedule"]})
    assert rules.dq10_currency_formatting(frame) == []
    found, _ran = schema.validate(frame)
    assert _rows(found, "DQ-02") == [2]


# --------------------------------------------------------------------------
# DQ-11 — sprinkler codes (owned by schema.py, enum + the fix table)
# --------------------------------------------------------------------------
def test_dq11_leaves_the_four_permitted_codes_alone():
    frame = frame_from_columns(
        {"Fire Sprinklers (Y/N)": ["Y", "N", "Y13", "Y(13R)"]}
    )
    found, ran = schema.validate(frame)
    assert ran
    assert _rows(found, "DQ-11") == []


@pytest.mark.parametrize(
    "written,expected",
    [("Yes", "Y"), ("No", "N"), ("sprinklered", "Y"), ("NFPA 13", "Y13"),
     ("NFPA 13R", "Y(13R)"), ("0", "N"), ("1", "Y")],
)
def test_dq11_maps_an_unambiguous_spelling_or_coverage_figure(written, expected):
    frame = frame_from_columns({"Fire Sprinklers (Y/N)": [written]})
    found, _ran = schema.validate(frame)
    finding = _only(found, "DQ-11")
    assert finding.op == "map_values"
    assert finding.after == expected


@pytest.mark.parametrize("written", ["Partial", "partially sprinklered",
                                     "under review", "Varies", "Warehouse only"])
def test_dq11_never_auto_maps_an_ambiguous_term(written):
    """SRS 9.2 DQ-11: ambiguous terms always go to a human. The engine is capable
    of guessing here and must not: Y overstates the protection and N understates
    it, and both are rating inputs."""
    frame = frame_from_columns({"Fire Sprinklers (Y/N)": [written]})
    found, _ran = schema.validate(frame)
    finding = _only(found, "DQ-11")
    assert finding.op is None and finding.after is None
    assert "human" in finding.detail


@pytest.mark.parametrize("written", ["TBD", "None", "N/A", "unknown"])
def test_a_sprinkler_cell_naming_an_absence_is_dq01_and_not_dq11(written):
    """Measured behaviour, pinned here because it is a precedence rule and not an
    accident: `NULL_TOKENS` in `frame.py` claims these strings before DQ-11 ever
    sees them, so the cell is reported as a missing value rather than as an
    unrecognised sprinkler code.

    For "TBD" and "unknown" that is plainly right. For **"None" it is arguable and
    currently loses information**: "None" in a sprinkler column means no
    sprinklers, `SPRINKLER_SYNONYMS` maps it to "N", and that entry is therefore
    unreachable — a reviewer is asked to supply a missing value instead of being
    offered `map_values` -> N. Recorded rather than quietly changed, because the
    fix is a decision about which table wins and the brief defers that kind of
    tuning until the review loop is wired."""
    frame = frame_from_columns({"Fire Sprinklers (Y/N)": [written]})
    found, _ran = schema.validate(frame)
    assert _rows(found, "DQ-11") == []
    assert _rows(rules.dq01_missing_values(frame), "DQ-01") == [2]


@pytest.mark.parametrize("written", ["0.5", "50%", "0.07"])
def test_dq11_sends_partial_coverage_to_a_human_whatever_the_notation(written):
    """The real corpus writes sprinkler coverage as a fraction (`0.5`) or a
    percentage (`50%`). Both are partial coverage, and partial coverage has no
    code in the SRS 5.1 enum, so neither gets an op."""
    frame = frame_from_columns({"Fire Sprinklers (Y/N)": [written]})
    found, _ran = schema.validate(frame)
    finding = _only(found, "DQ-11")
    assert finding.op is None and finding.after is None


def test_dq11_reports_an_unrecognised_value_without_inventing_a_code():
    frame = frame_from_columns({"Fire Sprinklers (Y/N)": ["halon"]})
    found, _ran = schema.validate(frame)
    finding = _only(found, "DQ-11")
    assert finding.op is None
    assert "matches no known spelling" in finding.detail


# --------------------------------------------------------------------------
# DQ-02, DQ-03 — the cleaner's per-cell typing verdict
# --------------------------------------------------------------------------
def test_dq02_fires_on_a_value_field_holding_prose_but_not_on_formatting():
    frame = frame_from_columns(
        {"Building Value": ["see attached schedule", "$1,250,000", 400000.0]}
    )
    found, _ran = schema.validate(frame)
    finding = _only(found, "DQ-02")
    assert finding.row == 2
    assert finding.op is None, "no SRS 9.1 op turns prose into a number"


def test_dq03_separates_a_fractional_integer_from_an_unreadable_one():
    """Both are DQ-03, but only one of them has a `to_int` that would work — and
    SRS 9.1 requires `to_int` to *reject* non-whole input, so neither gets an op
    and the difference lives in the explanation the reviewer acts on."""
    frame = frame_from_columns({"Storeys": ["1.5", "two", 3.0, 4]})
    found, _ran = schema.validate(frame)
    assert _rows(found, "DQ-03") == [2, 3], "3.0 is a whole number written as a float"

    by_row = {f.row: f for f in found if f.rule == "DQ-03"}
    assert by_row[2].evidence["status"] == "fractional"
    assert "rounded" in by_row[2].detail
    assert by_row[3].evidence["status"] == "unparseable"
    assert all(f.op is None for f in by_row.values())


# --------------------------------------------------------------------------
# DQ-04, DQ-05, DQ-06 — the pandera range checks
# --------------------------------------------------------------------------
def test_dq04_flags_a_negative_amount_without_flipping_the_sign():
    frame = frame_from_columns({"Building Value": [-50000.0, 0.0, 120000.0]})
    found, ran = schema.validate(frame)
    assert ran
    finding = _only(found, "DQ-04")
    assert finding.row == 2, "zero is not negative"
    assert finding.op is None and finding.after is None
    assert "credit" in finding.detail


def test_dq05_flags_an_impossible_year_but_accepts_the_current_one():
    frame = frame_from_columns({"Year Built": [3015, 1650, 1700, 1985,
                                               datetime.now().year]})
    found, _ran = schema.validate(frame)
    assert _rows(found, "DQ-05") == [2, 3]
    assert all(f.op is None for f in found if f.rule == "DQ-05")
    assert "3015" in _only([f for f in found if f.row == 2], "DQ-05").detail


def test_dq06_flags_a_count_below_one_in_both_fields_it_owns():
    frame = frame_from_columns(
        {"Storeys": [0, 1, 12], "Number of Buildings": [1, 0, 3]}
    )
    found, _ran = schema.validate(frame)
    dq06 = [f for f in found if f.rule == "DQ-06"]
    assert sorted((f.row, f.field) for f in dq06) == [
        (2, "Storeys"), (3, "Number of Buildings")
    ]
    assert all(f.op is None for f in dq06), "the engine will not substitute 1"


def test_schema_validation_reports_that_it_ran_even_on_a_clean_frame():
    """A clean sheet and a skipped validator must not look identical, which is the
    same distinction `rules_not_applicable` exists for."""
    frame = frame_from_columns({"Building Value": [100.0], "Storeys": [2]})
    found, ran = schema.validate(frame)
    assert found == []
    assert ran is True


# --------------------------------------------------------------------------
# DQ-12 — placeholders
# --------------------------------------------------------------------------
def test_dq12_flags_year_zero_on_its_own_and_proposes_an_empty_cell():
    """The case the brief names explicitly: Sample 1's `Yr Blt = 0`."""
    frame = frame_from_columns({"Year Built": [1985, 0, 1990]})
    finding = _only(rules.dq12_placeholders(frame), "DQ-12")
    assert finding.row == 3
    assert finding.op == "set_null"
    assert finding.after is None, "C-02: set_null empties the cell, never fills it"
    assert "not a year" in finding.detail


def test_dq12_needs_repetition_before_calling_1900_a_placeholder():
    """One building really was built in 1900. Three on one schedule were not."""
    once = frame_from_columns({"Year Built": [1900, 1985, 1990]})
    assert rules.dq12_placeholders(once) == []

    repeated = frame_from_columns({"Year Built": [1900, 1900, 1900, 1985]})
    found = rules.dq12_placeholders(repeated)
    assert _rows(found, "DQ-12") == [2, 3, 4]
    assert found[0].evidence["occurrences"] == 3


def test_dq12_flags_zero_storeys_but_not_one():
    frame = frame_from_columns({"Storeys": [0, 1, 4]})
    finding = _only(rules.dq12_placeholders(frame), "DQ-12")
    assert finding.row == 2
    assert finding.op == "set_null"


def test_dq12_flags_a_value_of_one_but_not_a_real_small_value():
    frame = frame_from_columns({"Building Value": [1.0, 100.0, 2500000.0]})
    finding = _only(rules.dq12_placeholders(frame), "DQ-12")
    assert finding.row == 2
    assert finding.op == "set_null" and finding.after is None


# --------------------------------------------------------------------------
# DQ-13 — duplicate locations
# --------------------------------------------------------------------------
def test_dq13_matches_two_spellings_of_one_address():
    frame = frame_from_columns(
        {"Address": ["100 Main Street", "100 Main St.", "42 Oak Ave"],
         "Zip": ["02110", "02110", "02110"]}
    )
    found = rules.dq13_duplicate_locations(frame)
    assert _rows(found, "DQ-13") == [2, 3], "42 Oak Ave is on its own"
    assert all(f.op is None for f in found), "which duplicate to drop is a judgement"
    assert found[0].evidence["normalised"] == "100 main st"


def test_dq13_does_not_fold_the_house_number_away():
    """Case, punctuation and street-type words are folded; digits never are. "100
    Main St" and "1000 Main St" are different buildings."""
    frame = frame_from_columns(
        {"Address": ["100 Main St", "1000 Main St"], "Zip": ["02110", "02110"]}
    )
    assert rules.dq13_duplicate_locations(frame) == []


def test_dq13_separates_one_address_in_two_zips():
    frame = frame_from_columns(
        {"Address": ["100 Main St", "100 Main St"], "Zip": ["02110", "90210"]}
    )
    assert rules.dq13_duplicate_locations(frame) == []


def test_dq13_reports_distinct_location_references_as_the_campus_reading():
    """A school district or a shopping centre lists several buildings at one street
    address, each with its own location reference. The finding still stands — SRS
    9.2 defines the duplicate by address and zip — but it says which of the two
    situations the rows are in, and `recommend.py` lowers its confidence for it."""
    frame = frame_from_columns(
        {"Reference": ["LOC-001", "LOC-002"],
         "Address": ["1 Campus Dr", "1 Campus Drive"],
         "Zip": ["02110", "02110"]}
    )
    found = rules.dq13_duplicate_locations(frame)
    assert _rows(found, "DQ-13") == [2, 3]
    assert found[0].evidence["distinct_references"] is True
    assert "separate buildings" in found[0].detail

    recs = recommend.build_recommendations(found)
    assert len(recs) == 1
    assert recs[0].confidence <= 0.40
    assert recs[0].uncertainty, "SRS 5.4 requires one below 0.70"


def test_dq13_reads_a_shared_reference_as_a_possible_double_entry():
    frame = frame_from_columns(
        {"Reference": ["LOC-001", "LOC-001"],
         "Address": ["1 Campus Dr", "1 Campus Drive"],
         "Zip": ["02110", "02110"]}
    )
    found = rules.dq13_duplicate_locations(frame)
    assert found[0].evidence["distinct_references"] is False
    assert "double the insured value" in found[0].detail


# --------------------------------------------------------------------------
# DQ-14 — frame height
# --------------------------------------------------------------------------
def test_dq14_flags_a_wood_frame_tower():
    frame = frame_from_columns(
        {"Construction": ["Wood Frame", "Wood Frame"], "Storeys": [14, 3]}
    )
    finding = _only(rules.dq14_frame_height(frame), "DQ-14")
    assert finding.row == 2, "three storeys of frame is a normal building"
    assert finding.op is None, "the class or the count is wrong; the row does not say"


@pytest.mark.parametrize(
    "construction",
    ["Masonry Non-Combustible (ISO 4)", "Steel Frame", "Reinforced Concrete",
     "Fire Resistive", "Joisted Masonry", "Non Combustible"],
)
def test_dq14_tolerates_the_lookalikes_that_contain_its_own_keywords(construction):
    """This is the test the module exists for. "Masonry Non-Combustible (ISO 4)"
    appears 154 times in SOV_Q8B3 and contains the substring "combustible";
    "Steel Frame" contains "frame". A substring test would have reported a few
    hundred masonry and steel buildings as wood towers."""
    assert rules.is_frame_construction(construction) is False
    frame = frame_from_columns(
        {"Construction": [construction], "Storeys": [22]}
    )
    assert rules.dq14_frame_height(frame) == []


def test_every_frame_exclusion_actually_excludes():
    """The exclusion table is load-bearing, so it is checked as a table rather
    than only through the phrases that happen to appear in the corpus."""
    for term in FRAME_EXCLUSIONS:
        assert rules.is_frame_construction(f"{term} construction") is False


@pytest.mark.parametrize(
    "construction", ["Wood Frame", "frame", "Timber", "Joisted Wood", "ISO 1"]
)
def test_dq14_still_recognises_real_frame_descriptions(construction):
    assert rules.is_frame_construction(construction) is True


def test_dq14_says_nothing_when_the_storey_count_is_unreadable():
    frame = frame_from_columns({"Construction": ["Wood Frame"], "Storeys": ["many"]})
    assert rules.dq14_frame_height(frame) == []


# --------------------------------------------------------------------------
# DQ-17 — mixed date formats
# --------------------------------------------------------------------------
def test_dq17_flags_the_minority_notation_and_recovers_the_year_from_a_date():
    """Excel turning a bare year into a date is the common cause of a mixed
    column, and the year is recoverable from the value itself — so this is the one
    DQ-17 shape that carries `to_int` and a worked after-value."""
    frame = frame_from_columns({"Year Built": ["1985", "1990", date(1972, 1, 1)]})
    finding = _only(rules.dq17_mixed_dates(frame), "DQ-17")
    assert finding.row == 4
    assert finding.op == "to_int"
    assert finding.after == "1972"
    assert finding.evidence["dominant_shape"] == "bare year"


def test_dq17_proposes_nothing_for_a_decade():
    """"1920's" names no single year, so there is no value to convert it to
    without choosing one."""
    frame = frame_from_columns({"Year Built": ["1985", "1990", "1920's"]})
    finding = _only(rules.dq17_mixed_dates(frame), "DQ-17")
    assert finding.op is None and finding.after is None
    assert finding.evidence["shape"] == "decade"


def test_dq17_ignores_a_consistently_written_column():
    """A column written entirely as 01/01/1990 is consistent. It may still be the
    wrong type for the field, but that is a mapping and typing question, not a
    mixed-format one — and reporting it here would flag every row of a clean
    column."""
    frame = frame_from_columns(
        {"Year Built": ["01/01/1985", "06/01/1990", "12/31/1972"]}
    )
    assert rules.dq17_mixed_dates(frame) == []


def test_dq17_leaves_a_column_of_bare_years_alone():
    frame = frame_from_columns({"Year Built": ["1985", "1990", "1972"]})
    assert rules.dq17_mixed_dates(frame) == []


# --------------------------------------------------------------------------
# DQ-18 — component outliers
# --------------------------------------------------------------------------
def test_dq18_flags_contents_far_above_building_value():
    frame = frame_from_columns(
        {"Building Value": [1_000_000.0, 1_000_000.0],
         "Contents": [50_000_000.0, 5_000_000.0]}
    )
    finding = _only(rules.dq18_component_outliers(frame), "DQ-18")
    assert finding.row == 2, "5x is high but inside the 10x limit"
    assert finding.op is None
    assert finding.evidence["ratio"] == 50.0


def test_dq18_checks_bi_as_well_as_contents():
    frame = frame_from_columns(
        {"Building Value": [1_000_000.0], "BI": [40_000_000.0]}
    )
    finding = _only(rules.dq18_component_outliers(frame), "DQ-18")
    assert finding.field == "BI"


def test_dq18_skips_a_row_with_no_building_value_to_compare_against():
    """A ratio against nothing is not a ratio. Reporting these rows would turn
    every empty Building Value into a second, misleading finding — and SOV_K4T9's
    Building Value column is 0% populated."""
    frame = frame_from_columns(
        {"Building Value": [None, 0.0], "Contents": [50_000_000.0, 50_000_000.0]}
    )
    assert rules.dq18_component_outliers(frame) == []


# --------------------------------------------------------------------------
# DQ-15 — totals rows
# --------------------------------------------------------------------------
def test_dq15_finds_a_totals_row_by_arithmetic_alone():
    """No label at all: the row is identified because its figures equal the sum of
    every other row in two independent value columns. This is the SOV_B4ID row 28
    case, where `label` is None and three columns reconciled."""
    frame = frame_from_columns(
        {"Building Value": [1000.0, 2000.0, 3000.0, 6000.0],
         "Contents": [100.0, 200.0, 300.0, 600.0]}
    )
    finding = _only(structural.dq15_totals_rows(frame), "DQ-15")
    assert finding.row == 5
    assert finding.op == "exclude_row"
    assert finding.field is None, "DQ-15 is about the row, not a field"
    assert finding.evidence["reconciling_columns"] == ["Building Value", "Contents"]
    assert finding.evidence["label"] is None


def test_dq15_will_not_call_a_row_a_total_on_its_label_alone():
    """A label is corroboration, never grounds on its own. Searching the real
    corpus for "total", "grand" and "sum" matched "Rio Grande Blvd", "Grand
    Prairie" and "Samuel Grand" — every one a genuine location."""
    frame = frame_from_columns(
        {"Reference": ["Total", "A", "B"],
         "Building Value": [1000.0, 2000.0, 3000.0]}
    )
    assert structural.dq15_totals_rows(frame) == []


def test_dq15_ignores_the_place_names_that_look_like_totalling_words():
    frame = frame_from_columns(
        {"Address": ["100 Rio Grande Blvd", "Grand Prairie Mall", "Samuel Grand Park"],
         "Building Value": [1000.0, 2000.0, 3000.0]}
    )
    assert structural.dq15_totals_rows(frame) == []


def test_dq15_needs_two_reconciling_columns_without_a_label_and_one_with():
    """One column reconciling is a coincidence: on SOV_B4ID a genuine location row
    matched in its single populated value column. A corroborating label is what
    makes one column enough."""
    unlabelled = frame_from_columns(
        {"Building Value": [1000.0, 2000.0, 3000.0, 6000.0]}
    )
    assert structural.dq15_totals_rows(unlabelled) == []

    labelled = frame_from_columns(
        {"Reference": ["A", "B", "C", "Total"],
         "Building Value": [1000.0, 2000.0, 3000.0, 6000.0]}
    )
    finding = _only(structural.dq15_totals_rows(labelled), "DQ-15")
    assert finding.row == 5
    assert finding.evidence["label"] == "Total"


def test_dq15_is_declared_not_applicable_rather_than_clean_on_a_two_row_sheet():
    frame = frame_from_columns({"Building Value": [1000.0, 1000.0]})
    found, skipped = structural.run_structural_rules(frame)
    assert found == []
    assert "DQ-15" in skipped
    assert "at least three" in skipped["DQ-15"]


# --------------------------------------------------------------------------
# DQ-16 — TIV reconciliation
#
# CONSTRUCTED INPUT, and this is the one rule that has no alternative. In all four
# real sample workbooks the TIV column is an Excel `=SUM(...)` formula whose cached
# result is absent from the saved file, so every candidate column reads as empty,
# `find_tiv_column` correctly returns None, and the rule can never fire against
# real data. The frames below are built in memory — but through the same cleaner
# and typed-frame builder as production, via `frame_from_columns`.
# --------------------------------------------------------------------------
def test_dq16_reports_a_disagreement_and_fills_nothing():
    frame = frame_from_columns(
        {"Building Value": [1_000_000.0, 2_000_000.0],
         "Contents": [100_000.0, 200_000.0]},
        unmapped={"Total Insured Value": [1_100_000.0, 3_000_000.0]},
    )
    finding = _only(structural.dq16_tiv_reconciliation(frame), "DQ-16")
    assert finding.row == 3, "row 2's components match its stated total exactly"
    assert finding.op is None, "FR-DQ-06: mismatch flagged, no value filled"
    assert finding.after is None
    assert finding.evidence["stated"] == 3_000_000.0
    assert finding.evidence["computed"] == 2_200_000.0
    assert finding.evidence["difference"] == -800_000.0
    assert finding.source_column == "Total Insured Value"


def test_dq16_tolerates_a_rounding_level_disagreement():
    """SRS 9.2 sets the threshold at 1%. Floats that came from currency strings do
    not add up exactly, and a rule that flagged every cent would flag every file."""
    frame = frame_from_columns(
        {"Building Value": [1_000_000.0], "Contents": [5_000.0]},
        unmapped={"TIV": [1_000_000.0]},
    )
    assert structural.dq16_tiv_reconciliation(frame) == []


def test_dq16_skips_a_row_dq15_already_called_a_total():
    """A totals row's own TIV legitimately differs from its own components'
    arithmetic. Flagging it would report the consequence of a problem already
    reported."""
    frame = frame_from_columns(
        {"Building Value": [1_000_000.0, 2_000_000.0]},
        unmapped={"TIV": [5_000_000.0, 9_000_000.0]},
    )
    assert len(structural.dq16_tiv_reconciliation(frame)) == 2
    assert structural.dq16_tiv_reconciliation(frame, totals_rows={2, 3}) == []


def test_dq16_finds_no_tiv_column_when_the_formulas_carry_no_cached_result():
    """Exactly the real-corpus situation, reproduced: a correctly-named total
    column holding nothing. The rule must report that it could not run rather than
    that the file reconciled."""
    frame = frame_from_columns(
        {"Building Value": [1_000_000.0, 2_000_000.0, 3_000_000.0]},
        unmapped={"Total Insured Value": [None, None, None]},
    )
    assert structural.find_tiv_column(frame) is None
    _found, skipped = structural.run_structural_rules(frame)
    assert "DQ-16" in skipped
    assert "=SUM()" in skipped["DQ-16"]


def test_dq16_ignores_a_mapped_component_column():
    """Only unmapped headers are candidates. A column Agent 2 resolved to a target
    field is a component, and reconciling the components against one of themselves
    would always agree."""
    frame = frame_from_columns(
        {"Building Value": [1_000_000.0], "Other": [2_000_000.0]},
        source_column={"Building Value": "Building TIV", "Other": "Total Other"},
    )
    assert structural.find_tiv_column(frame) is None


# --------------------------------------------------------------------------
# reconcile — one cell, two true findings, one review decision
# --------------------------------------------------------------------------
def test_reconcile_lets_the_placeholder_finding_supersede_the_range_finding():
    """Year Built 0 is both a placeholder (DQ-12, with an op) and out of range
    (DQ-05, without one). Both are true; a reviewer does not need telling twice,
    and the one carrying a fix is the one to keep. The loser is recorded on the
    survivor rather than dropped silently."""
    frame = frame_from_columns({"Year Built": [0]})
    schema_found, _ran = schema.validate(frame)
    domain_found = rules.dq12_placeholders(frame)
    assert _rows(schema_found, "DQ-05") == [2]
    assert _rows(domain_found, "DQ-12") == [2]

    kept = rules.reconcile(domain_found + schema_found)
    assert [f.rule for f in kept] == ["DQ-12"]
    assert kept[0].evidence["also_flagged"] == ["DQ-05"]
    assert kept[0].op == "set_null"


def test_reconcile_supersedes_the_storeys_range_finding_too():
    frame = frame_from_columns({"Storeys": [0]})
    schema_found, _ran = schema.validate(frame)
    kept = rules.reconcile(rules.dq12_placeholders(frame) + schema_found)
    assert [f.rule for f in kept] == ["DQ-12"]
    assert kept[0].evidence["also_flagged"] == ["DQ-06"]


def test_reconcile_leaves_findings_on_different_cells_alone():
    frame = frame_from_columns({"Year Built": [0, 3015]})
    schema_found, _ran = schema.validate(frame)
    kept = rules.reconcile(rules.dq12_placeholders(frame) + schema_found)
    assert sorted((f.rule, f.row) for f in kept) == [("DQ-05", 3), ("DQ-12", 2)]


# --------------------------------------------------------------------------
# run_domain_rules — and saying which rules could not run
# --------------------------------------------------------------------------
def test_run_domain_rules_names_the_fields_a_skipped_rule_needed():
    """"No issues" and "the rule never ran" are different facts, and `QualityBlock`
    refuses to construct unless all eighteen rules are accounted for one way or the
    other."""
    frame = frame_from_columns({"Building Value": [100.0, 200.0]})
    _found, skipped = rules.run_domain_rules(frame)
    assert set(skipped) == {"DQ-07", "DQ-08", "DQ-09", "DQ-13", "DQ-14", "DQ-17"}
    assert skipped["DQ-07"] == "needs Zip, State, which Agent 2 did not map"
    assert skipped["DQ-08"] == "needs Zip, which Agent 2 did not map"


def test_rules_that_need_no_particular_field_always_run():
    frame = frame_from_columns({"County": ["Suffolk", None]})
    found, skipped = rules.run_domain_rules(frame)
    assert {"DQ-01", "DQ-10", "DQ-12"}.isdisjoint(skipped)
    assert _rows(found, "DQ-01") == [3]


def test_all_eighteen_srs_rules_are_owned_by_exactly_one_module():
    owned = (
        {rule for rule, _fn, _req in rules.DOMAIN_RULES}
        | set(schema.SCHEMA_RULES)
        | {"DQ-15", "DQ-16"}
    )
    assert owned == set(RULE_CATALOGUE)


def test_no_rule_ever_proposes_an_op_outside_the_srs_91_whitelist():
    """`Recommendation` would reject one anyway, but a rule that produced an
    unknown op would fail late and opaquely. This frame triggers nine rules at
    once and checks every op they propose."""
    frame = frame_from_columns(
        {"Reference": ["LOC-1", "LOC-2", "LOC-3", None],
         "Address": ["100 Main Street", "100 Main St.", "42 Oak Ave", "42 Oak Ave"],
         "State": ["Massachusetts", "MA", "MA", "MA"],
         "Zip": ["2110", "02110-1234", "02110", "02110"],
         "Building Value": ["$2,100,000.00", -5000.0, 1.0, None],
         "Storeys": [0, 14, "1.5", 2],
         "Construction": ["Wood Frame", "Wood Frame", "Masonry Non-Combustible", "x"],
         "Year Built": [0, "1920's", 1985, 1990]}
    )
    domain, _skipped = rules.run_domain_rules(frame)
    schema_found, _ran = schema.validate(frame)
    structural_found, _s = structural.run_structural_rules(frame)
    found = rules.reconcile(domain + schema_found + structural_found)

    assert len(found) > 12, "the frame is meant to trigger most of the catalogue"
    for finding in found:
        assert finding.rule in RULE_CATALOGUE
        assert finding.op is None or finding.op in OP_WHITELIST
        if finding.op == "set_null":
            assert finding.after is None, "C-02 holds at the finding, not just the rec"


# --------------------------------------------------------------------------
# Completeness and the intake quality score
# --------------------------------------------------------------------------
def test_completeness_counts_reported_absences_as_absent():
    frame = frame_from_columns(
        {"Building Value": [100.0, None, "N/A", 400.0], "State": ["MA"] * 4}
    )
    per_field = {c.field: c for c in completeness.field_completeness(frame)}
    assert len(per_field) == 17, "FR-DQ-01 reports all seventeen SRS 5.1 fields"
    assert per_field["Building Value"].non_null == 2
    assert per_field["Building Value"].completeness == 0.5
    assert per_field["State"].completeness == 1.0


def test_an_unmapped_field_is_zero_percent_rather_than_absent():
    """A file missing Year Built entirely is in worse shape than one that merely
    has it empty on three rows, and a score that ignored unmapped fields would
    rank the two the same way."""
    frame = frame_from_columns({"Building Value": [100.0]})
    per_field = {c.field: c for c in completeness.field_completeness(frame)}
    assert per_field["Year Built"].mapped is False
    assert per_field["Year Built"].completeness == 0.0
    assert per_field["Year Built"].source_column is None


def test_the_score_is_reported_with_the_three_terms_that_produced_it():
    frame = frame_from_columns(
        {"Building Value": [100.0, None], "State": ["MA", "MA"]}
    )
    per_field = completeness.field_completeness(frame)
    score, components = completeness.intake_quality_score(per_field, [], frame.n_rows)
    assert set(components) == {"completeness", "validity", "coverage"}
    assert 0.0 <= score <= 1.0
    assert components["validity"] == 1.0, "no findings were passed in"
    assert components["coverage"] < 0.3, "two of seventeen fields are mapped"
    expected = sum(components[k] * w for k, w in completeness.SCORE_WEIGHTS.items())
    assert score == pytest.approx(round(expected, 4))


def test_missing_values_are_not_counted_against_the_score_twice():
    """DQ-01 is excluded from the validity term because missing values are already
    the completeness term. Counting them in both would make an incomplete file look
    doubly bad for one reason."""
    frame = frame_from_columns({"Building Value": [100.0, None]})
    per_field = completeness.field_completeness(frame)
    dq01 = rules.dq01_missing_values(frame)
    assert dq01, "the frame has a missing value to be found"

    clean = completeness.intake_quality_score(per_field, [], 2)
    with_dq01 = completeness.intake_quality_score(per_field, dq01, 2)
    assert with_dq01 == clean

    negative = schema.validate(
        frame_from_columns({"Building Value": [100.0, -5.0]})
    )[0]
    with_dq04 = completeness.intake_quality_score(per_field, negative, 2)
    assert with_dq04[0] < clean[0], "a DQ-04 finding does cost the validity term"


def test_value_field_gaps_names_an_empty_but_mapped_value_column():
    """SOV_K4T9's Building Value column is mapped and 0% populated, which is the
    single most consequential thing about that file and is easy to miss in a list
    of seventeen percentages."""
    frame = frame_from_columns(
        {"Building Value": [None, None], "Contents": [100.0, 200.0]}
    )
    per_field = completeness.field_completeness(frame)
    assert completeness.value_field_gaps(per_field) == ["Building Value"]


# --------------------------------------------------------------------------
# Recommendations (FR-DQ-08, FR-DQ-09, SRS 5.4)
# --------------------------------------------------------------------------
def test_one_recommendation_per_rule_field_and_op_not_per_cell():
    """FR-DQ-08's grouping rule, and the reason 623 sprinkler findings on one real
    file became three reviewable decisions instead of 623 identical questions."""
    frame = frame_from_columns(
        {"Zip": ["2110", "2111", "02110"], "State": ["MA", "MA", "MA"]}
    )
    found = rules.dq08_zip_format(frame)
    assert len(found) == 2

    recs = recommend.build_recommendations(found)
    assert len(recs) == 1
    assert recs[0].rows == [2, 3]
    assert recs[0].op == "zip5"
    assert [(e.row, e.before, e.after) for e in recs[0].before_after] == [
        (2, "2110", "02110"), (3, "2111", "02111")
    ]


def test_an_actionable_and_a_flag_group_split_even_within_one_rule():
    """Same rule, same field, different op — and that is the distinction a reviewer
    needs, because one of the two groups is a button and the other is a question."""
    frame = frame_from_columns(
        {"Zip": ["2110", "not a zip"], "State": ["MA", "MA"]}
    )
    recs = recommend.build_recommendations(rules.dq08_zip_format(frame))
    by_action = {r.action_type: r for r in recs}
    assert set(by_action) == {"data_correction", "flag_for_review"}
    assert by_action["data_correction"].op == "zip5"
    assert by_action["flag_for_review"].op is None
    assert by_action["flag_for_review"].before_after == [], (
        "SRS 5.4: a flag proposes no change, so it carries no before/after"
    )


def test_the_review_queue_is_ordered_by_severity_and_ids_follow_the_order():
    """SRS 9.2: severity "sets the order of the review queue; it never triggers an
    automatic change". Ids are assigned after sorting, so R-001 is the first thing
    a reviewer should look at."""
    frame = frame_from_columns(
        {"Building Value": [None, 100.0, 200.0],
         "Zip": ["2110", "02110", "02110"],
         "State": ["MA", "MA", "MA"]}
    )
    domain, _skipped = rules.run_domain_rules(frame)
    recs = recommend.build_recommendations(domain)
    assert [(r.id, r.rule, r.severity) for r in recs] == [
        ("R-001", "DQ-01", "High"), ("R-002", "DQ-08", "Low")
    ]
    assert recs[0].action_type == "flag_for_review"
    assert recs[1].op == "zip5"


def test_a_set_null_recommendation_cannot_carry_a_replacement_value():
    """`Recommendation` enforces this rather than trusting the caller, so no future
    rule can smuggle a value in through the one op whose output C-02 fixes."""
    frame = frame_from_columns({"Year Built": [1985, 0, 1990]})
    rec = recommend.build_recommendations(rules.dq12_placeholders(frame))[0]
    assert rec.op == "set_null"
    assert rec.action_type == "data_correction"
    assert [e.after for e in rec.before_after] == [None]
    assert rec.before_after[0].before == "0", "FR-DQ-09: the before is the real cell"


def test_every_recommendation_below_the_threshold_states_its_uncertainty():
    frame = frame_from_columns(
        {"Address": ["100 Main St", "100 Main St"],
         "Zip": ["02110", "02110"],
         "Building Value": [None, -5000.0],
         "Contents": ["see attached", 100.0]}
    )
    domain, _skipped = rules.run_domain_rules(frame)
    schema_found, _ran = schema.validate(frame)
    recs = recommend.build_recommendations(
        rules.reconcile(domain + schema_found)
    )
    assert recs, "the frame is meant to produce recommendations"
    for rec in recs:
        assert rec.rows, "SRS 5.4 requires the affected rows"
        assert rec.rationale.startswith(rec.rule)
        if rec.confidence < 0.70:
            assert (rec.uncertainty or "").strip(), f"{rec.id} states no uncertainty"
        if rec.op is None:
            assert rec.action_type == "flag_for_review"
            assert rec.before_after == []
        else:
            assert rec.op in OP_WHITELIST
            assert rec.before_after, "FR-DQ-09 requires a worked example"
        assert rec.status == "pending", "C-01: nothing is applied without approval"
        assert rec.attempts == 0
        assert rec.llm_reasoned is False, "no reasoner was consulted here"


def test_the_rationale_quotes_the_rule_the_rows_and_the_engines_own_detail():
    frame = frame_from_columns(
        {"Zip": ["2110", "2111", "2112"], "State": ["MA", "MA", "MA"]}
    )
    rec = recommend.build_recommendations(rules.dq08_zip_format(frame))[0]
    assert "DQ-08" in rec.rationale
    assert "3 rows" in rec.rationale
    assert "leading zero" in rec.rationale
    assert "nothing is changed until this is approved" in rec.rationale


def test_a_flag_rationale_says_no_whitelisted_op_can_fix_it():
    frame = frame_from_columns({"State": ["Atlantis"]})
    rec = recommend.build_recommendations(rules.dq09_state_format(frame))[0]
    assert "raised for review rather than corrected" in rec.rationale
    assert rec.op is None
