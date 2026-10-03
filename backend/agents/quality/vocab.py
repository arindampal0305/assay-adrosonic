"""Reference data the SRS 9.2 rules check against.

Separated from the rules themselves because this is a table, not a judgement: a
reviewer who doubts a zip-vs-state flag should be able to read the allocation
that produced it without reading any logic. Every entry here is public postal or
insurance reference data.

The tables are deliberately conservative. A wrong entry does not merely miss a
finding, it manufactures a false one on a real client file, so where a boundary
is genuinely fuzzy the table errs toward silence — `STATE_ZIP_PREFIXES` includes
the odd outliers (Texarkana, Fishers Island, the Austin PO boxes) precisely
because omitting them would flag correct addresses.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# States (DQ-07, DQ-09)
# --------------------------------------------------------------------------
# name -> USPS code, for `state_to_abbrev`. Includes DC and the territories a
# commercial SOV can legitimately contain.
FULL_STATE_NAMES: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "district of columbia": "DC", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI",
    "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "puerto rico": "PR", "virgin islands": "VI", "guam": "GU",
    "american samoa": "AS", "northern mariana islands": "MP",
    # Written forms that appear in real files. "v.i." is here because SOV_B4ID
    # spells the US Virgin Islands that way on all thirteen of its location rows,
    # and a dotted abbreviation is exactly the "non-standard code" DQ-09 is for.
    "washington dc": "DC", "washington d.c.": "DC", "d.c.": "DC", "dc": "DC",
    "v.i.": "VI", "u.s.v.i.": "VI", "usvi": "VI", "virgin islands (us)": "VI",
    "p.r.": "PR", "puerto rico (us)": "PR",
    "n.y.": "NY", "calif": "CA", "calif.": "CA", "mass": "MA", "mass.": "MA",
    "penn": "PA", "penna": "PA", "penn.": "PA", "fla": "FL", "fla.": "FL",
    "tex": "TX", "tex.": "TX", "ariz": "AZ", "ariz.": "AZ", "wash": "WA",
}

# The codes SRS 5.1 accepts as already-normalised. Derived from the values above
# so a state can never be valid as an abbreviation but unknown as a name.
STATE_ABBREVS: frozenset[str] = frozenset(FULL_STATE_NAMES.values())

# 3-digit ZIP prefix ranges, inclusive, per USPS allocation. A 5-digit zip whose
# prefix falls outside its state's ranges is a zip-vs-state mismatch (DQ-07).
STATE_ZIP_PREFIXES: dict[str, tuple[tuple[int, int], ...]] = {
    "AL": ((350, 369),),
    "AK": ((995, 999),),
    "AZ": ((850, 853), (855, 857), (859, 860), (863, 865)),
    "AR": ((716, 729), (755, 755)),          # 755 = Texarkana, split with TX
    "CA": ((900, 908), (910, 928), (930, 961)),
    "CO": ((800, 816),),
    "CT": ((60, 69),),
    "DC": ((200, 205), (569, 569)),
    "DE": ((197, 199),),
    "FL": ((320, 339), (341, 342), (344, 344), (346, 347), (349, 349)),
    "GA": ((300, 319), (398, 399)),
    "HI": ((967, 968),),
    "ID": ((832, 838),),
    "IL": ((600, 629),),
    "IN": ((460, 479),),
    "IA": ((500, 528),),
    "KS": ((660, 679),),
    "KY": ((400, 427),),
    "LA": ((700, 701), (703, 708), (710, 714)),
    "ME": ((39, 49),),
    "MD": ((206, 219),),
    "MA": ((10, 27), (55, 55)),
    "MI": ((480, 499),),
    "MN": ((550, 567),),
    "MS": ((386, 397),),
    "MO": ((630, 658),),
    "MT": ((590, 599),),
    "NE": ((680, 693),),
    "NV": ((889, 898),),
    "NH": ((30, 38),),
    "NJ": ((70, 89),),
    "NM": ((870, 884),),
    "NY": ((4, 5), (63, 63), (100, 149)),    # 063 = Fishers Island
    "NC": ((269, 289),),
    "ND": ((580, 588),),
    "OH": ((430, 459),),
    "OK": ((730, 731), (734, 749)),
    "OR": ((970, 979),),
    "PA": ((150, 196),),
    "RI": ((28, 29),),
    "SC": ((290, 299),),
    "SD": ((570, 577),),
    "TN": ((370, 385),),
    "TX": ((733, 733), (750, 799), (885, 885)),   # 733 = Austin PO boxes
    "UT": ((840, 847),),
    "VT": ((50, 59),),
    "VA": ((201, 201), (220, 246)),
    "WA": ((980, 994),),
    "WV": ((247, 268),),
    "WI": ((530, 549),),
    "WY": ((820, 831),),
    "PR": ((6, 9),),
    "VI": ((8, 8),),
    "GU": ((969, 969),),
    "AS": ((967, 967),),
    "MP": ((969, 969),),
}

# States with zips that genuinely begin with 0, so a 4-digit zip there is a lost
# leading zero rather than a contradiction. This is the exact set DQ-07 excludes.
LEADING_ZERO_STATES: frozenset[str] = frozenset(
    code for code, ranges in STATE_ZIP_PREFIXES.items()
    if any(lo < 100 for lo, _ in ranges)
)

# --------------------------------------------------------------------------
# Fire sprinklers (DQ-11)
# --------------------------------------------------------------------------
# Normalised source text -> canonical SRS 5.1 code. Only entries whose meaning is
# beyond doubt belong here: anything in this table can be applied by `map_values`
# once a reviewer approves, so an arguable synonym would become a silent
# mis-statement of a protection class.
SPRINKLER_SYNONYMS: dict[str, str] = {
    "y": "Y", "yes": "Y", "true": "Y", "t": "Y", "sprinklered": "Y",
    "fully sprinklered": "Y", "full": "Y", "wet": "Y", "dry": "Y",
    "n": "N", "no": "N", "false": "N", "f": "N",
    "none": "N", "not sprinklered": "N", "unsprinklered": "N", "non-sprinklered": "N",
    "y13": "Y13", "y 13": "Y13", "y-13": "Y13", "nfpa 13": "Y13", "nfpa13": "Y13",
    "y(13r)": "Y(13R)", "y13r": "Y(13R)", "y 13r": "Y(13R)", "y-13r": "Y(13R)",
    "nfpa 13r": "Y(13R)", "nfpa13r": "Y(13R)",
}
# Bare numbers are deliberately absent above and handled as coverage percentages
# instead. Every sprinkler column in the real corpus is a percentage column, so
# `1` is full coverage rather than a code — and `13` is the case that proves the
# point: in a "%Sprink" column it is 13% coverage, in a "Sprinkler Type" column it
# is NFPA 13, and the two readings disagree about whether the building is
# protected. A lookup table would have to pick one silently; a human should pick.

# SRS 9.2 DQ-11: 'ambiguous terms ("Partial") always go to a human'. These are
# never auto-mapped, however confident a model claims to be, because the
# difference between a partly sprinklered building and a sprinklered one is a
# rating difference and the source does not say which is meant.
SPRINKLER_AMBIGUOUS: frozenset[str] = frozenset(
    {
        "partial", "partially", "partially sprinklered", "part", "partial coverage",
        "some", "some areas", "p", "partial/none", "y/n", "mixed", "varies",
        "see schedule", "per schedule", "tbd", "under review", "in progress",
        "y - partial", "partial - y", "limited", "warehouse only", "office only",
    }
)

# --------------------------------------------------------------------------
# Construction (DQ-14)
# --------------------------------------------------------------------------
# Phrases that denote combustible wood-frame construction. DQ-14 pairs these with
# a storey count above 6, which is implausible for frame and usually means the
# construction class or the storey count is wrong.
FRAME_CONSTRUCTION_TERMS: tuple[str, ...] = (
    "frame", "wood", "woodframe", "wood frame", "timber", "joisted wood",
    "iso 1", "iso class 1", "construction class 1", "frame/wood",
)
# Checked *before* the terms above, and a match here settles the question.
#
# "combustible" is deliberately absent from the term list and "non-combustible"
# present here: the real corpus contains "Masonry Non-Combustible (ISO 4)" 154
# times, and a substring test for "combustible" calls every one of those rows
# wood frame. The exclusion list is the only thing standing between DQ-14 and a
# few hundred confident false positives.
FRAME_EXCLUSIONS: tuple[str, ...] = (
    "non-combustible", "noncombustible", "non combustible",
    "steel frame", "metal frame", "concrete frame", "steel framed",
    "reinforced concrete", "fire resistive", "masonry",
)
MAX_PLAUSIBLE_FRAME_STOREYS = 6

# --------------------------------------------------------------------------
# Placeholders (DQ-12)
# --------------------------------------------------------------------------
# SRS 9.2 DQ-12: "Year Built 0 or repeated 1900, Storeys 0, value of 1".
PLACEHOLDER_YEARS: frozenset[int] = frozenset({0, 1900, 1800, 1000, 1111, 9999})
# A year is only a placeholder by repetition if it recurs at least this often and
# is one of the suspicious round years above. One building really was built in
# 1900; forty of them on one schedule were not.
PLACEHOLDER_REPEAT_MIN = 3
PLACEHOLDER_VALUES: frozenset[float] = frozenset({1.0})
PLACEHOLDER_STOREYS: frozenset[int] = frozenset({0})

# --------------------------------------------------------------------------
# Outliers and reconciliation (DQ-16, DQ-18)
# --------------------------------------------------------------------------
# SRS 9.2 DQ-16: "Components disagree with a source TIV column by more than 1%".
TIV_TOLERANCE = 0.01
# SRS 9.2 DQ-18: "Contents or BI above 10x Building Value".
COMPONENT_RATIO_LIMIT = 10.0
# Headers that name a total-insured-value column Agent 2 left unmapped.
TIV_HEADER_HINTS: tuple[str, ...] = (
    "tiv", "total insured value", "total value", "total insurable value",
    "grand total", "total", "sum insured", "total limit", "combined value",
)

__all__ = [
    "COMPONENT_RATIO_LIMIT",
    "FRAME_CONSTRUCTION_TERMS",
    "FRAME_EXCLUSIONS",
    "FULL_STATE_NAMES",
    "LEADING_ZERO_STATES",
    "MAX_PLAUSIBLE_FRAME_STOREYS",
    "PLACEHOLDER_REPEAT_MIN",
    "PLACEHOLDER_STOREYS",
    "PLACEHOLDER_VALUES",
    "PLACEHOLDER_YEARS",
    "SPRINKLER_AMBIGUOUS",
    "SPRINKLER_SYNONYMS",
    "STATE_ABBREVS",
    "STATE_ZIP_PREFIXES",
    "TIV_HEADER_HINTS",
    "TIV_TOLERANCE",
]
