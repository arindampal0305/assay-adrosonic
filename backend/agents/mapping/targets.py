"""The target catalogue Agent 2 maps onto (FR-MAP-01).

`backend/state/target_schema.py` owns the frozen 17 fields and the shallow
vocabulary Agent 1 needs. This module is the mapping-specific enrichment layer on
top of it: for each target, an insurance glossary of source-header spellings, a
natural-language description for the semantic channel, and the value *shape* the
fingerprint channel expects to see in a column that really holds that field.

Nothing here is derived from a model. It is a hand-authored broker glossary, which
is why a mapping decision can cite it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from backend.state.target_schema import (
    TARGET_FIELD_NAMES,
    TARGET_FIELDS,
    TARGET_VOCABULARY,
)

# The value shapes the fingerprint channel can recognise. Deliberately coarse:
# a fingerprint cannot tell Building Value from Contents (both are currency), and
# pretending otherwise would be false precision. Its job is to *rule out*
# nonsense, e.g. that a column of four-digit years is an Address.
#
# Two known indistinguishable groups, both handled by listing the shapes as
# mutually acceptable rather than by trying to tell them apart:
#
#   currency              Building Value / Contents / BI / Other
#   place_name ~ category City / County / Country / Occupancy / Construction
#
# The second was found by running real data: `Masonry, Frame, Masonry` under the
# header `Const Type` fingerprinted as `place_name`, and because `Construction`
# accepted only `category` the channel scored the correct target 0.0 and the
# Hungarian solver handed the column to `Country`. Cardinality heuristics cannot
# fix this reliably — `Houston, Chicago` and `Masonry, Frame` are the same shape
# to any value-level test. So both shapes are accepted by both groups, and lexical
# and semantic do the separating.
ValueShape = Literal[
    "currency",
    "zip",
    "year",
    "small_int",
    "us_state",
    "boolean",
    "street_address",
    "place_name",
    "identifier",
    "category",
]


@dataclass(frozen=True)
class TargetSpec:
    name: str
    dtype: Literal["string", "integer", "float"]
    # Expanded for the semantic channel. Written as a sentence a broker would
    # recognise, because cosine similarity against "BI" alone is meaningless.
    description: str
    # Accepted value shapes, best first. The fingerprint channel scores a column
    # by how much of its observed shape distribution lands in this set.
    shapes: tuple[ValueShape, ...]
    # Additional header spellings beyond TARGET_VOCABULARY. Kept separate so the
    # Agent 1 vocabulary stays small and fast while Agent 2 gets the long tail.
    extra_synonyms: tuple[str, ...] = ()


_SPECS: tuple[TargetSpec, ...] = (
    TargetSpec(
        "Reference",
        "string",
        "A short identifier the broker uses for this insured location — a location "
        "number, site code, premises number or row label that distinguishes one "
        "scheduled property from another. Not a street address.",
        # `zip` is accepted because a column of `1001, 1002, 1003` is
        # indistinguishable by value from a postal code that lost its leading
        # zero. The header decides which it is; the values cannot.
        ("identifier", "small_int", "zip"),
        ("loc #", "location #", "no", "num", "item", "item no", "seq", "sequence",
         "line", "line no", "row", "key", "unit", "branch", "branch no", "store",
         "store no", "plant", "plant no", "facility id", "sov id"),
    ),
    TargetSpec(
        "Address",
        "string",
        "The street address of the insured property — house or building number plus "
        "street name, as it would be geocoded. Excludes city, state and postal code "
        "when those are separate columns.",
        ("street_address", "place_name"),
        ("address1", "addr1", "street no and name", "physical address",
         "risk address", "insured location", "location desc", "site address",
         "premises address", "building address", "loc address", "street 1"),
    ),
    TargetSpec(
        "City",
        "string",
        "The city, town or municipality the insured property sits in.",
        ("place_name", "category"),
        ("city town", "city or town", "locality", "suburb", "place"),
    ),
    TargetSpec(
        "State",
        "string",
        "The US state or Canadian province the property is in, usually a two-letter "
        "code such as TX, CA or NY, occasionally spelled out in full.",
        ("us_state", "place_name"),
        ("st.", "state prov", "state/prov", "province state", "region", "st cd"),
    ),
    TargetSpec(
        "Zip",
        "integer",
        "The postal code of the property — a five-digit US ZIP, sometimes a nine-digit "
        "ZIP+4 or a Canadian alphanumeric postal code. Leading zeros are significant.",
        ("zip", "identifier"),
        ("zip 4", "zip+4", "zip code 5", "postal", "pc", "post"),
    ),
    TargetSpec(
        "County",
        "string",
        "The county, parish or borough containing the property. Used for catastrophe "
        "zone and tax lookups.",
        ("place_name", "category"),
        ("county parish", "borough", "cnty", "county/parish"),
    ),
    TargetSpec(
        "Country",
        "string",
        "The country the property is in, as a name or ISO code such as US, USA or CA.",
        ("place_name", "identifier", "category"),
        ("country name", "iso country", "cntry", "domicile"),
    ),
    TargetSpec(
        "Building Value",
        "float",
        "The insured value of the building structure itself — replacement cost or "
        "total insurable value of the real property, excluding contents and business "
        "income. A currency amount, typically the largest value on the row.",
        ("currency",),
        ("bldg", "building val", "bldg val", "building rc", "bldg rc", "rc value",
         "building replacement value", "real property value", "structure",
         "building tiv value", "bldg limit", "building sum insured", "prop damage bldg"),
    ),
    TargetSpec(
        "Contents",
        "float",
        "The insured value of contents at the location — business personal property, "
        "stock, machinery, equipment and furnishings. A currency amount, usually "
        "smaller than the building value.",
        ("currency",),
        ("contents val", "content", "bpp value", "personal prop", "pers property",
         "machinery and equipment", "m&e", "equipment", "stock and contents",
         "contents sum insured", "contents rc"),
    ),
    TargetSpec(
        "BI",
        "float",
        "Business income or business interruption value — lost earnings and extra "
        "expense if the location cannot operate after a loss. Often abbreviated BI, "
        "BI/EE, or shown as a time-element value. A currency amount.",
        ("currency",),
        ("bi value", "bi/ee value", "business income value", "business interruption value",
         "time element value", "ee", "extra exp", "rental income", "loss of rents",
         "gross earnings", "bi limit value", "profits"),
    ),
    TargetSpec(
        "Occupancy",
        "string",
        "What the building is used for — office, warehouse, retail, apartment, "
        "manufacturing, school. A descriptive category, not a construction material.",
        ("category", "place_name"),
        ("occupancy desc", "occupancy description", "building use", "property use",
         "risk class", "business type", "industry", "sic", "naics", "occ code",
         "occ class", "tenancy"),
    ),
    TargetSpec(
        "Construction",
        "string",
        "What the building is physically made of — frame, masonry, joisted masonry, "
        "non-combustible, fire resistive, reinforced concrete, steel. An ISO "
        "construction class, not what the building is used for.",
        ("category", "place_name"),
        ("construction desc", "construction class", "const class", "constr class",
         "iso const", "building type", "frame type", "material", "wall construction",
         "roof and wall", "cons", "construction code"),
    ),
    TargetSpec(
        "Storeys",
        "integer",
        "How many floors or storeys the building has above grade. A small whole "
        "number, almost always between 1 and 100.",
        ("small_int",),
        ("no stories", "num stories", "# stories", "# floors", "storys above grade",
         "floor count", "levels", "no of levels", "height", "stories/floors"),
    ),
    TargetSpec(
        "Number of Buildings",
        "integer",
        "How many separate structures are covered at this one location. A small whole "
        "number, very often exactly 1.",
        ("small_int",),
        ("# bldgs", "# buildings", "no bldgs", "num bldgs", "count of buildings",
         "structures", "no of structures", "bldg qty", "units", "no of units"),
    ),
    TargetSpec(
        "Year Built",
        "integer",
        "The calendar year the building was originally constructed — a four-digit year "
        "such as 1978 or 2015. Values like 1900 or 0 are often placeholders for unknown.",
        ("year",),
        ("yr", "year", "yr of construction", "date built", "year of build",
         "original construction year", "construction date", "built year", "vintage",
         "effective year built", "eyb"),
    ),
    TargetSpec(
        "Fire Sprinklers (Y/N)",
        "string",
        "Whether the building has an automatic fire sprinkler system. A yes/no, Y/N "
        "or true/false flag, occasionally a percentage of area sprinklered.",
        ("boolean", "category"),
        # 'sprink' is a truncation seen on a real SOV as the header '%Sprink'. It
        # needs its own entry because token_set_ratio scores it only 0.80 against
        # 'sprinkler' — enough to rank the target first, not enough to clear the
        # admissibility floor, so the column was left unmapped while the target
        # went unclaimed.
        ("sprinkler y/n", "sprinklered y n", "sprinkler system", "auto sprinklers",
         "fire suppression", "suppression", "sprinkler indicator", "spr", "spkr",
         "sprink", "sprinkler coverage", "fire protection"),
    ),
    TargetSpec(
        "Other",
        "float",
        "Any further insured value at the location that is not building, contents or "
        "business income — outdoor property, yard improvements, fences, signs, "
        "miscellaneous or equipment-breakdown values. A currency amount.",
        ("currency",),
        ("other val", "misc value", "other prop value", "yard improvements",
         "outdoor property", "fences and signs", "improvements", "betterments",
         "tenant improvements", "eq value", "equipment breakdown", "all other"),
    ),
)

TARGETS: dict[str, TargetSpec] = {spec.name: spec for spec in _SPECS}

# Fail loudly at import if this catalogue and the frozen SRS 5.1 schema ever drift
# apart, rather than silently mapping onto a field that no longer exists.
_missing = set(TARGET_FIELD_NAMES) - set(TARGETS)
_extra = set(TARGETS) - set(TARGET_FIELD_NAMES)
if _missing or _extra:
    raise RuntimeError(
        "mapping target catalogue is out of sync with target_schema.TARGET_FIELDS: "
        f"missing={sorted(_missing)} unexpected={sorted(_extra)}"
    )
for _spec, _field in zip(_SPECS, TARGET_FIELDS):
    if _spec.dtype != _field.dtype:
        raise RuntimeError(
            f"dtype drift for '{_spec.name}': catalogue says {_spec.dtype}, "
            f"target_schema says {_field.dtype}"
        )


def synonyms_for(target: str) -> tuple[str, ...]:
    """Every header spelling that evidences `target`, deduplicated, longest first.

    Longest first matters: a lexical match on 'business interruption' is far more
    informative than one on 'bi', so the citation should quote the specific phrase.
    """
    spec = TARGETS[target]
    seen: dict[str, None] = {}
    for phrase in (target.lower(), *TARGET_VOCABULARY.get(target, ()), *spec.extra_synonyms):
        seen.setdefault(phrase.lower().strip(), None)
    return tuple(sorted(seen, key=lambda p: (-len(p), p)))


GLOSSARY: dict[str, tuple[str, ...]] = {name: synonyms_for(name) for name in TARGETS}

GLOSSARY_SIZE = sum(len(v) for v in GLOSSARY.values())


def description_for(target: str) -> str:
    """The text the semantic channel embeds. Name plus description, because the
    bare name carries real signal for 'City' and none at all for 'BI'."""
    spec = TARGETS[target]
    return f"{spec.name}: {spec.description}"
