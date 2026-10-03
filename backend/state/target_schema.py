"""The frozen 17-field target schema from SRS 5.1, plus the vocabulary used to
recognise it in client files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TargetField:
    name: str
    dtype: Literal["string", "integer", "float"]


TARGET_FIELDS: tuple[TargetField, ...] = (
    TargetField("Reference", "string"),
    TargetField("Address", "string"),
    TargetField("City", "string"),
    TargetField("State", "string"),
    TargetField("Zip", "integer"),
    TargetField("County", "string"),
    TargetField("Country", "string"),
    TargetField("Building Value", "float"),
    TargetField("Contents", "float"),
    TargetField("BI", "float"),
    TargetField("Occupancy", "string"),
    TargetField("Construction", "string"),
    TargetField("Storeys", "integer"),
    TargetField("Number of Buildings", "integer"),
    TargetField("Year Built", "integer"),
    TargetField("Fire Sprinklers (Y/N)", "string"),
    TargetField("Other", "float"),
)

TARGET_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in TARGET_FIELDS)

OUTPUT_FILE_NAME = "Cleaned_SOV.xlsx"
OUTPUT_SHEET_NAME = "Cleaned_SOV"
AUDIT_FILE_NAME = "Audit_Log.xlsx"

# Phrases that indicate a target field in a source header. Agent 1 only needs
# these for the target-vocabulary-overlap factor (FR-SHT-01); Agent 2 builds the
# full insurance glossary on top of the same table.
TARGET_VOCABULARY: dict[str, tuple[str, ...]] = {
    "Reference": (
        "reference", "ref", "loc", "loc no", "loc number", "location", "location id",
        "location no", "location number", "location ref", "site", "site id",
        "premises", "premises no", "building id", "account", "acct", "policy loc",
    ),
    "Address": (
        "address", "street address", "street", "addr", "address 1", "address line 1",
        "location address", "situs address", "property address", "mailing address",
    ),
    "City": ("city", "town", "municipality", "city name"),
    "State": ("state", "st", "province", "state code", "state name", "state abbrev"),
    "Zip": ("zip", "zipcode", "zip code", "postal code", "postcode", "zip5", "post code"),
    "County": ("county", "parish", "district", "county name"),
    "Country": ("country", "nation", "country code", "ctry"),
    "Building Value": (
        "building value", "bldg value", "building", "bldg", "building tiv",
        "building replacement cost", "bldg repl cost", "repl cost", "replacement cost",
        "real property", "structure value", "building amount", "building limit",
    ),
    "Contents": (
        "contents", "contents value", "cont", "personal property",
        "business personal property", "bpp", "stock", "contents tiv", "contents limit",
    ),
    "BI": (
        "bi", "business income", "business interruption", "bi ee", "bi/ee", "bi and ee",
        "extra expense", "time element", "loss of income", "bi value", "bi limit",
    ),
    "Occupancy": (
        "occupancy", "occ", "occupancy type", "occupancy class", "use", "usage",
        "operations", "class of occupancy",
    ),
    "Construction": (
        "construction", "const", "constr", "construction type", "constr type",
        "iso class", "iso construction", "building construction", "wall type",
    ),
    "Storeys": (
        "storeys", "stories", "story", "storys", "floors", "num floors", "no of floors",
        "number of storeys", "number of stories", "stories above grade", "height stories",
    ),
    "Number of Buildings": (
        "number of buildings", "no of buildings", "num buildings", "bldgs", "buildings",
        "bldg count", "building count", "no of bldgs", "qty buildings",
    ),
    "Year Built": (
        "year built", "yr blt", "yr built", "yob", "built", "year of construction",
        "construction year", "year constructed", "orig year built",
    ),
    "Fire Sprinklers (Y/N)": (
        "fire sprinklers", "fire sprinkler", "sprinklers", "sprinkler", "sprk",
        "sprinklered", "auto sprinkler", "sprinkler y n", "prot sprinkler", "fire prot",
    ),
    "Other": (
        "other", "other value", "misc", "miscellaneous", "other property", "other tiv",
        "eq other", "other limit",
    ),
}
