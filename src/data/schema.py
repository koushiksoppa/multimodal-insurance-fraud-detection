"""The unified claim-level schema.

One row per claim, regardless of which modalities that claim happens to carry.
The schema is derived from what the source data can actually support: the real
structured dataset has no claim amount, no free-text narrative and no damage
photograph, so those fields either do not exist (claim amount, on the real-label
track) or are explicitly marked as constructed.

Three ideas drive the layout.

Modality channels
    The real structured columns are partitioned into two *disjoint* channels: a
    narrative channel, whose fields are rendered into claim text and then
    dropped from the tabular feature set, and a tabular channel, which stays
    numeric/categorical. The partition rule is "what kind of information is
    this?" -- circumstantial detail a person would describe in a report goes to
    the narrative, policy and vehicle attributes stay tabular. The rule was
    fixed before any per-feature predictive power was measured, so the split
    is not tuned to flatter either modality. Without a disjoint partition, the
    text modality would be a lossy re-encoding of the tabular one and any
    "multimodal gain" would be an artefact.

Explicit provenance per field
    Each field records whether it is real, derived from real data, or
    synthetic, so that no reader has to guess which parts of a claim record
    came from a genuine source.

Tracks
    ``real`` uses the genuine fraud labels. ``synthetic`` is a controlled
    benchmark with a known multimodal generating process. Both share this
    schema so that the same models, splits and evaluation code run unchanged
    on either.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Provenance(str, Enum):
    """Where a field's values come from."""

    REAL = "real"
    """Taken verbatim from a real third-party dataset."""

    DERIVED = "derived"
    """Computed deterministically from real fields."""

    CONSTRUCTED = "constructed"
    """Produced by this project's documented, seeded construction code."""


class Channel(str, Enum):
    """Which modality carries the field into the models."""

    IDENTIFIER = "identifier"
    TABULAR = "tabular"
    NARRATIVE = "narrative"
    IMAGE = "image"
    TARGET = "target"
    BOOKKEEPING = "bookkeeping"


@dataclass(frozen=True)
class Field:
    name: str
    dtype: str
    channel: Channel
    provenance: Provenance
    description: str
    source_column: str | None = None
    nullable: bool = False


# --------------------------------------------------------------------------
# identifiers
# --------------------------------------------------------------------------

IDENTIFIERS = [
    Field(
        "claim_id",
        "string",
        Channel.IDENTIFIER,
        Provenance.DERIVED,
        "Stable primary key, formatted CLM-XXXXXXX from the source row order.",
    ),
    Field(
        "policy_number",
        "int64",
        Channel.IDENTIFIER,
        Provenance.REAL,
        "Policy identifier from the source table.",
        source_column="PolicyNumber",
    ),
    Field(
        "rep_number",
        "int64",
        Channel.IDENTIFIER,
        Provenance.REAL,
        "Handling representative. Used as a grouping candidate when checking "
        "for leakage, not as a model feature by default.",
        source_column="RepNumber",
    ),
]


# --------------------------------------------------------------------------
# tabular channel
#
# Policy, vehicle and party attributes: the structured facts an insurer holds
# about a claim independently of how the incident was described.
# --------------------------------------------------------------------------

TABULAR_FIELDS = [
    Field("make", "category", Channel.TABULAR, Provenance.REAL,
          "Vehicle manufacturer.", source_column="Make"),
    Field("vehicle_category", "category", Channel.TABULAR, Provenance.REAL,
          "Sedan, Sport or Utility.", source_column="VehicleCategory"),
    Field("vehicle_price_band", "category", Channel.TABULAR, Provenance.REAL,
          "Banded vehicle value, for example '20000 to 29000'.",
          source_column="VehiclePrice"),
    Field("vehicle_age_band", "category", Channel.TABULAR, Provenance.REAL,
          "Banded vehicle age, for example '5 years'.", source_column="AgeOfVehicle"),
    Field("driver_age", "int64", Channel.TABULAR, Provenance.REAL,
          "Driver age in years. Contains a zero-coded missing value in the "
          "source data; see the data quality report.", source_column="Age"),
    Field("driver_sex", "category", Channel.TABULAR, Provenance.REAL,
          "Driver sex. Retained for the subgroup fairness analysis; whether it "
          "is used as a model feature is an explicit experiment setting.",
          source_column="Sex"),
    Field("marital_status", "category", Channel.TABULAR, Provenance.REAL,
          "Marital status.", source_column="MaritalStatus"),
    Field("policyholder_age_band", "category", Channel.TABULAR, Provenance.REAL,
          "Banded policyholder age.", source_column="AgeOfPolicyHolder"),
    Field("policy_type", "category", Channel.TABULAR, Provenance.REAL,
          "Combined vehicle category and base policy, for example "
          "'Sport - Liability'. Redundant with vehicle_category and "
          "base_policy; the redundancy is quantified in the data quality report.",
          source_column="PolicyType"),
    Field("base_policy", "category", Channel.TABULAR, Provenance.REAL,
          "Liability, Collision or All Perils.", source_column="BasePolicy"),
    Field("deductible", "int64", Channel.TABULAR, Provenance.REAL,
          "Policy deductible.", source_column="Deductible"),
    Field("driver_rating", "int64", Channel.TABULAR, Provenance.REAL,
          "Internal driver rating, 1-4.", source_column="DriverRating"),
    Field("number_of_cars", "category", Channel.TABULAR, Provenance.REAL,
          "Banded count of vehicles on the policy.", source_column="NumberOfCars"),
    Field("agent_type", "category", Channel.TABULAR, Provenance.REAL,
          "Internal or External agent.", source_column="AgentType"),
    Field("policy_year", "int64", Channel.TABULAR, Provenance.REAL,
          "Calendar year of the claim. Drives the temporal split.",
          source_column="Year"),
]


# --------------------------------------------------------------------------
# narrative channel
#
# Circumstantial detail about the incident and the reporting process. These
# fields are rendered into claim_text and are *excluded* from the tabular
# feature matrix, so that the text modality carries information the tabular
# modality genuinely does not have.
# --------------------------------------------------------------------------

NARRATIVE_FIELDS = [
    Field("accident_area", "category", Channel.NARRATIVE, Provenance.REAL,
          "Urban or Rural.", source_column="AccidentArea"),
    Field("fault", "category", Channel.NARRATIVE, Provenance.REAL,
          "Whether the policy holder or a third party was at fault.",
          source_column="Fault"),
    Field("police_report_filed", "category", Channel.NARRATIVE, Provenance.REAL,
          "Whether a police report was filed.", source_column="PoliceReportFiled"),
    Field("witness_present", "category", Channel.NARRATIVE, Provenance.REAL,
          "Whether a witness was present.", source_column="WitnessPresent"),
    Field("days_policy_accident", "category", Channel.NARRATIVE, Provenance.REAL,
          "Banded days between policy inception and the accident.",
          source_column="Days_Policy_Accident"),
    Field("days_policy_claim", "category", Channel.NARRATIVE, Provenance.REAL,
          "Banded days between policy inception and the claim.",
          source_column="Days_Policy_Claim"),
    Field("past_number_of_claims", "category", Channel.NARRATIVE, Provenance.REAL,
          "Banded count of the claimant's previous claims.",
          source_column="PastNumberOfClaims"),
    Field("address_change_claim", "category", Channel.NARRATIVE, Provenance.REAL,
          "How recently the policyholder changed address.",
          source_column="AddressChange_Claim"),
    Field("number_of_supplements", "category", Channel.NARRATIVE, Provenance.REAL,
          "Banded count of supplementary documents filed with the claim.",
          source_column="NumberOfSuppliments"),
    Field("incident_month", "category", Channel.NARRATIVE, Provenance.REAL,
          "Month of the incident.", source_column="Month"),
    Field("incident_week_of_month", "int64", Channel.NARRATIVE, Provenance.REAL,
          "Week within the incident month.", source_column="WeekOfMonth"),
    Field("incident_day_of_week", "category", Channel.NARRATIVE, Provenance.REAL,
          "Day of week of the incident.", source_column="DayOfWeek"),
    Field("claim_month", "category", Channel.NARRATIVE, Provenance.REAL,
          "Month the claim was filed.", source_column="MonthClaimed"),
    Field("claim_week_of_month", "int64", Channel.NARRATIVE, Provenance.REAL,
          "Week within the claim month.", source_column="WeekOfMonthClaimed"),
    Field("claim_day_of_week", "category", Channel.NARRATIVE, Provenance.REAL,
          "Day of week the claim was filed.", source_column="DayOfWeekClaimed"),
    Field("report_delay_weeks", "float64", Channel.NARRATIVE, Provenance.DERIVED,
          "Approximate weeks between incident and claim, computed from the "
          "month and week-of-month fields. Negative or implausible values are "
          "flagged rather than silently clipped.", nullable=True),
]


# --------------------------------------------------------------------------
# text and image channels
# --------------------------------------------------------------------------

TEXT_FIELDS = [
    Field("claim_text", "string", Channel.NARRATIVE, Provenance.CONSTRUCTED,
          "First-notice-of-loss narrative. Generated by a seeded, templated "
          "renderer from the narrative-channel fields above. The underlying "
          "facts are real; their expression as prose is constructed, because "
          "no public corpus of fraud-labelled claim narratives exists."),
]

IMAGE_FIELDS = [
    Field("has_image", "bool", Channel.IMAGE, Provenance.CONSTRUCTED,
          "Whether a damage photograph is attached to this claim. Most claims "
          "have none: the image corpus is far smaller than the claim table, "
          "and each photograph is attached to at most one claim so that no "
          "image can appear on both sides of a split."),
    Field("image_id", "string", Channel.IMAGE, Provenance.CONSTRUCTED,
          "Identifier of the attached photograph.", nullable=True),
    Field("image_path", "string", Channel.IMAGE, Provenance.CONSTRUCTED,
          "Path to the photograph, relative to the image root.", nullable=True),
    Field("damage_location", "category", Channel.IMAGE, Provenance.REAL,
          "Front or rear. A real human-assigned label from the image dataset.",
          nullable=True),
    Field("damage_condition", "category", Channel.IMAGE, Provenance.REAL,
          "Normal, breakage or crushed. A real human-assigned label.",
          nullable=True),
    Field("damage_severity", "int64", Channel.IMAGE, Provenance.DERIVED,
          "Ordinal encoding of damage_condition: normal 0, breakage 1, "
          "crushed 2.", nullable=True),
]


# --------------------------------------------------------------------------
# targets and bookkeeping
# --------------------------------------------------------------------------

TARGET_FIELDS = [
    Field("fraud_label", "int64", Channel.TARGET, Provenance.REAL,
          "1 if the claim was recorded as fraudulent. Real on the 'real' "
          "track; regenerated by the documented process on the 'synthetic' "
          "track, where its provenance is CONSTRUCTED."),
]

BOOKKEEPING_FIELDS = [
    Field("split", "category", Channel.BOOKKEEPING, Provenance.CONSTRUCTED,
          "train, val or test."),
    Field("track", "category", Channel.BOOKKEEPING, Provenance.CONSTRUCTED,
          "real or synthetic."),
    Field("dataset_version", "string", Channel.BOOKKEEPING, Provenance.CONSTRUCTED,
          "Version string of the build that produced this row."),
]


ALL_FIELDS: list[Field] = (
    IDENTIFIERS
    + TABULAR_FIELDS
    + NARRATIVE_FIELDS
    + TEXT_FIELDS
    + IMAGE_FIELDS
    + TARGET_FIELDS
    + BOOKKEEPING_FIELDS
)

FIELDS_BY_NAME: dict[str, Field] = {f.name: f for f in ALL_FIELDS}

#: Columns fed to the tabular models. Deliberately excludes every
#: narrative-channel column, so the tabular and text modalities do not overlap.
TABULAR_FEATURES: list[str] = [f.name for f in TABULAR_FIELDS]

#: Columns rendered into claim_text and withheld from the tabular models.
NARRATIVE_FEATURES: list[str] = [f.name for f in NARRATIVE_FIELDS]

TARGET = "fraud_label"
DAMAGE_TARGET = "damage_condition"
GROUP_KEY = "claim_id"

#: Attributes available for the subgroup fairness analysis. These are real
#: attributes of the source data, not inferred or imputed.
FAIRNESS_ATTRIBUTES: list[str] = ["driver_sex", "marital_status", "policyholder_age_band"]

#: Source-column renames, built from the schema so the mapping cannot drift.
SOURCE_RENAMES: dict[str, str] = {
    f.source_column: f.name for f in ALL_FIELDS if f.source_column is not None
}


def assert_disjoint_channels() -> None:
    """Fail loudly if the tabular and narrative channels ever overlap.

    Called by the build pipeline and by the test suite. An overlap would make
    every multimodal result in this project suspect, so it is checked rather
    than assumed.
    """
    overlap = set(TABULAR_FEATURES) & set(NARRATIVE_FEATURES)
    if overlap:
        raise AssertionError(
            f"tabular and narrative channels overlap on {sorted(overlap)}; "
            "the text modality would be a re-encoding of the tabular one"
        )

    tabular_sources = {
        f.source_column for f in TABULAR_FIELDS if f.source_column is not None
    }
    narrative_sources = {
        f.source_column for f in NARRATIVE_FIELDS if f.source_column is not None
    }
    source_overlap = tabular_sources & narrative_sources
    if source_overlap:
        raise AssertionError(
            f"channels share source columns {sorted(source_overlap)}"
        )
