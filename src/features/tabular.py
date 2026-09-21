"""Structured feature engineering and preprocessing.

Design decisions worth stating, because they are the ones an interviewer would
push on:

Ordinal bands are encoded as ordinals, not one-hot.
    Several source columns are banded versions of an underlying continuous
    quantity -- vehicle price, vehicle age, policyholder age, number of
    vehicles. One-hot encoding them throws away the ordering and inflates the
    feature count for no benefit. They are mapped to integers in their natural
    order instead, and the orderings are declared explicitly below rather than
    inferred from whatever order the values happen to appear in.

Undeclared sentinels become explicit missingness.
    ``driver_age`` is zero for a block of rows that all carry the youngest
    policyholder band. That is a coding artefact, not an age. It is converted
    to NaN and accompanied by an indicator column, so a model can use the fact
    that the value was missing without being told the driver is a newborn.

Everything is fitted inside a pipeline.
    Scalers, imputers and encoders are fitted on the training split only, as
    steps of an sklearn pipeline. Fitting a scaler on the full frame before
    splitting is one of the most common quiet leaks in tabular ML, and putting
    the fit inside the pipeline makes it structurally impossible here.

One engineered feature comes out of the data quality analysis.
    ``policy_type`` encodes a vehicle class in its own string, and in roughly a
    third of rows that class disagrees with ``vehicle_category``. The
    disagreement is real and systematic, so its presence is exposed as a
    feature rather than silently ignored or silently repaired.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.structured import SENTINEL_AGE
from src.utils.logging import get_logger

logger = get_logger(__name__)


#: Explicit orderings for banded columns. Declared rather than sorted, because
#: lexicographic order would put "more than 69000" before "less than 20000".
ORDINAL_ORDERS: dict[str, list[str]] = {
    "vehicle_price_band": [
        "less than 20000",
        "20000 to 29000",
        "30000 to 39000",
        "40000 to 59000",
        "60000 to 69000",
        "more than 69000",
    ],
    "vehicle_age_band": [
        "new",
        "2 years",
        "3 years",
        "4 years",
        "5 years",
        "6 years",
        "7 years",
        "more than 7",
    ],
    "policyholder_age_band": [
        "16 to 17",
        "18 to 20",
        "21 to 25",
        "26 to 30",
        "31 to 35",
        "36 to 40",
        "41 to 50",
        "51 to 65",
        "over 65",
    ],
    "number_of_cars": [
        "1 vehicle",
        "2 vehicles",
        "3 to 4",
        "5 to 8",
        "more than 8",
    ],
    # --- narrative-channel bands, used only by the narrative-as-columns
    # --- diagnostic model described in src/models/nlp
    "days_policy_accident": ["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"],
    "days_policy_claim": ["none", "8 to 15", "15 to 30", "more than 30"],
    "past_number_of_claims": ["none", "1", "2 to 4", "more than 4"],
    "number_of_supplements": ["none", "1 to 2", "3 to 5", "more than 5"],
    # Ordered by how recently the address changed: a more recent change ranks
    # higher, since recency is the property an underwriter would attend to.
    "address_change_claim": [
        "no change",
        "4 to 8 years",
        "2 to 3 years",
        "1 year",
        "under 6 months",
    ],
}

TABULAR_NUMERIC = ["driver_age", "deductible", "driver_rating"]
TABULAR_ORDINAL = [
    "vehicle_price_band",
    "vehicle_age_band",
    "policyholder_age_band",
    "number_of_cars",
]
TABULAR_NOMINAL = [
    "make",
    "vehicle_category",
    "driver_sex",
    "marital_status",
    "policy_type",
    "base_policy",
    "agent_type",
]
TABULAR_ENGINEERED = ["driver_age_missing", "policy_type_vehicle_mismatch"]

NARRATIVE_NUMERIC = ["report_delay_weeks", "incident_week_of_month", "claim_week_of_month"]
NARRATIVE_ORDINAL = [
    "days_policy_accident",
    "days_policy_claim",
    "past_number_of_claims",
    "number_of_supplements",
    "address_change_claim",
]
NARRATIVE_NOMINAL = [
    "accident_area",
    "fault",
    "police_report_filed",
    "witness_present",
    "incident_month",
    "incident_day_of_week",
    "claim_month",
    "claim_day_of_week",
]


@dataclass
class FeatureSpec:
    """Which columns a preprocessor consumes, by kind."""

    numeric: list[str] = field(default_factory=list)
    ordinal: list[str] = field(default_factory=list)
    nominal: list[str] = field(default_factory=list)

    @property
    def columns(self) -> list[str]:
        return [*self.numeric, *self.ordinal, *self.nominal]


def tabular_spec(*, include_year: bool = False, include_sex: bool = True) -> FeatureSpec:
    """Feature set for the structured baseline.

    ``include_year`` defaults to False: under the temporal split the calendar
    year is constant within the test period, so including it gives the model a
    feature whose test-time value it has never seen in training.

    ``include_sex`` exists so the fairness analysis can compare a model that
    uses the attribute against one that does not. Dropping it does not by
    itself make a model fair -- correlated features remain -- which is part of
    what that comparison is meant to show.
    """
    nominal = [c for c in TABULAR_NOMINAL if include_sex or c != "driver_sex"]
    numeric = [*TABULAR_NUMERIC, *TABULAR_ENGINEERED]
    if include_year:
        numeric = [*numeric, "policy_year"]
    return FeatureSpec(numeric=numeric, ordinal=list(TABULAR_ORDINAL), nominal=nominal)


def narrative_spec() -> FeatureSpec:
    """Feature set for the narrative-fields-as-columns diagnostic.

    This is not one of the seven ablation cells. It measures the ceiling the
    text model is working towards: how well the narrative channel predicts
    fraud when handed over as clean columns rather than as prose. The gap
    between this and the transformer is the cost of expressing those facts in
    natural language and then having to read them back out.
    """
    return FeatureSpec(
        numeric=list(NARRATIVE_NUMERIC),
        ordinal=list(NARRATIVE_ORDINAL),
        nominal=list(NARRATIVE_NOMINAL),
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns. Pure: returns a copy, mutates nothing."""
    out = df.copy()

    # The zero-coded age sentinel: record that it was missing, then blank it so
    # the imputer handles it as missing rather than as a real value of zero.
    if "driver_age" in out.columns:
        is_sentinel = out["driver_age"] == SENTINEL_AGE
        out["driver_age_missing"] = is_sentinel.astype("int64")
        out.loc[is_sentinel, "driver_age"] = np.nan

    # policy_type carries a vehicle class in its own text. Where that class
    # disagrees with vehicle_category, the record is internally inconsistent.
    if {"policy_type", "vehicle_category"} <= set(out.columns):
        declared = out["policy_type"].astype(str).str.split(" - ").str[0]
        out["policy_type_vehicle_mismatch"] = (
            declared != out["vehicle_category"].astype(str)
        ).astype("int64")

    return out


def _ordinal_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Map banded strings to their declared ordinal positions.

    Unrecognised values become NaN and are imputed downstream, rather than
    being silently assigned position zero.
    """
    data = {}
    for col in columns:
        order = ORDINAL_ORDERS[col]
        lookup = {value: index for index, value in enumerate(order)}
        mapped = df[col].astype(str).map(lookup)
        n_unknown = int(mapped.isna().sum() - df[col].isna().sum())
        if n_unknown > 0:
            logger.warning(
                "%d values of %s are outside the declared ordinal scale", n_unknown, col
            )
        data[col] = mapped
    return pd.DataFrame(data, index=df.index)


class OrdinalBandEncoder:
    """Stateless transformer mapping declared bands to integers.

    Implemented as a small class rather than a ``FunctionTransformer`` so that
    the column list travels with it and the feature names it produces are
    available for SHAP.
    """

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803 - sklearn API
        del X, y
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803 - sklearn API
        return _ordinal_frame(X, self.columns).to_numpy(dtype="float64")

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:  # noqa: N803
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        del input_features
        return np.asarray(self.columns, dtype=object)


def build_preprocessor(
    spec: FeatureSpec,
    *,
    scale_numeric: bool = True,
    sparse_output: bool = False,
) -> ColumnTransformer:
    """Assemble the preprocessing ColumnTransformer for a feature spec.

    ``scale_numeric`` is on for linear models and neural fusion heads, which
    need comparable scales, and can be turned off for tree ensembles, which do
    not care and are easier to interpret on the raw scale.
    """
    numeric_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median", add_indicator=False)),
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    ordinal_steps: list[tuple[str, object]] = [
        ("encode", OrdinalBandEncoder(spec.ordinal)),
        ("impute", SimpleImputer(strategy="most_frequent")),
    ]
    if scale_numeric:
        ordinal_steps.append(("scale", StandardScaler()))

    transformers = []
    if spec.numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), spec.numeric))
    if spec.ordinal:
        transformers.append(("ordinal", Pipeline(ordinal_steps), spec.ordinal))
    if spec.nominal:
        transformers.append(
            (
                "nominal",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="constant", fill_value="__missing__"),
                        ),
                        (
                            "encode",
                            # Unknown categories at inference time are ignored
                            # rather than raising: a live claim may legitimately
                            # carry a vehicle make absent from the training data.
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=sparse_output,
                                min_frequency=10,
                            ),
                        ),
                    ]
                ),
                spec.nominal,
            )
        )

    return ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Readable feature names for a fitted preprocessor, for SHAP and reports."""
    return [str(n) for n in preprocessor.get_feature_names_out()]
