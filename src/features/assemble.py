"""Assemble every modality's features once, for all experiments to share.

Encoding text and images is the expensive part of this project -- roughly
fifteen minutes of CPU for the full corpus -- and it produces the same output
every time for the same inputs. Doing it inside each ablation cell would mean
paying it seven times over, per seed. So it happens once here, the results are
cached on disk, and every downstream experiment loads them.

The tabular block is a special case: its preprocessor must be *fitted on the
training split only*, so it is fitted here against the training rows and then
applied to all three splits. Fitting it on the full frame would be a textbook
leak, and doing it in one place makes that easy to verify.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.sources import image_root_path
from src.features.tabular import FeatureSpec, build_preprocessor, engineer_features, tabular_spec
from src.models.nlp.encoder import TextEncoder, TextEncoderConfig
from src.models.vision.encoder import VisionEncoder, VisionEncoderConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AssembledFeatures:
    """Feature matrices for the whole dataset, aligned row-for-row with ``claims``."""

    claims: pd.DataFrame
    tabular: np.ndarray
    tabular_names: list[str]
    text: np.ndarray
    image: np.ndarray
    reports: dict = field(default_factory=dict)

    @property
    def dims(self) -> dict[str, int]:
        return {
            "tabular": int(self.tabular.shape[1]),
            "text": int(self.text.shape[1]),
            "image": int(self.image.shape[1]),
        }

    def split_index(self, split: str) -> np.ndarray:
        return np.flatnonzero((self.claims["split"] == split).to_numpy())


def assemble(
    claims: pd.DataFrame,
    *,
    spec: FeatureSpec | None = None,
    text_config: TextEncoderConfig | None = None,
    vision_config: VisionEncoderConfig | None = None,
    image_root: Path | None = None,
    refresh: bool = False,
    include_text: bool = True,
    include_image: bool = True,
) -> AssembledFeatures:
    """Build the tabular, text and image blocks for every claim."""
    spec = spec or tabular_spec()
    reports: dict = {}

    # --- tabular ----------------------------------------------------------
    engineered = engineer_features(claims)
    train_mask = (claims["split"] == "train").to_numpy()
    if not train_mask.any():
        raise ValueError("no training rows; the preprocessor has nothing to fit on")

    preprocessor = build_preprocessor(spec, scale_numeric=True)
    # Fitted on train only. Applied to everything.
    preprocessor.fit(engineered.loc[train_mask, spec.columns])
    tabular = preprocessor.transform(engineered[spec.columns])
    tabular_names = [str(n) for n in preprocessor.get_feature_names_out()]
    tabular = np.asarray(tabular, dtype="float32")
    reports["tabular"] = {
        "n_features": len(tabular_names),
        "fitted_on": "train",
        "n_train_rows": int(train_mask.sum()),
    }
    logger.info("tabular block: %d features fitted on %d training rows",
                len(tabular_names), int(train_mask.sum()))

    # --- text -------------------------------------------------------------
    if include_text:
        encoder = TextEncoder(text_config)
        texts = claims["claim_text"].fillna("").tolist()
        text, text_report = encoder.encode_cached(texts, refresh=refresh)
        reports["text"] = text_report.as_dict()
    else:
        text = np.zeros((len(claims), 0), dtype="float32")

    # --- image ------------------------------------------------------------
    if include_image:
        image, image_report = _encode_images(
            claims, vision_config, image_root=image_root, refresh=refresh
        )
        reports["image"] = image_report
    else:
        image = np.zeros((len(claims), 0), dtype="float32")

    return AssembledFeatures(
        claims=claims,
        tabular=tabular,
        tabular_names=tabular_names,
        text=text,
        image=image,
        reports=reports,
    )


def _encode_images(
    claims: pd.DataFrame,
    config: VisionEncoderConfig | None,
    *,
    image_root: Path | None,
    refresh: bool,
) -> tuple[np.ndarray, dict]:
    """Encode the attached photographs and scatter them back onto all claims.

    Only about fifteen percent of claims carry a photograph, so only those are
    passed through the backbone; the rest keep a zero row, and the presence
    mask tells the fusion model which is which. Encoding 15,420 zero images to
    keep the array rectangular would waste most of the compute budget.
    """
    encoder = VisionEncoder(config)
    root = image_root or image_root_path()

    has_image = claims["has_image"].to_numpy().astype(bool)
    positions = np.flatnonzero(has_image)
    relative = claims.loc[has_image, "image_path"].tolist()
    paths = [root / rel for rel in relative]

    if not paths:
        return np.zeros((len(claims), encoder.embedding_dim), "float32"), {
            "n_encoded": 0, "note": "no images attached"
        }

    embeddings, report = encoder.encode_cached(paths, refresh=refresh)

    full = np.zeros((len(claims), embeddings.shape[1]), dtype="float32")
    full[positions] = embeddings

    payload = report.as_dict()
    payload["n_encoded"] = len(paths)
    payload["coverage"] = float(has_image.mean())
    return full, payload
