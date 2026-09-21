"""Vision encoder with on-disk embedding cache.

Backbone choice. ResNet-18, ImageNet-pretrained, used as a frozen feature
extractor. The reasoning is dataset size and compute, in that order. There are
2,300 images; fine-tuning a ViT or a ConvNeXt on that many photographs would
overfit long before it learned anything a smaller network could not, and on a
four-core CPU with no GPU a single ViT-B epoch would take longer than this
entire project's training budget. ResNet-18 extracts the whole corpus in a few
minutes and its 512-dimensional penultimate layer is a well-understood generic
visual representation. EfficientNet-B0 is available through the same interface
for comparison.

Why transfer learning at all. 2,300 images across six classes is roughly 380
per class. A convolutional network trained from scratch on that would learn
edge and texture detectors that ImageNet already provides, using data that
would be better spent on the damage-specific decision. The early layers of an
ImageNet model are generic -- edges, corners, textures -- and only the later
layers are class-specific, which is exactly the situation transfer learning
exists for.

Frozen versus fine-tuned. The default is frozen: extract once, cache, and train
light heads on top. That is what makes running seven ablation cells across
several seeds feasible here. Selective fine-tuning of the final residual block
is implemented in :mod:`src.models.vision.damage` for the damage task, where the
labels are real and the task is genuinely visual, and the two are compared.

Preprocessing follows the ImageNet convention the weights were trained under --
resize the short side to 256, centre crop to 224, normalise with the ImageNet
channel statistics. Deviating from it would hand the network inputs that do not
match the distribution its filters were learned on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.utils.config import get_paths, resolve_device
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: ImageNet channel statistics. The pretrained weights were learned on inputs
#: normalised this way; using different values would shift the input
#: distribution away from what the filters expect.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BACKBONES = {
    "resnet18": {"dim": 512, "weights": "ResNet18_Weights"},
    "resnet34": {"dim": 512, "weights": "ResNet34_Weights"},
    "efficientnet_b0": {"dim": 1280, "weights": "EfficientNet_B0_Weights"},
}


@dataclass(frozen=True)
class VisionEncoderConfig:
    backbone: str = "resnet18"
    image_size: int = 224
    resize_to: int = 256
    batch_size: int = 32
    normalize: bool = True
    device: str = "auto"

    def cache_key(self, image_paths: list[str]) -> str:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "backbone": self.backbone,
                    "image_size": self.image_size,
                    "resize_to": self.resize_to,
                    "normalize": self.normalize,
                },
                sort_keys=True,
            ).encode("utf-8")
        )
        for path in image_paths:
            digest.update(path.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()[:24]


@dataclass
class VisionEncodingReport:
    n_images: int
    embedding_dim: int
    backbone: str
    n_failed: int
    failed_examples: list[str]
    seconds: float
    from_cache: bool

    def as_dict(self) -> dict:
        return asdict(self)


def build_transform(config: VisionEncoderConfig, *, train: bool = False):
    """Preprocessing pipeline.

    Augmentation is applied only when ``train`` is set, and only the operations
    that preserve what the label means:

    * **Horizontal flip** is safe. The classes distinguish front from rear
      damage, and mirroring a car left-to-right changes neither.
    * **Small rotation and translation** are safe. They model the photographer
      standing slightly differently, which is real variation in claim photos.
    * **Mild colour jitter** is safe and models lighting and camera differences.
      It is kept mild deliberately: aggressive saturation or hue shifts can
      mask or manufacture the paint damage and rust that partly define the
      classes.

    Deliberately excluded:

    * **Vertical flip** -- an upside-down car is not a claim photograph, and the
      network would spend capacity on an orientation it will never see.
    * **Aggressive random cropping** -- the damage is often confined to one
      region, so a hard crop can remove the only evidence for the label while
      keeping the label, which teaches the model to guess.
    * **Cutout and erasing** -- same failure, for the same reason.
    """
    from torchvision import transforms

    normalise = [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]

    if not train:
        return transforms.Compose(
            [
                transforms.Resize(config.resize_to),
                transforms.CenterCrop(config.image_size),
                *normalise,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(config.resize_to),
            transforms.RandomResizedCrop(
                config.image_size,
                # Never crop away more than a fifth of the frame: the damage is
                # usually localised and a harder crop can remove it entirely.
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
            *normalise,
        ]
    )


def build_backbone(name: str, *, pretrained: bool = True):
    """Return ``(feature_extractor, embedding_dim)`` with the classifier removed."""
    import torch
    from torchvision import models

    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; available: {sorted(BACKBONES)}")

    weights_enum = getattr(models, BACKBONES[name]["weights"])
    weights = weights_enum.DEFAULT if pretrained else None
    model = getattr(models, name)(weights=weights)

    if name.startswith("resnet"):
        dim = model.fc.in_features
        model.fc = torch.nn.Identity()
    else:  # efficientnet
        dim = model.classifier[-1].in_features
        model.classifier = torch.nn.Identity()

    return model, dim


class ImagePathDataset:
    """Minimal dataset over a list of image paths.

    A corrupted or unreadable file yields a zero tensor and is recorded, rather
    than raising. Losing an entire encoding run to one bad JPEG would be a poor
    trade, and the inventory step has already verified decodability, so a
    failure here means something changed on disk.
    """

    def __init__(self, paths: list[Path], transform) -> None:
        self.paths = paths
        self.transform = transform
        self.failures: list[str] = []

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        import torch
        from PIL import Image

        path = self.paths[index]
        try:
            with Image.open(path) as image:
                return self.transform(image.convert("RGB")), True
        except Exception as exc:  # noqa: BLE001 - any read failure is recorded
            self.failures.append(f"{path.name}: {type(exc).__name__}")
            return torch.zeros(3, 224, 224), False


class VisionEncoder:
    """Frozen backbone producing one embedding per image."""

    def __init__(self, config: VisionEncoderConfig | None = None) -> None:
        self.config = config or VisionEncoderConfig()
        self._model = None
        self._dim = 0
        self.device = resolve_device(self.config.device)

    def _load(self) -> None:
        if self._model is not None:
            return
        logger.info(
            "loading vision backbone %s on %s", self.config.backbone, self.device
        )
        model, dim = build_backbone(self.config.backbone)
        model.eval().to(self.device)
        # Freeze this instance's parameters only. Gradients are disabled per
        # call site with `torch.no_grad()`, never process-wide: a global
        # `set_grad_enabled(False)` here would silently stop the damage model
        # from learning if it were trained later in the same process.
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._model = model
        self._dim = dim

    @property
    def embedding_dim(self) -> int:
        self._load()
        return self._dim

    def encode(
        self, image_paths: list[Path], *, show_progress: bool = True
    ) -> tuple[np.ndarray, VisionEncodingReport]:
        import time

        import torch
        from torch.utils.data import DataLoader

        self._load()
        started = time.perf_counter()

        dataset = ImagePathDataset(image_paths, build_transform(self.config, train=False))
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,  # image decoding on a 4-core box: workers cost more than they save
        )

        iterator = loader
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(loader, desc="encoding images", unit="batch")

        vectors: list[np.ndarray] = []
        n_failed = 0
        for batch, ok in iterator:
            with torch.no_grad():
                features = self._model(batch.to(self.device))
            if self.config.normalize:
                features = torch.nn.functional.normalize(features, p=2, dim=1)
            array = features.cpu().numpy().astype("float32")
            # A file that failed to load must not contribute a bogus embedding.
            array[~ok.numpy()] = 0.0
            n_failed += int((~ok.numpy()).sum())
            vectors.append(array)

        embeddings = np.vstack(vectors) if vectors else np.zeros((0, self._dim), "float32")
        report = VisionEncodingReport(
            n_images=len(image_paths),
            embedding_dim=embeddings.shape[1] if len(embeddings) else self._dim,
            backbone=self.config.backbone,
            n_failed=n_failed,
            failed_examples=dataset.failures[:10],
            seconds=time.perf_counter() - started,
            from_cache=False,
        )
        logger.info(
            "encoded %d images to %d dims in %.1fs (%.1f img/s); %d failed",
            report.n_images, report.embedding_dim, report.seconds,
            report.n_images / max(report.seconds, 1e-9), report.n_failed,
        )
        return embeddings, report

    def encode_cached(
        self,
        image_paths: list[Path],
        *,
        cache_dir: Path | None = None,
        refresh: bool = False,
    ) -> tuple[np.ndarray, VisionEncodingReport]:
        cache_dir = cache_dir or (get_paths().cache / "image")
        cache_dir.mkdir(parents=True, exist_ok=True)

        key = self.config.cache_key([str(p) for p in image_paths])
        vectors_path = cache_dir / f"{key}.npy"
        meta_path = cache_dir / f"{key}.json"

        if vectors_path.exists() and meta_path.exists() and not refresh:
            embeddings = np.load(vectors_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            report = VisionEncodingReport(**{**meta["report"], "from_cache": True})
            logger.info(
                "loaded %d cached image embeddings from %s", len(embeddings), vectors_path
            )
            return embeddings, report

        embeddings, report = self.encode(image_paths)
        np.save(vectors_path, embeddings)
        meta_path.write_text(
            json.dumps(
                {"config": asdict(self.config), "report": report.as_dict()}, indent=2
            ),
            encoding="utf-8",
        )
        return embeddings, report
