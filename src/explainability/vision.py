"""Grad-CAM for the damage model.

What it does, stated precisely, because "the model looked here" is a claim that
is easy to overstate:

Grad-CAM takes the activations of the last convolutional layer -- a stack of
channels, each a coarse spatial map of some learned pattern -- and asks how
much the score for a chosen class would change if each channel's activation
rose. Those gradients, pooled over space, become per-channel weights. The
weighted sum of the channels, with negatives clipped, is a map of where the
evidence for that class was strongest.

Two honest caveats:

* The map is at the spatial resolution of the final convolutional layer, which
  for ResNet-18 at 224 pixels is 7x7. Everything finer than a 32-pixel block is
  interpolation, so a sharply-outlined heatmap is a rendering artefact, not
  precision.
* A highlighted region is where the evidence was, not why. Grad-CAM cannot
  distinguish "the model recognised a crushed bumper" from "the model
  recognised the kind of background that crushed-bumper photos tend to have."
  That distinction is exactly what the error analysis looks for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.models.vision.encoder import IMAGENET_MEAN, IMAGENET_STD
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GradCAMResult:
    heatmap: np.ndarray
    """Shape (H, W), normalised to [0, 1] and upsampled to the input size."""

    predicted_class: int
    predicted_probability: float
    target_class: int
    true_class: int | None = None
    image_path: str | None = None

    def as_dict(self) -> dict:
        return {
            "predicted_class": self.predicted_class,
            "predicted_probability": self.predicted_probability,
            "target_class": self.target_class,
            "true_class": self.true_class,
            "image_path": self.image_path,
            "heatmap_shape": list(self.heatmap.shape),
        }


class GradCAM:
    """Grad-CAM over a backbone plus head.

    The backbone and head are separate modules in this project, so the hook
    goes on the backbone's last convolutional block and the gradient flows back
    from the head's class logit.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module, target_layer: nn.Module | None = None):
        self.backbone = backbone
        self.head = head
        self.target_layer = target_layer or self._find_target_layer(backbone)

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: list = []

    @staticmethod
    def _find_target_layer(backbone: nn.Module) -> nn.Module:
        """The last convolutional block: the deepest layer that still has space."""
        if hasattr(backbone, "layer4"):
            return backbone.layer4
        for module in reversed(list(backbone.children())):
            if isinstance(module, nn.Sequential):
                return module
        raise ValueError("could not locate a convolutional block to hook")

    def __enter__(self) -> GradCAM:
        self._handles = [
            self.target_layer.register_forward_hook(self._save_activations),
            self.target_layer.register_full_backward_hook(self._save_gradients),
        ]
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _save_activations(self, module, inputs, output) -> None:
        del module, inputs
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output) -> None:
        del module, grad_input
        self._gradients = grad_output[0].detach()

    def explain(
        self, image_tensor: torch.Tensor, *, target_class: int | None = None
    ) -> GradCAMResult:
        """Produce a heatmap for one image.

        ``target_class`` defaults to the predicted class. Passing the true
        class instead answers a different and often more useful question: where
        the model looked for evidence of the right answer when it got it wrong.
        """
        was_backbone_training = self.backbone.training
        was_head_training = self.head.training
        self.backbone.eval()
        self.head.eval()

        batch = image_tensor.unsqueeze(0) if image_tensor.dim() == 3 else image_tensor

        # Gradients are needed here even though both modules are frozen for
        # training: Grad-CAM differentiates the output with respect to
        # activations, not parameters.
        with torch.enable_grad():
            features = self.backbone(batch)
            logits = self.head(features)
            probabilities = torch.softmax(logits, dim=1)

            predicted = int(logits.argmax(dim=1).item())
            target = predicted if target_class is None else int(target_class)

            self.backbone.zero_grad(set_to_none=True)
            self.head.zero_grad(set_to_none=True)
            logits[0, target].backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "no activations captured; use GradCAM as a context manager so the "
                "hooks are registered"
            )

        # Channel weights: how much raising each channel would raise the score.
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        # Negative contributions argue against the class; the convention is to
        # show only the evidence for it.
        cam = torch.relu(cam)

        cam = torch.nn.functional.interpolate(
            cam, size=batch.shape[-2:], mode="bilinear", align_corners=False
        )
        heatmap = cam.squeeze().cpu().numpy()
        span = heatmap.max() - heatmap.min()
        # A uniformly zero map means the gradient never favoured this class;
        # normalising it would manufacture structure that is not there.
        heatmap = (heatmap - heatmap.min()) / span if span > 1e-12 else np.zeros_like(heatmap)

        self.backbone.train(was_backbone_training)
        self.head.train(was_head_training)

        return GradCAMResult(
            heatmap=heatmap,
            predicted_class=predicted,
            predicted_probability=float(probabilities[0, predicted].item()),
            target_class=target,
        )


def denormalise(image_tensor: torch.Tensor) -> np.ndarray:
    """Undo the ImageNet normalisation, for display alongside a heatmap."""
    array = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    array = array * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(array, 0.0, 1.0)


def save_overlay(
    image_tensor: torch.Tensor,
    heatmap: np.ndarray,
    path: Path,
    *,
    title: str = "",
    alpha: float = 0.45,
) -> Path:
    """Write an image with its heatmap overlaid, plus the original beside it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original = denormalise(image_tensor)

    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original)
    axes[0].set_title("input")
    axes[1].imshow(original)
    axes[1].imshow(heatmap, cmap="jet", alpha=alpha)
    axes[1].set_title("Grad-CAM")
    for axis in axes:
        axis.axis("off")
    if title:
        figure.suptitle(title, fontsize=10)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return path
