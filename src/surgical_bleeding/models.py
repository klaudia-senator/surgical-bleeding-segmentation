from __future__ import annotations

import importlib


def make_smp_model(name: str, encoder: str):
    import segmentation_models_pytorch as smp
    normalized = name.lower().replace("+", "plus")
    if normalized in {"unetplusplus", "unetpp", "unetplusplus"}:
        return smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1, activation=None)
    if normalized in {"deeplabv3plus", "deeplab"}:
        return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1, activation=None)
    raise ValueError("Model must be unetplusplus or deeplabv3plus")


def load_owned_unet_class(import_path: str = "src.models.unet:UNetBinary"):
    """Load Klaudia's original model without inventing a replacement implementation."""
    module_name, class_name = import_path.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "The original UNetBinary source is not distributed in this public repository. "
            "Provide the owned src/models/unet.py implementation and use --model-import "
            "src.models.unet:UNetBinary. The SMP cross-validation workflow is fully runnable without it."
        ) from exc

