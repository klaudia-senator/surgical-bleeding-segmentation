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


def load_owned_unet_class(import_path: str = "surgical_bleeding.unet:UNetBinary"):
    """Load the in-house binary U-Net or another explicitly requested compatible class."""
    module_name, class_name = import_path.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(f"Could not import model class {import_path!r}") from exc
