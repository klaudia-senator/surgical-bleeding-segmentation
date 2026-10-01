import torch

from surgical_bleeding.unet import UNetBinary


def test_unet_preserves_spatial_shape():
    model = UNetBinary(base_ch=8, dropout=0.1)
    output = model(torch.randn(2, 3, 65, 81))
    assert output.shape == (2, 1, 65, 81)
