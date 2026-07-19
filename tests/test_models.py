import pytest
import torch

from models import build_model


torch.set_num_threads(1)


@pytest.mark.parametrize("name", ["unet", "attention_unet", "transformer_unet", "proposed"])
def test_model_output_shape(name: str) -> None:
    config = {
        "name": name,
        "in_channels": 3,
        "out_channels": 1,
        "base_channels": 2,
    }
    if name in {"transformer_unet", "proposed"}:
        config.update(
            transformer_heads=1,
            transformer_depth=1,
            transformer_mlp_ratio=2.0,
            dropout=0.0,
        )
    model = build_model(config).eval()
    x = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 16, 16)
    assert torch.isfinite(y).all()
