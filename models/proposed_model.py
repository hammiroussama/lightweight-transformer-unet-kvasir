from .unet import ConfigurableUNet


class LightweightTransformerAttentionUNet(ConfigurableUNet):
    def __init__(self, **kwargs):
        super().__init__(use_attention=True, use_transformer=True, **kwargs)
