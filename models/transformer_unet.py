from .unet import ConfigurableUNet


class TransformerUNet(ConfigurableUNet):
    def __init__(self, **kwargs):
        super().__init__(use_attention=False, use_transformer=True, **kwargs)
