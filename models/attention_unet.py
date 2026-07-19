from .unet import ConfigurableUNet


class AttentionUNet(ConfigurableUNet):
    def __init__(self, **kwargs):
        super().__init__(use_attention=True, use_transformer=False, **kwargs)
