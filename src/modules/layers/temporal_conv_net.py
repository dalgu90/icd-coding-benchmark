import torch.nn as nn

from src.modules.layers.temporal_block import TemporalBlock


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        conv_channel_sizes_,
        kernel_sizes_,
        strides_,
        paddings_,
        dilations_,
        dropouts_,
        weight_norm=True,
        activation="relu",
    ):
        super(TemporalConvNet, self).__init__()
        self.temporal_blocks = nn.ModuleList(
            [
                TemporalBlock(
                    conv_channel_sizes=conv_channel_sizes,
                    kernel_sizes=kernel_sizes,
                    strides=strides,
                    paddings=paddings,
                    dilations=dilations,
                    dropouts=dropouts,
                    weight_norm=weight_norm,
                    activation=activation,
                )
                for (
                    conv_channel_sizes,
                    kernel_sizes,
                    strides,
                    paddings,
                    dilations,
                    dropouts,
                ) in zip(
                    conv_channel_sizes_,
                    kernel_sizes_,
                    strides_,
                    paddings_,
                    dilations_,
                    dropouts_,
                )
            ]
        )

    def forward(self, x):
        for temporal_block in self.temporal_blocks:
            x = temporal_block(x)
        return x
