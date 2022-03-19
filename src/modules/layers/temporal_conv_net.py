# flake8: noqa

import torch.nn as nn

from src.modules.activations import *
from src.modules.layers.temporal_block import TemporalBlock


class TemporalConvNet(nn.Module):
    """
    Stack of `TemporalBlock`s. Used in the DCAN model.
    References:
        Paper: https://arxiv.org/abs/2009.14578
        Repository: https://github.com/shaoxiongji/DCAN/blob/master/models.py#L114

    Args:
        conv_channel_sizes_ (list): List of lists of integers. Each list
                                   represents the channel sizes of convolutional
                                   layers in a `TemporalBlock`. So, for example,
                                   if the list is [[100, 600, 600],
                                                   [600, 600, 600]].
                                   the `TemporalConvNet` layer will have 2
                                   `TemporalBlock`s, each temporal block have
                                   2 convolutional layers:
                                   Conv(100, 600), Conv(600, 600) for the first
                                   one, and Conv(600, 600), Conv(600, 600). If
                                   the `add_emb_size_to_channel_sizes`, we don't
                                   have to pass the input channel size. So, in
                                   the above case, we can just pass
                                   [[600, 600], [600, 600, 600]].
        kernel_sizes_ (list): List of list of integers (same format as
                             `conv_channel_sizes`). Each integer represents the
                             kernel size/filter size of the respective
                             convolutional layer in `TemporalBlock` layer.
        strides_ (list): List of list of integers (same format as
                        `conv_channel_sizes`). Each integer represents the
                        stride of the respective convolutional layer in
                        `TemporalBlock` layer.
        paddings_ (list): List of list of integers (same format as
                         `conv_channel_sizes`). Each integer represents the
                         padding of the respective convolutional layer in
                         `TemporalBlock` layer. in DCAN, this value is set to
                         "(kernel_size - 1) * dilation_size".
        dilations_ (list): List of list of integers (same format as
                          `conv_channel_sizes`). Each integer represents the
                          dilation size of the respective convolutional layer
                          `TemporalBlock` layer.` In DCAN, this value is
                          "2^(temporal_block_level)".
        dropouts_ (list): List of list of floats (same format as
                         `conv_channel_sizes`). Each float represents the
                         dropout probability of the respective convolutional
                         `TemporalBlock` layer.
        weight_norm (bool): If True, apply weight normalization to the
                            convolutional layers.
        activation (str): Activation function to use. DCAN uses "relu".
    """

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
