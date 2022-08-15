# flake8: noqa

import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.nn.utils import weight_norm as weight_norm_

from src.utils.mapper import ConfigMapper
from src.utils.text_loggers import get_logger

logger = get_logger(__name__)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class ConvTemporalSubBlock(nn.Module):
    """
    A simple temporal convolutional block. Adapted from
    https://github.com/shaoxiongji/DCAN/blob/master/models.py#L84-L88. This
    layer has a dilated convolutional layer, a `chomp1d` layer, followed by
    activation and dropout. For the parameters related to convolutional layers,
    please see this:
    https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html.

    Args:
        in_channels (int): The number of input channels in the convolutional
                           layer.
        out_channels (int): The number of output channels in the convolutional
                            layer.
        kernel_size (int): The size of the kernel in the convolutional layer.
        stride (int): The stride of the convolutional layer.
        padding (int): The padding of the convolutional layer.
        dilation (int): The dilation size of the convolutional layer.
        dropout (float): The dropout probability.
        weight_norm (bool): Whether to apply weight normalization to the
                            convolutional layer.
        activation (str): The activation function to use. DCAN uses "relu".
                          For all available activations, see
                          https://github.com/dalgu90/icd-coding-benchmark/blob/main/src/modules/activations.py.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        dropout=0.2,
        weight_norm=True,
        activation="relu",
    ):
        super(ConvTemporalSubBlock, self).__init__()
        logger.debug(
            f"Initialising {self.__class__.__name__} with "
            f"in_channels = {in_channels}, out_channels = "
            f"{out_channels}, kernel_size = {kernel_size}, "
            f"stride = {stride}, padding = {padding}, "
            f"dilation = {dilation}, dropout = {dropout}, "
            f"weight_norm = {weight_norm}, activation = {activation}"
        )

        self.conv_layer = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        if weight_norm:
            self.conv_layer = weight_norm_(self.conv_layer)
        self.chomp1d = Chomp1d(padding)
        self.activation = ConfigMapper.get_object("activations", activation)()
        self.dropout = nn.Dropout(dropout)

        self.__init_weights__()

    def __init_weights__(self):
        xavier_uniform_(self.conv_layer.weight)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.chomp1d(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class TemporalBlock(nn.Module):
    """
    A Temporal Block containing stacks of `ConvTemporalSubBlocks`, followed
    by activation.
    References:
        Paper: https://arxiv.org/abs/2009.14578
        Repository: https://github.com/shaoxiongji/DCAN/blob/master/models.py#L81

    Args:
        conv_channel_sizes (list): List of integers, with channel sizes of
                                   convolutional layers. For example, if the
                                   list is [100, 200, 300], there will be two
                                   convolutional layers: Conv1d(100, 200) and
                                   Conv1d(200, 300).
        kernel_sizes (list): List of integers, with kernel sizes of every
                             `ConvTemporalSubBlock`.
        strides (list): List of integers, with strides of convolutional layers.
        paddings (list): List of integers, with paddings of every
                         `ConvTemporalSubBlock`.
        dilations (list): List of integers, with dilation sizes of every
                          `ConvTemporalSubBlock`.
        dropouts (list): List of floats, with dropout probabilities of every
                         `ConvTemporalSubBlock`.
        weight_norm (bool): Whether to apply weight normalization to every
                             convolutional layer. DCAN uses weight norm.
        activation (str): The activation function to use. DCAN uses "relu".
    """

    def __init__(
        self,
        conv_channel_sizes,
        kernel_sizes,
        strides,
        paddings,
        dilations,
        dropouts,
        weight_norm=True,
        activation="relu",
    ):
        super(TemporalBlock, self).__init__()
        conv_channel_size_pairs = list(
            zip(conv_channel_sizes[:-1], conv_channel_sizes[1:])
        )

        self.conv_temporal_sub_blocks = nn.ModuleList(
            [
                ConvTemporalSubBlock(
                    in_channels=conv_channel_size_pair[0],
                    out_channels=conv_channel_size_pair[1],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    dropout=dropout,
                    weight_norm=weight_norm,
                    activation=activation,
                )
                for (
                    conv_channel_size_pair,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    dropout,
                ) in zip(
                    conv_channel_size_pairs,
                    kernel_sizes,
                    strides,
                    paddings,
                    dilations,
                    dropouts,
                )
            ]
        )

        self.downsample = (
            nn.Conv1d(conv_channel_sizes[0], conv_channel_sizes[-1], 1)
            if conv_channel_sizes[0] != conv_channel_sizes[-1]
            else None
        )
        self.output_activation = ConfigMapper.get_object(
            "activations", activation
        )()

        self.init_weights()

    def init_weights(self):
        if self.downsample is not None:
            xavier_uniform_(self.downsample.weight)

    def forward(self, x):
        conv_layer_output = x
        for conv_temporal_sub_block in self.conv_temporal_sub_blocks:
            conv_layer_output = conv_temporal_sub_block(conv_layer_output)
        res = x if self.downsample is None else self.downsample(x)
        return self.output_activation(conv_layer_output + res)
