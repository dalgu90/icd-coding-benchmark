import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.nn.utils import weight_norm as weight_norm_

from src.utils.mapper import ConfigMapper


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class ConvTemporalSubBlock(nn.Module):
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
        conv_layer_output = self.conv_temporal_sub_blocks[0](x)
        if len(self.conv_temporal_sub_blocks) > 0:
            for conv_temporal_sub_block in self.conv_temporal_sub_blocks[1:]:
                conv_layer_output = conv_temporal_sub_block(conv_layer_output)
        res = x if self.downsample is None else self.downsample(x)
        return self.output_activation(conv_layer_output + res)
