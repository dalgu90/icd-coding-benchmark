import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from src.modules.activations import *
from src.modules.layers.label_wise_attn import LabelWiseAttn
from src.modules.layers.temporal_conv_net import TemporalConvNet
from src.modules.layers.word_embedding_layer import WordEmbeddingLayer
from src.utils.mapper import ConfigMapper
from src.utils.text_loggers import get_logger

logger = get_logger(__name__)


@ConfigMapper.map("models", "dcan")
class DCAN(nn.Module):
    """
    This class is used to create the DCAN model.
    References:
        Paper: https://aclanthology.org/2020.clinicalnlp-1.8/
        GitHub Repository: https://github.com/shaoxiongji/DCAN
    For the parameters related to convolutional layers, please see this:
    https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html.
    Args:
        num_classes (int): Number of classes (ICD codes).
        conv_channel_sizes (list): List of lists of integers. Each list
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
        add_emb_size_to_channel_sizes (bool): If True, you need not specify
                                              the input channel size. Please
                                              see the description of
                                              `conv_channel_sizes`.
        kernel_sizes (list): List of list of integers (same format as
                             `conv_channel_sizes`). Each integer represents the
                             kernel size/filter size of the respective
                             convolutional layer in `TemporalBlock` layer.
        strides (list): List of list of integers (same format as
                        `conv_channel_sizes`). Each integer represents the
                        stride of the respective convolutional layer in
                        `TemporalBlock` layer.
        paddings (list): List of list of integers (same format as
                         `conv_channel_sizes`). Each integer represents the
                         padding of the respective convolutional layer in
                         `TemporalBlock` layer. in DCAN, this value is set to
                         "(kernel_size - 1) * dilation_size".
        dilations (list): List of list of integers (same format as
                          `conv_channel_sizes`). Each integer represents the
                          dilation size of the respective convolutional layer
                          `TemporalBlock` layer.` In DCAN, this value is
                          "2^(temporal_block_level)".
        dropouts (list): List of list of floats (same format as
                         `conv_channel_sizes`). Each float represents the
                         dropout probability of the respective convolutional
                         `TemporalBlock` layer.
        weight_norm (bool): If True, apply weight normalization to the
                            convolutional layers.
        activation (str): Activation function to use. Should be one of "relu",
                          "elu", "leaky_relu".
    """

    def __init__(self, config):
        super(DCAN, self).__init__()
        logger.info(f"Initialising {self.__class__.__name__}")
        logger.debug(
            f"Initialising {self.__class__.__name__} with " f"config: {config}"
        )

        self.config = config

        self.word_embedding_layer = WordEmbeddingLayer(
            **config.word_representation_layer.params.init_params.as_dict()
        )
        if config.word_representation_layer.params.freeze_layer:
            self.freeze_layer(self.word_embedding_layer.embed)

        num_levels = len(config.kernel_sizes)
        num_inner_conv_levels = len(config.kernel_sizes[0])

        conv_channel_sizes = copy.deepcopy(config.conv_channel_sizes)
        if config.add_emb_size_to_channel_sizes:
            conv_channel_sizes[0] = [
                self.word_embedding_layer.embedding_size
            ] + conv_channel_sizes[0]
        dropouts = [
            [config.dropout for _ in range(num_inner_conv_levels)]
            for _ in range(num_levels)
        ]

        self.temporal_conv_net = TemporalConvNet(
            conv_channel_sizes_=conv_channel_sizes,
            kernel_sizes_=config.kernel_sizes,
            strides_=config.strides,
            paddings_=config.paddings,
            dilations_=config.dilations,
            dropouts_=dropouts,
            weight_norm=config.weight_norm,
            activation=config.activation,
        )

        self.linear_layer = nn.Linear(
            conv_channel_sizes[-1][-1], config.projection_size
        )
        self.activation = ConfigMapper.get_object(
            "activations", config.activation
        )()

        self.output_layer = OutputLayer(
            config.projection_size, config.num_classes
        )

        xavier_uniform_(self.linear_layer.weight)

    def forward(self, data):
        x = self.word_embedding_layer(data)
        hid_seq = self.temporal_conv_net(x.transpose(1, 2)).transpose(1, 2)
        hid_seq = self.activation(self.linear_layer(hid_seq))
        logits = self.output_layer(hid_seq)
        return logits

    def freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

    def get_input_attention(self):
        # Use the attention score computed in the forward pass
        return self.output_layer.label_wise_attn.alpha.cpu().detach().numpy()


class OutputLayer(nn.Module):
    def __init__(self, input_size, num_classes):
        super(OutputLayer, self).__init__()
        self.label_wise_attn = LabelWiseAttn(input_size, num_classes)

        self.final = nn.Linear(input_size, num_classes)
        xavier_uniform_(self.final.weight)

    def forward(self, x):
        m = self.label_wise_attn(x)
        logits = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        return logits
