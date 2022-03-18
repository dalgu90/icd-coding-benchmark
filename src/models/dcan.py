import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from src.modules.layers.label_wise_attn import LabelWiseAttn
from src.modules.layers.temporal_conv_net import TemporalConvNet
from src.modules.layers.word_embedding_layer import WordEmbeddingLayer
from src.utils.mapper import ConfigMapper


@ConfigMapper.map("models", "dcan")
class DCAN(nn.Module):
    def __init__(self, config):
        super(DCAN, self).__init__()
        self.config = config

        self.word_embedding_layer = WordEmbeddingLayer(
            **config.word_representation_layer.params.init_params.as_dict()
        )
        if config.word_representation_layer.params.freeze_layer:
            self.freeze_embedding_layer(
                self.word_embedding_layer.embed.parameters()
            )

        conv_channel_sizes = config.conv_channel_sizes
        if config.add_emb_size_to_channel_sizes:
            for dim_1 in conv_channel_sizes:
                dim_1 = self.word_embedding_layer.embedding_size + dim_1

        self.temporal_conv_net = TemporalConvNet(
            conv_channel_sizes_=conv_channel_sizes,
            kernel_sizes_=config.kernel_sizes,
            strides_=config.strides,
            paddings_=config.paddings,
            dilations_=config.dilations,
            dropouts_=config.dropouts,
            weight_norm=config.weight_norm,
            activation=config.activation,
        )

        self.linear_layer = nn.Linear(
            conv_channel_sizes[-1][-1], config.projection_size
        )
        self.activation = ConfigMapper.get_object(
            "activations", config.activation
        )

        self.output_layer = OutputLayer(
            config.projection_size, config.num_classes
        )

        xavier_uniform_(self.linear_layer.weight)

    def forward(self, data):
        x = self.word_embedding_layer(data)
        hid_seq = self.temporal_conv_net(x.transpose(1, 2)).transpose(1, 2)
        hid_seq = self.activation(self.lin(hid_seq))
        logits = self.output_layer(hid_seq)
        return logits

    def freeze_embedding_layer(self):
        for param in self.word_embedding_layer.embed.parameters():
            param.requires_grad = False


class OutputLayer(nn.Module):
    def __init__(self, input_size, num_classes):
        super(OutputLayer, self).__init__()
        self.label_wise_attn = LabelWiseAttn(self, input_size, num_classes)

        self.final = nn.Linear(input_size, num_classes)
        xavier_uniform_(self.final.weight)

    def forward(self, x):
        m = self.label_wise_attn(x)
        logits = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        return logits
