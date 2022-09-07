"""
ICD Coding from Clinical Text Using Multi-Filter Residual Convolutional Neural
Network, 2020
https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network
"""

from math import floor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform

from anemic.utils.mapper import ConfigMapper
from anemic.utils.model_utils import load_lookups
from anemic.utils.text_loggers import get_logger

logger = get_logger(__name__)


class WordRep(nn.Module):
    def __init__(self, config):
        super(WordRep, self).__init__()

        # Embedding layer
        embedding_cls = ConfigMapper.get_object("embeddings", "word2vec")
        W = torch.Tensor(embedding_cls.load_emb_matrix(config.word2vec_dir))
        self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
        self.embed.weight.data = W.clone()

        self.feature_size = self.embed.embedding_dim

        # Embedding dropout
        self.embed_drop = nn.Dropout(p=config.dropout)

        # Conv layer filter sizes
        self.conv_dict = {
            1: [self.feature_size, config.num_filter_maps],
            2: [self.feature_size, 100, config.num_filter_maps],
            3: [self.feature_size, 150, 100, config.num_filter_maps],
            4: [self.feature_size, 200, 150, 100, config.num_filter_maps],
        }

    def forward(self, x):
        x = self.embed(x)
        x = self.embed_drop(x)
        return x


class OutputLayer(nn.Module):
    def __init__(self, Y, input_size):
        super(OutputLayer, self).__init__()

        self.U = nn.Linear(input_size, Y)
        xavier_uniform(self.U.weight)

        self.final = nn.Linear(input_size, Y)
        xavier_uniform(self.final.weight)

    def forward(self, x):
        self.alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        m = self.alpha.matmul(x)
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        return y


@ConfigMapper.map("models", "MultiCNN")
class MultiCNN(nn.Module):
    def __init__(self, config):
        super(MultiCNN, self).__init__()

        Y = config.num_classes
        self.dicts = load_lookups(
            dataset_dir=config.dataset_dir,
            mimic_dir=config.mimic_dir,
            static_dir=config.static_dir,
            word2vec_dir=config.word2vec_dir,
            version=config.version,
        )

        self.word_rep = WordRep(config)

        if config.filter_size.find(",") == -1:
            self.filter_num = 1
            filter_size = int(config.filter_size)
            self.conv = nn.Conv1d(
                self.word_rep.feature_size,
                config.num_filter_maps,
                kernel_size=filter_size,
                padding=floor(filter_size / 2),
            )
            xavier_uniform(self.conv.weight)
        else:
            self.filter_num = len(config.filter_size)
            self.conv = nn.ModuleList()
            for filter_size in config.filter_size:
                filter_size = int(filter_size)
                tmp = nn.Conv1d(
                    self.word_rep.feature_size,
                    config.num_filter_maps,
                    kernel_size=filter_size,
                    padding=floor(filter_size / 2),
                )
                xavier_uniform(tmp.weight)
                self.conv.add_module("conv-{}".format(filter_size), tmp)

        self.output_layer = OutputLayer(
            Y, self.filter_num * config.num_filter_maps
        )

    def forward(self, x, target, text_inputs):
        x = self.word_rep(x, target, text_inputs)
        x = x.transpose(1, 2)

        if self.filter_num == 1:
            x = torch.tanh(self.conv(x).transpose(1, 2))
        else:
            conv_result = []
            for tmp in self.conv:
                conv_result.append(torch.tanh(tmp(x).transpose(1, 2)))
            x = torch.cat(conv_result, dim=2)

        y, loss = self.output_layer(x, target, text_inputs)
        return y, loss


class ResidualBlock(nn.Module):
    def __init__(
        self, inchannel, outchannel, kernel_size, stride, use_res, dropout
    ):
        super(ResidualBlock, self).__init__()

        self.left = nn.Sequential(
            nn.Conv1d(
                inchannel,
                outchannel,
                kernel_size=kernel_size,
                stride=stride,
                padding=floor(kernel_size / 2),
                bias=False,
            ),
            nn.BatchNorm1d(outchannel),
            nn.Tanh(),
            nn.Conv1d(
                outchannel,
                outchannel,
                kernel_size=kernel_size,
                stride=1,
                padding=floor(kernel_size / 2),
                bias=False,
            ),
            nn.BatchNorm1d(outchannel),
        )

        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    inchannel,
                    outchannel,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(outchannel),
            )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out


@ConfigMapper.map("models", "ResCNN")
class ResCNN(nn.Module):
    def __init__(self, config):
        super(ResCNN, self).__init__()

        Y = config.num_classes
        self.dicts = load_lookups(
            dataset_dir=config.dataset_dir,
            mimic_dir=config.mimic_dir,
            static_dir=config.static_dir,
            word2vec_dir=config.word2vec_dir,
            version=config.version,
        )

        self.word_rep = WordRep(config)

        self.conv = nn.ModuleList()
        conv_dimension = self.word_rep.conv_dict[config.conv_layer]
        for idx in range(config.conv_layer):
            tmp = ResidualBlock(
                conv_dimension[idx],
                conv_dimension[idx + 1],
                int(config.filter_size),
                1,
                True,
                config.dropout,
            )
            self.conv.add_module("conv-{}".format(idx), tmp)

        self.output_layer = OutputLayer(Y, config.num_filter_maps)

    def forward(self, x):
        x = self.word_rep(x)
        x = x.transpose(1, 2)
        for conv in self.conv:
            x = conv(x)
        x = x.transpose(1, 2)
        y, loss = self.output_layer(x)
        return y, loss


@ConfigMapper.map("models", "multirescnn")
class MultiResCNN(nn.Module):
    def __init__(self, config):
        super(MultiResCNN, self).__init__()

        Y = config.num_classes
        self.dicts = load_lookups(
            dataset_dir=config.dataset_dir,
            mimic_dir=config.mimic_dir,
            static_dir=config.static_dir,
            word2vec_dir=config.word2vec_dir,
            version=config.version,
        )

        self.word_rep = WordRep(config)

        self.conv = nn.ModuleList()

        self.filter_num = len(config.filter_size)
        for filter_size in config.filter_size:
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(
                self.word_rep.feature_size,
                self.word_rep.feature_size,
                kernel_size=filter_size,
                padding=floor(filter_size / 2),
            )
            xavier_uniform(tmp.weight)
            one_channel.add_module("baseconv", tmp)

            conv_dimension = self.word_rep.conv_dict[config.conv_layer]
            for idx in range(config.conv_layer):
                tmp = ResidualBlock(
                    conv_dimension[idx],
                    conv_dimension[idx + 1],
                    filter_size,
                    1,
                    True,
                    config.dropout,
                )
                one_channel.add_module("resconv-{}".format(idx), tmp)
            self.conv.add_module("channel-{}".format(filter_size), one_channel)

        self.output_layer = OutputLayer(
            Y, self.filter_num * config.num_filter_maps
        )

    def forward(self, x):
        x = self.word_rep(x)
        x = x.transpose(1, 2)
        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)
        y = self.output_layer(x)
        return y

    def get_input_attention(self):
        # Use the attention score computed in the forward pass
        return self.output_layer.alpha.cpu().detach().numpy()
