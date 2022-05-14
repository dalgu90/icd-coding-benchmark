import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.utils import weight_norm as weight_norm_

from src.modules.activations import *
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

        conv_channel_sizes = config.conv_channel_sizes
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


class WordEmbeddingLayer(nn.Module):
    """
    A Word Embedding Layer. This layer loads a pre-trained word embedding matrix
    , and copies its weights to an nn.Embedding layer.

    Args:
        embed_dir (str): A directory containing the pre-trained word embedding
                         matrix, among other things. Please see
                         https://github.com/dalgu90/icd-coding-benchmark/blob/main/src/modules/embeddings.py#L17
                         for more details.
        dropout (float): The dropout probability.
    """

    def __init__(self, embed_dir, dropout):
        super(WordEmbeddingLayer, self).__init__()
        logger.debug(
            f"Initialising {self.__class__.__name__} with "
            f"embed_dir = {embed_dir}, dropout = {dropout}"
        )

        # Note: This should be changed, since we won't always use Word2Vec.
        embedding_cls = ConfigMapper.get_object("embeddings", "word2vec")

        W = torch.Tensor(embedding_cls.load_emb_matrix(embed_dir))
        self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
        self.embed.weight.data = W.clone()

        self.embedding_size = self.embed.embedding_dim

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedding = self.embed(x)
        x = self.dropout(embedding)
        return x


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
        Repository:
        https://github.com/shaoxiongji/DCAN/blob/master/models.py#L81

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


class TemporalConvNet(nn.Module):
    """
    Stack of `TemporalBlock`s. Used in the DCAN model.
    References:
        Paper: https://arxiv.org/abs/2009.14578
        Repository:
        https://github.com/shaoxiongji/DCAN/blob/master/models.py#L114

    Args:
        conv_channel_sizes_ (list): List of lists of integers. Each list
                                    represents the channel sizes of
                                    convolutional layers in a `TemporalBlock`.
                                    So, for example, if the list is
                                    [
                                        [100, 600, 600],
                                        [600, 600, 600]
                                    ],
                                    the `TemporalConvNet` layer will have 2
                                    `TemporalBlock`s, each temporal block have
                                    2 convolutional layers:
                                    Conv(100, 600), Conv(600, 600) for the first
                                    one, and Conv(600, 600), Conv(600, 600). If
                                    the `add_emb_size_to_channel_sizes`, we
                                    don't have to pass the input channel size.
                                    So, in the above case, we can just pass
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
        logger.debug(
            f"Initialising {self.__class__.__name__} with "
            f"conv_channel_sizes_ = {conv_channel_sizes_}, "
            f"kernel_sizes_ = {kernel_sizes_}, "
            f"strides_ = {strides_}, paddings_ = {paddings_}, "
            f"dilations_ = {dilations_}, dropouts_ = {dropouts_}, "
            f"weight_norm = {weight_norm}, activation = {activation}"
        )

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


class LabelWiseAttn(nn.Module):
    """
    A Label-wise Attention layer (as implemented in CAML, DCAN, etc.).
    References:
        Papers: https://arxiv.org/abs/1802.05695 (Section 2.2)
        Repository:
        https://github.com/jamesmullenbach/caml-mimic/blob/master/learn/models.py#L184

    Args:
        input_size (int): The size of the input, i.e., the number of channels
                          if the output is from a convolutional layer/embedding
                          size if the output is from a fully connected layer.
        num_classes (int): The number of classes.
    """

    def __init__(self, input_size, num_classes):
        super(LabelWiseAttn, self).__init__()
        logger.debug(
            f"Initialising {self.__class__.__name__} with "
            f"input size = {input_size}, num_classes = {num_classes}"
        )

        self.U = nn.Linear(input_size, num_classes)
        xavier_uniform_(self.U.weight)

    def forward(self, x):
        att = self.U.weight.matmul(x.transpose(1, 2))  # [bs, Y, seq_len]
        alpha = F.softmax(att, dim=2)
        m = alpha.matmul(x)  # [bs, Y, dim]
        return m


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
