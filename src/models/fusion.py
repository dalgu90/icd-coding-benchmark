"""
    Fusion model (Luo et al. 2021)
    https://github.com/machinelearning4health/Fusion-Towards-Automated-ICD-Coding
"""

from math import floor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform

from src.utils.caml_utils import load_lookups
from src.utils.mapper import ConfigMapper
from src.utils.text_loggers import get_logger

logger = get_logger(__name__)


class WordRep(nn.Module):
    def __init__(self, config):
        super(WordRep, self).__init__()

        # make embedding layer
        embedding_cls = ConfigMapper.get_object("embeddings", "word2vec")
        W = torch.Tensor(embedding_cls.load_emb_matrix(config.word2vec_dir))
        self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
        self.embed.weight.data = W.clone()

        self.feature_size = self.embed.embedding_dim

        self.embed_drop = nn.Dropout(p=config.dropout)

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


class AttentionBolckV2(nn.Module):
    def __init__(self, inchannel, pool_size, is_max_pool=True):
        super(AttentionBolckV2, self).__init__()
        self.is_max_pool = is_max_pool
        self.pool_size = pool_size
        self.att_conv = nn.Sequential(
            nn.Conv1d(inchannel, 1, kernel_size=1, stride=1, bias=False),
        )
        self.squeeze_pool = (
            nn.MaxPool1d(pool_size, pool_size, return_indices=True)
            if is_max_pool
            else nn.AvgPool1d(pool_size, pool_size)
        )

    def forward(self, x):
        if self.is_max_pool:
            if x.shape[2] % self.pool_size != 0:
                x = torch.nn.functional.pad(
                    x, [0, (self.pool_size - (x.shape[2] % self.pool_size))]
                )
            att = self.att_conv(x)
            att = att.view(att.shape[0], att.shape[1], -1, self.pool_size)
            att = torch.softmax(att, dim=3)
            x = x.view(x.shape[0], x.shape[1], -1, self.pool_size)
            x = x * att
            x = torch.sum(x, dim=3)
        else:
            att = self.att_conv(x)
            x = x * att
            x = self.squeeze_pool(x)
        return x


class ResidualBlockHidden(nn.Module):
    def __init__(
        self,
        inchannel,
        outchannel,
        kernel_size,
        stride,
        use_res,
        dropout,
        use_layer_norm=False,
        is_relu=True,
    ):
        super(ResidualBlockHidden, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(
                inchannel,
                outchannel,
                kernel_size=kernel_size,
                stride=stride,
                padding=int(floor(kernel_size / 2)),
                bias=False,
            ),
            nn.GroupNorm(1, outchannel)
            if use_layer_norm
            else nn.BatchNorm1d(outchannel),
            nn.Tanh() if not is_relu else nn.LeakyReLU(),
            nn.Conv1d(
                outchannel,
                outchannel,
                kernel_size=kernel_size,
                stride=1,
                padding=int(floor(kernel_size / 2)),
                bias=False,
            ),
            nn.GroupNorm(1, outchannel)
            if use_layer_norm
            else nn.BatchNorm1d(outchannel),
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
                nn.GroupNorm(1, outchannel)
                if use_layer_norm
                else nn.BatchNorm1d(outchannel),
            )
        self.dropout = nn.Dropout(p=dropout)
        self.out_activation = nn.Tanh() if not is_relu else nn.LeakyReLU()

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = self.out_activation(out)
        out = self.dropout(out)
        return out


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """Forward propagation.

        Args:
                q: Queries tensor with shape [B, L_q, D_q]
                k: Keys tensor with shape [B, L_k, D_k]
                v: Values tensor with shape [B, L_v, D_v]，generally k
                scale: scale factor, a floating-point scalar
                attn_mask: Masking tensor with shape [B, L_q, L_k]

        Returns:
                Context tensor and attention tensor
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask
        )

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention


class EncoderLayer(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2018, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(
            model_dim, ffn_dim, dropout
        )

    def forward(self, inputs, attn_mask=None):
        # self attention
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention


class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        """Initializer

        Args:
            d_model: A scalar. The dimension of the model, defaults to 512
            max_seq_len: a scalar. Maximum length of text sequence
        """
        super(PositionalEncoding, self).__init__()

        # According to the formula given by the paper, construct the PE matrix
        position_encoding = np.array(
            [
                [
                    pos / np.power(10000, 2.0 * (j // 2) / d_model)
                    for j in range(d_model)
                ]
                for pos in range(max_seq_len)
            ]
        )
        # Use sin for even columns and cos for odd columns
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = torch.from_numpy(
            position_encoding.astype(np.float32)
        )
        # In the first row of the PE matrix, add a vector with all 0s,
        # representing the positional encoding of this `PAD`. `UNK` is often
        # added to the word embedding, which represents the word embedding of
        # the position word, the two are very similar. So why do you need this
        # extra PAD encoding? Quite simply, since the text sequences are of
        # varying lengths, we need to align. For short sequences, we use 0 to
        # complete at the end, and we also need the encoding of these completion
        # positions, which is the position encoding corresponding to `PAD`.
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))

        # Embedding operation, +1 is because the encoding of the completion
        # position of `PAD` has been added. If the dictionary adds `UNK` in Word
        # embedding, we also need +1. Look, the two are very similar
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(
            position_encoding, requires_grad=False
        )

    def forward(self, input_len):
        """Forward propagation.

        Args:
          input_len: A tensor with shape [BATCH_SIZE, 1]. The value of each
                     tensor represents the corresponding length in this batch
                     of text sequences.

        Returns: Returns the position code of this batch of sequences, aligned.
        """

        # Find the maximum length of this batch of sequences
        max_len = torch.max(input_len)
        tensor = torch.LongTensor
        # Align the position of each sequence and add 0 after the original
        # sequence position. Here the range starts from 1 also because it is
        # necessary to avoid the position of PAD(0).
        pos = np.zeros([len(input_len), max_len])
        for ind, length in enumerate(input_len):
            for pos_ind in range(1, length + 1):
                pos[ind, pos_ind - 1] = pos_ind
        input_pos = tensor(pos)
        if input_len.is_cuda:
            input_pos = input_pos.cuda()
        return self.position_encoding(input_pos), input_pos


class EncoderHidden(nn.Module):
    """
    Feature aggregation layer of Fusion model.
    Arguments `vocab_size` and `gpu` are removed when porting.
    """

    def __init__(
        self,
        max_seq_len,
        num_layers=1,
        model_dim=256,
        num_heads=4,
        ffn_dim=1024,
        dropout=0.0,
    ):
        super(EncoderHidden, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(model_dim, num_heads, ffn_dim, 0.0)
                for _ in range(num_layers)
            ]
        )
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, output):
        input_len = torch.LongTensor([x.shape[0] for x in output]).cuda()
        output_pos, ind_pos = self.pos_embedding(input_len.unsqueeze(1))
        output += output_pos
        attentions = []
        outputs = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output)
            attentions.append(attention)
            outputs.append(output)
        output = torch.tanh(output)
        output = self.dropout(output)
        return output


@ConfigMapper.map("models", "Fusion")
class Fusion(nn.Module):
    """
    This class is for the Fusion model suggested in the following paper
    (MultiResCNNHidden in the repo).
    We removed the elmo from the original code since it's unnecessary.
    Reference:
        Paper: Junyu Luo, et al., Fusion: Towards Automated ICD Coding via
               Feature Compression, in Findings of ACL 2021,
               https://aclanthology.org/2021.findings-acl.184/
        Repository: https://github.com/machinelearning4health/Fusion
    Args (in config):
        num_classes (int): Number of classes (ICD codes).
        dropout (int): The dropout ratio after word embedding
        filter_size (string): The concatenation of filter size of each channel
        use_attention_pool (bool): Flag to use attetion-based soft pooling layer
        pool_size (int): The pooling size of attention-base soft pooling layer
        conv_layer (int): The number of residual convolution blocks
        use_layer_norm (bool): Flag to use group norm in the residual blocks
        use_relu (bool): Flag to use leaky relu (or tanh) in the residual blocks
        use_transformer (bool): Flag to use feature aggregation layer
        max_length (int): The maximum length of the input text. Used by the
                          positional encoding inside feature aggregation
        transfer_layer (int): The number of transformer layer in the feature
                              aggregation layer
        transfer_attention_head (int): The number of heads in the transformer
                                       layers
        transfer_fsize (int): The dimension of the feedforward layer in the
                              transformer layers
        num_filter_maps (int): The feature dimension of the last feature
                               aggregation layer and the label attention layer

    """

    def __init__(self, config):
        super(Fusion, self).__init__()
        logger.info("Initialising %s", self.__class__.__name__)
        logger.debug(
            "Initialising %s with config: %s", self.__class__.__name__, config
        )
        self.config = config

        # From CAML implementation
        self.Y = config.num_classes
        self.dicts = load_lookups(
            dataset_dir=config.dataset_dir,
            mimic_dir=config.mimic_dir,
            static_dir=config.static_dir,
            word2vec_dir=config.word2vec_dir,
            version=config.version,
        )

        self.word_rep = WordRep(config)

        self.conv = nn.ModuleList()
        filter_sizes = config.filter_size
        self.relu = nn.ReLU(inplace=True)
        self.use_transformer = config.use_transformer
        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(
                self.word_rep.feature_size,
                self.word_rep.feature_size,
                kernel_size=filter_size,
                padding=int(floor(filter_size / 2)),
            )
            xavier_uniform(tmp.weight)
            one_channel.add_module("baseconv", tmp)
            if config.use_attention_pool:
                tmp = AttentionBolckV2(
                    self.word_rep.feature_size, config.pool_size, True
                )
                one_channel.add_module("basevonb-pool", tmp)

            conv_dimension = self.word_rep.conv_dict[config.conv_layer]
            for idx in range(config.conv_layer):
                tmp = ResidualBlockHidden(
                    conv_dimension[idx],
                    conv_dimension[idx + 1],
                    filter_size,
                    1,
                    True,
                    config.dropout if config.use_transformer else 0.0,
                    use_layer_norm=config.use_layer_norm,
                    is_relu=config.use_relu,
                )
                one_channel.add_module("resconv-{}".format(idx), tmp)
            self.conv.add_module("channel-{}".format(filter_size), one_channel)

        if config.use_transformer:
            self.transfer = EncoderHidden(
                config.max_length,
                config.transfer_layer,
                self.filter_num * config.num_filter_maps,
                config.transfer_attention_head,
                config.transfer_fsize,
                config.dropout,
            )

        # Label attention part of OutputLayer
        self.U = nn.Linear(self.filter_num * config.num_filter_maps, self.Y)
        xavier_uniform(self.U.weight)
        self.final = nn.Linear(self.filter_num * config.num_filter_maps, self.Y)
        xavier_uniform(self.final.weight)

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
        if self.use_transformer:
            x = self.transfer(x)

        # Label attention part of OutputLayer
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        m = alpha.matmul(x)
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        return y

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False
