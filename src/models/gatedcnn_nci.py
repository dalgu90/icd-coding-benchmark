import torch
import torch.nn.functional as F
from torch import nn


class GatedCNNencoder(nn.Module):
    def __init__(self, args, Y, dicts):
        super(GatedCNNencoder, self).__init__()
        self.args = args
        self.max_length = args.MAX_LENGTH
        self.dropout = args.dropout
        self.ninp = args.embed_size
        self.nhid = args.nhid
        self.nout = args.nout
        self.bidirectional = args.bidirectional

        self.word_rep = WordRep(args, Y, dicts)
        self.encoder = GatedCNN(args, Y, dicts, self.ninp, self.nout)
        self.network = nn.ModuleList([self.encoder])
        if self.bidirectional:
            self.output_layer = OutputLayer(args, Y, dicts, self.nout * 2)
        else:
            self.output_layer = OutputLayer(args, Y, dicts, self.nout)
        self.var_drop = VariationalDropout()

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False

    def _reverse_seq(self, X, mask, seq_max_len):
        """
        X -> batch, seq_len, dim
        mask -> batch, seq_len
        """
        mask_sum = torch.sum(mask, 1).int()
        xfs = []
        for x, c in zip(X, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)
        padded_rev = torch.zeros((len(xfs), X.size(1), X.size(2))).cuda()
        for i, mat in enumerate(xfs):
            padded_rev[i][: len(mat), :] = mat
        return padded_rev

    def forward(self, data, target, mask, hidden, desc):
        """
        :param data: The input sequence, with dimesion (N, L)
        :param target: labels
        :param mask: input sequence mask
        :param hidden: The initial hidden state (h, c)
        :param desc: Whether to use code description
        :return: logits, loss, hidden
        """
        emb = self.word_rep(data, target)
        if self.bidirectional:
            emb_reverse = self._reverse_seq(emb, mask, self.max_length)
        emb = emb.transpose(1, 2)  # emb: [bs, 100, len]
        if self.bidirectional:
            emb_reverse = emb_reverse.transpose(
                1, 2
            )  # emb_reverse: [bs, 100, len]
        cnn_encoder = self.network[0]
        raw_output, hidden = cnn_encoder(emb, hidden)
        if self.bidirectional:
            raw_out_re, hidden = cnn_encoder(emb_reverse, hidden)
        output = self.var_drop(raw_output, self.dropout)
        if self.bidirectional:
            output_re = self._reverse_seq(raw_out_re, mask, self.max_length)
            output_re = self.var_drop(output_re, self.dropout)
        if self.bidirectional:
            output = torch.cat([output, output_re], dim=2)
        if self.args.desc:
            logits, loss, _, interaction = self.output_layer(
                output, target, desc
            )
        else:
            logits, loss, _, interaction = self.output_layer(
                output, target, None
            )
        return logits, loss, hidden, interaction

    def init_hidden(self, bsz):
        h_size = self.nhid + self.nout
        weight = next(self.parameters()).data
        return (
            weight.new(bsz, h_size, 1).zero_(),
            weight.new(bsz, h_size, 1).zero_(),
        )


from typing import Tuple

import torch.nn as nn
from embeddings import build_pretrain_embedding, load_embeddings
from torch import Tensor
from torch.nn.init import kaiming_uniform_, normal_, xavier_uniform_


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

    def __init__(self, embed_dir, dropout, num_filter_maps):
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

        self.conv_dict = {
            1: [self.embedding_size, num_filter_maps],
            2: [self.embedding_size, 100, num_filter_maps],
            3: [self.embedding_size, 150, 100, num_filter_maps],
            4: [self.embedding_size, 200, 150, 100, num_filter_maps],
        }

    def forward(self, x):
        embedding = self.embed(x)
        x = self.dropout(embedding)
        return x


class VariationalHidDropout(nn.Module):
    def __init__(self, dropout=0.0):
        """
        Hidden-to-hidden (VD-based) dropout that applies the same mask at every
        time step and every layer of TrellisNet.

        Args:
            dropout (float): The dropout probability.
        """
        super(VariationalHidDropout, self).__init__()
        self.dropout_probability = dropout
        self.mask = None

    def reset_mask(self, input):

        # Dimension (N, C, L)
        m = input.data.new(input.size(0), input.size(1), 1).bernoulli_(
            1 - self.dropout_probability
        )
        with torch.no_grad():
            mask = m / (1 - self.dropout_probability)
            self.mask = mask
        return mask

    def forward(self, input):
        # We don't apply dropout if the model is in eval mode.
        if not self.training or self.dropout_probability == 0:
            return input

        assert (
            self.mask is not None
        ), "You need to reset mask before using VariationalHidDropout"
        mask = self.mask.expand_as(input)  # Make sure the dimension matches
        return mask * input


class WeightShareConv1d(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        out_channels,
        kernel_size,
        dropout=0.0,
        init_mean=0.0,
        init_std=0.01,
    ):
        """
        The weight-tied 1D convolution used in TrellisNet.

        Args:
            input_dim (int): The dimension of the input. This is equivalent to
                             the number of input channels in the first
                             convolutional layer.
            hidden_dim (int): The dimension of the hidden state. This is
                              equivalent to the number of input channels in the
                              second convolutional layer.
            out_channels (int): The number of output channels in both
                                convolutional layers.
            kernel_size (int): The size of the filter used in both
                               convolutional layers.
            dropout (float): Dropout probability for the hidden-to-hidden
                             dropout layer.
            init_mean (float): The mean of the normal distribution with which
                               weights of the convolutional layers are
                               initialised.
            init_std (float): The standard deviation of the normal distribution
                              with which weights of the convolutional layers are
                              initialised.
        """
        super(WeightShareConv1d, self).__init__()

        self.input_dim = input_dim
        self.kernel_size = kernel_size

        self._dict = {}

        conv_layer_1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )
        self.weight_1 = conv_layer_1.weight

        conv_layer_2 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )
        self.weight_2 = conv2.weight
        self.bias_2 = conv2.bias

        self.init_conv_weights(init_mean, init_std)

        self.dropout = VariationalHidDropout(dropout=dropout)

    def init_conv_weights(self, init_mean, init_std):
        self.weight_1.data.normal_(mean=init_mean, std=init_std)
        self.weight_2.data.normal_(mean=init_mean, std=init_std)
        self.bias_2.data.normal_(mean=init_mean, std=init_std)

    def forward(self, input, dilation, hid):
        batch_size = input.size(0)

        padding = (self.kernel_size - 1) * dilation  # Padding size.
        x = F.pad(input=input, pad=(padding, 0))  # Pad with zeros.

        x_1 = x[:, : self.input_dim]
        z_1 = x[:, self.input_dim :]
        z_1[:, :, :padding] = hid[:batch_size, :, :].repeat(1, 1, padding)

        device = x_1.get_device()

        if (dilation, device) not in self.dict or self.dict[
            (dilation, device)
        ] is None:
            self.dict[(dilation, device)] = F.conv1d(
                input=x_1, weight=self.weight1, dilation=dilation
            )

        z_1 = self.dropout(z_1)
        injected = self.dict[(dilation, device)] + F.conv1d(
            input=z_1, weight=self.weight2, bias=self.bias2, dilation=dilation
        )
        return injected


class GatedCNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        kernel_size,
        dropout,
        init_mean,
        init_std,
        levels,
    ):
        """
        Gated CNN module.

        Args:
            input_dim (int): The dimension of the input.
            hidden_dim (int): The hidden dimension. The hidden dimension for the
                              weight-shared Conv1D layer is
                              `hidden_dim + output_dim`.
            output_dim (int): The output dimension. The number of output
                              channels of the weight-shared Conv1D layer is
                              `4 * (hidden_dim + output_dim)`.
            kernel_size (int): The size of the filter used in
                               `WeightSharedConv1D`.
            dropout (float): Dropout probability for the `WeightSharedConv1D`.
            init_mean (float): The mean of the normal distribution with which
                               weights of the `WeightSharedConv1D` layer are
                               initialised.
            init_std (float): The standard deviation of the normal distribution
                              with which weights of the `WeightSharedConv1D`
                              layer are initialised.
        """
        super(GatedCNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = output_dim
        self.levels = levels

        self.hidden_dim_for_conv = hidden_dim + output_dim

        self.dilations = [i + 1 for i in range(levels)]

        self.full_conv = WeightShareConv1d(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim_for_conv,
            output_channels=4 * self.hidden_dim_for_conv,
            kernel_size=kernel_size,
            dropout=dropout,
            init_mean=init_mean,
            init_std=init_std,
        )

        self.ht = None

    def transform_input(self, X):
        device = X.get_device()
        if device == -1:
            device = "cpu"

        batch_size = X.size(0)
        seq_len = X.size(2)

        ht = torch.zeros(batch_size, self.hidden_dim_for_conv, seq_len).to(
            device
        )
        self.ct = torch.zeros(batch_size, self.hidden_dim_for_conv, seq_len).to(
            device
        )
        return torch.cat((X, ht), dim=1)

    def gating(self, Z, dilation=1, hc=None):
        batch_size = Z.size(0)
        (hid, cell) = hc

        out = self.full_conv(input=Z, dilation=dilation, hid=hid)

        ct_1 = F.pad(self.ct, (dilation, 0))[:, :, :-dilation]
        ct_1[:, :, :dilation] = cell[:batch_size].repeat(1, 1, dilation)

        it = torch.sigmoid(out[:, : self.hidden_dim_for_conv])
        ot = torch.sigmoid(
            out[:, self.hidden_dim_for_conv : 2 * self.hidden_dim_for_conv]
        )
        gt = torch.tanh(
            out[:, 2 * self.hidden_dim_for_conv : 3 * self.hidden_dim_for_conv]
        )
        ft = torch.sigmoid(
            out[:, 3 * self.hidden_dim_for_conv : 4 * self.hidden_dim_for_conv]
        )
        self.ct = ft * ct_1 + it * gt
        ht = ot * torch.tanh(self.ct)

        Z = torch.cat((Z[:, : self.input_dim], ht), dim=1)
        return Z

    def forward(self, emb, hc):
        Z = self.transform_input(emb)
        for key in self.full_conv.dict:
            if key[1] == emb.get_device():
                self.full_conv.dict[key] = None
        self.full_conv.drop.reset_mask(Z[:, self.input_dim :])

        for dilation_per_level in self.dilations:
            Z = self.gating(Z, dilation=dilation_per_level, hc=hc)

        out = Z[:, -self.output_dim :].transpose(1, 2)
        hc = (Z[:, self.input_dim :, -1:], self.ct[:, :, -1:])
        return out, hc


class OutputLayer(nn.Module):
    def __init__(
        self, input_size, num_labels, embed_dir, dropout, num_filter_maps
    ):
        super(OutputLayer, self).__init__()

        self.word_embedding_layer = WordEmbeddingLayer(
            embed_dir, dropout, num_filter_maps
        )

        self.U = nn.Linear(input_size, num_labels)
        self.final = nn.Linear(input_size, num_labels)
        self.proj_layer = nn.Linear(input_size, 1, bias=False)

        xavier_uniform_(self.U.weight)
        xavier_uniform_(self.final.weight)

    def forward(self, x, desc):
        if desc is not None:
            desc_vec = self.word_rep(desc, target)
            desc_vec = torch.mean(desc_vec, dim=1).unsqueeze(0)
            mmt = desc_vec.matmul(x.transpose(1, 2))
        else:
            mmt = self.U.weight.matmul(x.transpose(1, 2))

        m = mmt.matmul(x)

        y = self.final.weight.mul(m)
        logits = self.proj_layer(y).squeeze(-1).add(self.final.bias)

        return logits


class VariationalDropout(nn.Module):
    def __init__(self):
        """
        Feed-forward version of variational dropout that applies the same mask
        at every time step.
        """
        super(VariationalDropout, self).__init__()

    def forward(self, x, dropout=0.5, dim=3):
        if not self.training or not dropout:
            return x
        if dim == 4:
            # Dimension (M, N, L, C), where C stands for channels
            m = x.data.new(x.size(0), x.size(1), 1, x.size(3)).bernoulli_(
                1 - dropout
            )
        else:
            # Dimension (N, L, C)
            m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        with torch.no_grad():
            mask = m / (1 - dropout)
            mask = mask.expand_as(x)
        return mask * x
