import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_, xavier_uniform_

from src.utils.caml_utils import load_lookups, pad_desc_vecs
from src.utils.mapper import ConfigMapper
from src.utils.text_loggers import get_logger

logger = get_logger(__name__)


@ConfigMapper.map("models", "gatedcnn_nci")
class GatedCNNNCI(nn.Module):
    def __init__(self, config):
        super(GatedCNNNCI, self).__init__()
        self.max_length = config.max_length
        self.dropout = config.dropout
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.bidirectional = config.bidirectional
        self.use_description = config.use_description

        self.word_embedding_layer = WordEmbeddingLayer(
            embed_dir=config.embed_dir,
            dataset_dir=config.dataset_dir,
            mimic_dir=config.mimic_dir,
            static_dir=config.static_dir,
            version=config.version,
            dropout=config.dropout,
            pad_token=config.pad_token,
            unk_token=config.unk_token,
        )
        self.desc_vecs = self.word_embedding_layer.desc_vecs

        self.encoder = GatedCNN(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            init_mean=config.init_mean,
            init_std=config.init_std,
            levels=config.levels,
        )

        if self.bidirectional:
            self.output_layer = OutputLayer(
                embed_dir=config.embed_dir,
                dataset_dir=config.dataset_dir,
                mimic_dir=config.mimic_dir,
                static_dir=config.static_dir,
                version=config.version,
                input_dim=2 * config.input_dim,
                num_labels=config.output_dim,
                dropout=config.dropout,
                pad_token=config.pad_token,
                unk_token=config.unk_token,
            )
        else:
            self.output_layer = OutputLayer(
                embed_dir=config.embed_dir,
                dataset_dir=config.dataset_dir,
                mimic_dir=config.mimic_dir,
                static_dir=config.static_dir,
                version=config.version,
                input_dim=config.input_dim,
                num_labels=config.num_labels,
                dropout=config.dropout,
                pad_token=config.pad_token,
                unk_token=config.unk_token,
            )

        self.variational_dropout = VariationalDropout(dropout=config.dropout)

        self.hidden = None

    def freeze_net(self):
        for p in self.word_embedding_layer.embed.parameters():
            p.requires_grad = False

    def init_hidden(self, batch_size):
        h_size = self.hidden_dim + self.output_dim
        weight = next(self.parameters()).data
        return (
            weight.new(batch_size, h_size, 1).zero_(),
            weight.new(batch_size, h_size, 1).zero_(),
        )

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

    def forward(self, data):
        """
        :param data: The input sequence, with dimesion (N, L)
        :param desc: Whether to use code description
        :return: logits, loss, hidden
        """
        device = data.get_device()
        if device == -1:
            device = "cpu"

        # If this is the first forward pass, we will initialise the hidden
        # state.
        if self.hidden is None:
            self.init_hidden_flag = True
            self.hidden = self.init_hidden(data.size(0))

        # Look up the embeddings of all the tokens using the WordEmbeddingLayer.
        # `emb` shape: (batch_size, max_length, embed_size)
        emb, mask = self.word_embedding_layer(data)

        # If we want a bidirectional model, we reverse the sequence of
        # tokens.
        if self.bidirectional:
            # `emb_reverse` shape: (batch_size, max_length, embed_size)
            emb_reverse = self._reverse_seq(emb, mask, self.max_length)
            # `emb_reverse` shape`: [batch_size, embed_size, max_length]
            emb_reverse = emb_reverse.transpose(1, 2)
        # `emb` shape: (batch_size, embed_size, max_length)
        emb = emb.transpose(1, 2)

        # Pass the embeddings through the encoder. If the model is
        # bidirectional, we pass the reverse embeddings as well.
        raw_output, self.hidden = self.encoder(emb, self.hidden)
        if self.bidirectional:
            raw_out_reverse, self.hidden = self.encoder(
                emb_reverse, self.hidden
            )

        output = self.variational_dropout(raw_output)
        if self.bidirectional:
            output_reverse = self._reverse_seq(
                raw_out_reverse, mask, self.max_length
            )
            output_reverse = self.variational_dropout(output_reverse)
            output = torch.cat([output, output_reverse], dim=2)

        if self.use_description:
            logits = self.output_layer(output, self.desc_vecs.to(device))
        else:
            logits = self.output_layer(output, None)
        return logits


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

    def __init__(
        self,
        embed_dir,
        dataset_dir,
        mimic_dir,
        static_dir,
        version,
        dropout,
        pad_token="<pad>",
        unk_token="<unk>",
        return_pad_mask=True,
        use_description=True,
    ):
        super(WordEmbeddingLayer, self).__init__()
        logger.debug(
            f"Initialising {self.__class__.__name__} with "
            f"embed_dir = {embed_dir}, dropout = {dropout}"
        )

        self.return_pad_mask = return_pad_mask

        # Note: This should be changed, since we won't always use Word2Vec.
        embedding_cls = ConfigMapper.get_object("embeddings", "word2vec")
        vocab = embedding_cls.load_vocab(embed_dir)
        self.pad_token_id = vocab[pad_token]
        self.unk_token_id = vocab[unk_token]

        W = torch.Tensor(embedding_cls.load_emb_matrix(embed_dir))
        self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
        self.embed.weight.data = W.clone()

        self.embedding_size = self.embed.embedding_dim

        self.dropout = nn.Dropout(dropout)

        if use_description:
            dicts = load_lookups(
                dataset_dir=dataset_dir,
                mimic_dir=mimic_dir,
                static_dir=static_dir,
                word2vec_dir=embed_dir,
                version=version,
            )
            ind2c = dicts["ind2c"]
            w2ind = dicts["w2ind"]
            desc_dict = dicts["desc"]
            self.desc_vecs = []
            for i, c in ind2c.items():
                self.desc_vecs.append(
                    [
                        w2ind[w] if w in w2ind else self.unk_token_id
                        for w in desc_dict[c]
                    ]
                )

            # Pad and convert to torch tensor.
            self.desc_vecs = torch.Tensor(
                list(zip(*itertools.zip_longest(*self.desc_vecs, fillvalue=0)))
            ).long()

    def forward(self, x):
        if self.return_pad_mask:
            pad_mask = ~(x == self.pad_token_id)
        embedding = self.embed(x)
        x = self.dropout(embedding)
        if self.return_pad_mask:
            return x, pad_mask
        return x


class VariationalHidDropout(nn.Module):
    """
    Hidden-to-hidden (VD-based) dropout that applies the same mask at every
    time step and every layer of TrellisNet.

    Args:
        dropout (float): The dropout probability.
    """

    def __init__(self, dropout=0.0):
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
        self.weight_2 = conv_layer_2.weight
        self.bias_2 = conv_layer_2.bias

        self.init_conv_weights(init_mean, init_std)

        self.dropout = VariationalHidDropout(dropout=dropout)

        self.dict = {}

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
                input=x_1, weight=self.weight_1, dilation=dilation
            )

        z_1 = self.dropout(z_1)
        injected = self.dict[(dilation, device)] + F.conv1d(
            input=z_1, weight=self.weight_2, bias=self.bias_2, dilation=dilation
        )
        return injected


class GatedCNN(nn.Module):
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
        super(GatedCNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.levels = levels

        self.hidden_dim_for_conv = hidden_dim + output_dim

        self.dilations = [i + 1 for i in range(levels)]

        self.full_conv = WeightShareConv1d(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim_for_conv,
            out_channels=4 * self.hidden_dim_for_conv,
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
        self.full_conv.dropout.reset_mask(Z[:, self.input_dim :])

        for dilation_per_level in self.dilations:
            Z = self.gating(Z, dilation=dilation_per_level, hc=hc)

        out = Z[:, -self.output_dim :].transpose(1, 2)
        hc = (Z[:, self.input_dim :, -1:], self.ct[:, :, -1:])
        return out, hc


class VariationalDropout(nn.Module):
    """
    Feed-forward version of variational dropout that applies the same mask
    at every time step.
    """

    def __init__(self, dropout=0.5, dim=3):
        super(VariationalDropout, self).__init__()
        assert dim in (3, 4), "`dim` should be either 3 or 4"
        self.dropout = dropout
        self.dim = dim

    def forward(self, x):
        if not self.training or not self.dropout:
            return x

        if self.dim == 4:
            # Dimension (M, N, L, C), where C stands for channels
            m = x.data.new(x.size(0), x.size(1), 1, x.size(3)).bernoulli_(
                1 - self.dropout
            )
        else:
            # Dimension (N, L, C)
            m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout)
        with torch.no_grad():
            mask = m / (1 - self.dropout)
            mask = mask.expand_as(x)
        return mask * x


class OutputLayer(nn.Module):
    def __init__(
        self,
        embed_dir,
        dataset_dir,
        mimic_dir,
        static_dir,
        version,
        input_dim,
        num_labels,
        dropout=0.2,
        pad_token="<pad>",
        unk_token="<unk>",
    ):
        super(OutputLayer, self).__init__()

        self.word_embedding_layer = WordEmbeddingLayer(
            embed_dir=embed_dir,
            dataset_dir=dataset_dir,
            mimic_dir=mimic_dir,
            static_dir=static_dir,
            version=version,
            dropout=dropout,
            pad_token=pad_token,
            unk_token=unk_token,
        )

        self.U = nn.Linear(input_dim, num_labels)
        self.final = nn.Linear(input_dim, num_labels)
        self.proj_layer = nn.Linear(input_dim, 1, bias=False)

        xavier_uniform_(self.U.weight)
        xavier_uniform_(self.final.weight)

    def forward(self, x, desc):
        if desc is not None:
            desc_vec, _ = self.word_embedding_layer(desc)
            desc_vec = torch.mean(desc_vec, dim=1).unsqueeze(0)
            mmt = desc_vec.matmul(x.transpose(1, 2))
        else:
            mmt = self.U.weight.matmul(x.transpose(1, 2))

        m = mmt.matmul(x)

        y = self.final.weight.mul(m)
        logits = self.proj_layer(y).squeeze(-1).add(self.final.bias)

        return logits
