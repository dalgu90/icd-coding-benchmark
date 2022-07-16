import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from src.utils.mapper import ConfigMapper
from src.utils.text_loggers import get_logger

logger = get_logger(__name__)


@ConfigMapper.map("models", "transicd")
class TransICD(nn.Module):
    def __init__(self, config):
        super(TransICD, self).__init__()
        logger.info(f"Initialising {self.__class__.__name__}")
        logger.debug(
            f"Initialising {self.__class__.__name__} with " f"config: {config}"
        )

        if config.use_gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.word_embedding_layer = WordEmbeddingLayer(
            embed_dir=config.embed_dir
        )
        self.embed_size = self.word_embedding_layer.embedding_size
        self.dropout = nn.Dropout(config.dropout)
        self.positional_embedding_layer = PositionalEmbeddingLayer(
            d_model=self.embed_size,
            dropout=config.dropout,
            max_len=config.max_len,
            device=self.device,
        )

        if self.embed_size % config.num_heads != 0:
            raise ValueError(
                f"Embedding size {self.embed_size} needs to be divisible by "
                f"the number of heads {config.num_heads}"
            )

        if config.freeze_embedding_layer:
            self.freeze_layer(self.word_embedding_layer)

        self.pad_idx = config.pad_idx
        self.num_classes = config.num_classes

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_size,
            nhead=config.num_heads,
            dim_feedforward=config.transformer_ff_up_scale_factor_for_hidden_dim
            * self.embed_size,
            dropout=config.dropout,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=config.num_layers
        )
        self.label_attention_layer = LabelAttentionLayer(
            embed_size=self.embed_size,
            num_classes=config.num_classes,
            attn_expansion=config.attn_expansion,
            epsilon=config.epsilon,
        )

        # The official code (and paper) has a separate linear layer for every
        # code. This is different from the convention; generally, a shared
        # linear layer is used.
        self.ff_layers = nn.ModuleList(
            [
                nn.Linear(self.embed_size, 1)
                for code in range(config.num_classes)
            ]
        )

    def freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        # `inputs` shape: (batch_size, seq_len)
        batch_size = inputs.shape[0]
        # `attn_mask` shape: (batch_size, seq_len, 1)
        attn_mask = (inputs != self.pad_idx).unsqueeze(2)

        # `src_key_padding_mask` shape: (batch_size, seq_len)
        src_key_padding_mask = inputs == self.pad_idx

        # Look up the embeddings of the tokens present in the input.
        # `embeddings` shape: (batch_size, seq_len, embed_size)
        embeddings = self.word_embedding_layer(inputs)

        # The authors do some sort of scaling here - they multiply the
        # embeddings with the square root of `embed_size`.
        embeddings = embeddings * math.sqrt(self.embed_size)

        # Add the positional embedding to the word embedding.
        embeddings = self.positional_embedding_layer(embeddings)

        embeddings = self.dropout(embeddings)

        # `embeddings` shape: (seq_len, batch_size, embed_size)
        embeddings = embeddings.permute(1, 0, 2)

        # Pass the embedded input through the Transformer model.
        # `encoded_inputs` shape: (batch_size, seq_len, embed_size)
        encoded_inputs = self.encoder(
            embeddings, src_key_padding_mask=src_key_padding_mask
        )
        encoded_inputs = encoded_inputs.permute(1, 0, 2)

        # `weighted_outputs` shape: (batch_size, num_classes, embed_size)
        weighted_outputs, _ = self.label_attention_layer(
            encoded_inputs, attn_mask
        )

        outputs = torch.zeros(batch_size, self.num_classes).to(self.device)
        for code, ff_layer in enumerate(self.ff_layers):
            outputs[:, code : code + 1] = ff_layer(weighted_outputs[:, code, :])

        return outputs


class WordEmbeddingLayer(nn.Module):
    """
    A Word Embedding Layer. This layer loads a pre-trained word embedding matrix
    , and copies its weights to an nn.Embedding layer.

    Args:
        embed_dir (str): A directory containing the pre-trained word embedding
                         matrix, among other things. Please see
                         https://github.com/dalgu90/icd-coding-benchmark/blob/main/src/modules/embeddings.py#L17
                         for more details.
    """

    def __init__(self, embed_dir):
        super(WordEmbeddingLayer, self).__init__()
        logger.debug(
            f"Initialising {self.__class__.__name__} with "
            f"embed_dir = {embed_dir}"
        )

        # Note: This should be changed, since we won't always use Word2Vec.
        embedding_cls = ConfigMapper.get_object("embeddings", "word2vec")

        W = torch.Tensor(embedding_cls.load_emb_matrix(embed_dir))
        self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
        self.embed.weight.data = W.clone()

        self.embedding_size = self.embed.embedding_dim

    def forward(self, x):
        embedding = self.embed(x)
        return embedding


class PositionalEmbeddingLayer(nn.Module):
    """
    A layer for implementing position embeddings for transformers. This layer
    is inspired from https://nlp.seas.harvard.edu/2018/04/03/attention.html. For
    an intuitive explanation of how this layer works, please see
    https://kazemnejad.com/blog/transformer_architecture_positional_encoding/.

    Args:
        d_model (int): Input embedding size. Defaults to 512.
        dropout (float): Dropout probability.
        max_len (int): Maximum length of the input sequence.
    """

    def __init__(self, d_model=128, dropout=0.1, max_len=2500, device="cuda"):
        super(PositionalEmbeddingLayer, self).__init__()
        logger.debug(
            f"Initialising {self.__class__.__name__} with "
            f"d_model = {d_model}, dropout = {dropout}, max_len = {max_len}, "
            f"device = {device}"
        )

        self.device = device

        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        positional_emb = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        positional_emb[:, 0::2] = torch.sin(position * div_term)
        positional_emb[:, 1::2] = torch.cos(position * div_term)

        # Revisit this step. In the official TransICD code, they take the
        # transpose: positional_emb.unsqueeze(0).transpose(0, 1). Not sure if
        # that makes sense. The shape changes from (bsz, max_len, d_model) to
        # (max_len, bsz, d_model).
        positional_emb = positional_emb.unsqueeze(0)
        self.register_buffer("positional_emb", positional_emb)

    def forward(self, x):
        x = x + self.positional_emb[:, : x.size(1)]

        return self.dropout(x)


class LabelAttentionLayer(nn.Module):
    """
    This layer implements the label attention mechanism, i.e., it computes label
    attention for every ICD code.

    Note: The official TransICD code has two layers for attention:
    https://github.com/biplob1ly/TransICD/blob/main/code/models.py#L8,
    https://github.com/biplob1ly/TransICD/blob/main/code/models.py#L36.
    They use the first one in the TransICD model, whereas the second one isn't
    used anywhere in the code.

    Args:
        embed_size (int): Input embedding size. Defaults to 128.
        num_classes (int): Number of ICD codes, i.e., output size of the output
                          linear layer. Defaults to 50.
        attn_expansion (int): Factor for scaling up the input embeddings.
                              Defaults to 2.
        epsilon (float): Small float value for filling the attention mask.
                         Defaults to 1e-9.
    """

    def __init__(
        self, embed_size=128, num_classes=50, attn_expansion=2, epsilon=1e-9
    ):
        super(LabelAttentionLayer, self).__init__()
        logger.debug(
            f"Initialising {self.__class__.__name__} with "
            f"embed_size = {embed_size}, num_classes = {num_classes}, "
            f"attn_expansion = {attn_expansion}, epsilon = {epsilon}"
        )

        self.epsilon = epsilon

        self.linear_layer_1 = nn.Linear(
            in_features=embed_size, out_features=embed_size * attn_expansion
        )
        self.tanh_activation = nn.Tanh()

        self.linear_layer_2 = nn.Linear(
            in_features=embed_size * attn_expansion, out_features=num_classes
        )

        self.softmax_activation = nn.Softmax(dim=1)

    def forward(self, hidden, attn_mask=None):
        # `hidden` shape: (batch_size, seq_len, embed_size)
        # `output_1` shape: (batch_size, seq_len, (attn_expansion x embed_size))
        output_1 = self.linear_layer_1(hidden)
        output_1 = self.tanh_activation(output_1)

        # `output_2` shape: (batch_size, seq_len, num_classes)
        output_2 = self.linear_layer_2(output_1)

        # Masked fill to avoid softmaxing over padded words. The authors fill
        # the value with -1e9, which is probably incorrect. It should be 1e-9.
        if attn_mask is not None:
            output_2 = output_2.masked_fill_(
                mask=attn_mask == 0, value=self.epsilon
            )

        # `attn_weights` shape: (batch_size, num_classes, seq_len)
        attn_weights = self.softmax_activation(output_2).transpose(1, 2)

        # `weighted_outputs` shape: (batch_size, num_classes, embed_size)
        weighted_output = attn_weights @ hidden
        return weighted_output, attn_weights
