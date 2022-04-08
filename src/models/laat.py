# -*- coding: utf-8 -*-
"""
    Word-based RNN model for text classification
    @author: Thanh Vu <thanh.vu@csiro.au>
    @date created: 07/03/2019
    @date last modified: 19/08/2020
"""

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.utils.laat_attention_utils import *
from src.utils.laat_attention_utils import *
from src.utils.mapper import ConfigMapper
from src.utils.text_loggers import get_logger
from src.utils.caml_utils import load_lookups, pad_desc_vecs


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config

        self.Y = config.num_classes
        self.embed_drop = nn.Dropout(p=config.dropout)

        self.dicts = load_lookups(
            dataset_dir=config.dataset_dir,
            mimic_dir=config.mimic_dir,
            static_dir=config.static_dir,
            word2vec_dir=config.word2vec_dir,
            version=config.version,
        )

        # make embedding layer
        embedding_cls = ConfigMapper.get_object("embeddings", "word2vec")
        W = torch.Tensor(embedding_cls.load_emb_matrix(config.word2vec_dir))
        self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
        self.embed.weight.data = W.clone()
        self.embed.output_size = W.size()[1]

    def embed_descriptions(self, desc_data):
        # label description embedding via convolutional layer
        # number of labels is inconsistent across instances, so have to iterate
        # over the batch

        # Whether the model is using GPU
        gpu = next(self.parameters()).is_cuda

        b_batch = []
        for inst in desc_data:
            if len(inst) > 0:
                if gpu:
                    lt = Variable(torch.cuda.LongTensor(inst))
                else:
                    lt = Variable(torch.LongTensor(inst))
                d = self.desc_embedding(lt)
                d = d.transpose(1, 2)
                d = self.label_conv(d)
                d = F.max_pool1d(F.tanh(d), kernel_size=d.size()[2])
                d = d.squeeze(2)
                b_inst = self.label_fc1(d)
                b_batch.append(b_inst)
            else:
                b_batch.append([])
        return b_batch

    def _compare_label_embeddings(self, target, b_batch, desc_data):
        # description regularization loss
        # b is the embedding from description conv
        # iterate over batch because each instance has different # labels
        diffs = []
        for i, bi in enumerate(b_batch):
            ti = target[i]
            inds = torch.nonzero(ti.data).squeeze().cpu().numpy()

            zi = self.final.weight[inds, :]
            diff = (zi - bi).mul(zi - bi).mean()

            # multiply by number of labels to make sure overall mean is balanced
            # with regard to number of labels
            diffs.append(self.config.lmbda * diff * bi.size()[0])
        return diffs

@ConfigMapper.map("models", "LAAT")
class RNN(BaseModel):
    def __init__(self, args):
        """

        :param vocab: Vocab
            The vocabulary normally built on the training data
        :param args:
            mode: rand/static/non-static/multichannel the mode of initialising embeddings
            hidden_size: (int) The size of the hidden layer
            n_layers: (int) The number of hidden layers
            bidirectional: (bool) Whether or not using bidirectional connection
            dropout: (float) The dropout parameter for RNN (GRU or LSTM)
        """

        super(RNN, self).__init__(config=args)
        self.args = args
        self.n_layers = args.n_layers
        self.hidden_size = args.hidden_size
        self.bidirectional = bool(args.bidirectional)
        self.n_directions = int(self.bidirectional) + 1
        self.attention_mode = args.attention_mode
        self.output_size = self.hidden_size * self.n_directions
        self.rnn_model = args.rnn_model

        self.dropout = args.dropout

        if self.rnn_model.lower() == "gru":
            self.rnn = nn.GRU(self.args.embed_size, self.hidden_size, num_layers=self.n_layers,
                              bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)
        else:
            self.rnn = nn.LSTM(self.args.embed_size, self.hidden_size, num_layers=self.n_layers,
                               bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)

        self.use_dropout = args.dropout > 0
        self.dropout = nn.Dropout(args.dropout)
        init_attention_layer(self)

    def init_hidden(self,
                    batch_size: int = 1) -> Variable:
        """
        Initialise the hidden layer
        :param batch_size: int
            The batch size
        :return: Variable
            The initialised hidden layer
        """
        # [(n_layers x n_directions) x batch_size x hidden_size]
        h = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(device)
        c = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(device)
        if self.rnn_model.lower() == "gru":
            return h
        return h, c

    def forward(self,
                batch_data: torch.LongTensor) -> tuple:
        """

        :param batch_data: torch.LongTensor
            [batch_size x max_seq_len]
        :param lengths: torch.LongTensor
            [batch_size x 1]
        :return: output [batch_size x n_classes]
            attention_weights
        """

        batch_size = batch_data.size()[0]
        hidden = self.init_hidden(batch_size)

        embeds = self.embed(batch_data)

        if self.use_dropout:
            embeds = self.dropout(embeds)

        self.rnn.flatten_parameters()
        embeds = pack_padded_sequence(embeds, lengths, batch_first=True)

        rnn_output, hidden = self.rnn(embeds, hidden)
        if self.rnn_model.lower() == "lstm":
            hidden = hidden[0]

        rnn_output = pad_packed_sequence(rnn_output)[0]

        rnn_output = rnn_output.permute(1, 0, 2)

        weighted_outputs, _ = perform_attention(self, rnn_output,
                                                                self.get_last_hidden_output(hidden)
                                                                )
        return weighted_outputs

    def get_last_hidden_output(self, hidden):
        if self.bidirectional:
            hidden_forward = hidden[-1]
            hidden_backward = hidden[0]
            if len(hidden_backward.shape) > 2:
                hidden_forward = hidden_forward.squeeze(0)
                hidden_backward = hidden_backward.squeeze(0)
            last_rnn_output = torch.cat((hidden_forward, hidden_backward), 1)
        else:

            last_rnn_output = hidden[-1]
            if len(hidden.shape) > 2:
                last_rnn_output = last_rnn_output.squeeze(0)

        return last_rnn_output
