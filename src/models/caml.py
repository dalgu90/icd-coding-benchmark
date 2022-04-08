"""
    CAML model (Mullenbach et al. 2018)
    https://github.com/jamesmullenbach/caml-mimic
"""

from math import floor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_uniform

from src.utils.caml_utils import load_lookups, pad_desc_vecs
from src.utils.mapper import ConfigMapper
from src.utils.text_loggers import get_logger

logger = get_logger(__name__)


# From learn/models.py
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


@ConfigMapper.map("models", "CAML")
class ConvAttnPool(BaseModel):
    def __init__(self, config):
        cls_name = self.__class__.__name__
        logger.info(f"Initializing {cls_name}")
        logger.debug(f"Initializing {cls_name} with config: {config}")
        super(ConvAttnPool, self).__init__(config=config)

        self.pad_idx = self.dicts["w2ind"][config.pad_token]
        self.unk_idx = self.dicts["w2ind"][config.unk_token]

        # initialize conv layer as in 2.1
        self.conv = nn.Conv1d(
            config.embed_size,
            config.num_filter_maps,
            kernel_size=config.kernel_size,
            padding=int(floor(config.kernel_size / 2)),
        )
        xavier_uniform(self.conv.weight)

        # context vectors for computing attention as in 2.2
        self.U = nn.Linear(config.num_filter_maps, self.Y)
        xavier_uniform(self.U.weight)

        # final layer: create a matrix to use for the L binary classifiers as in
        # 2.3
        self.final = nn.Linear(config.num_filter_maps, self.Y)
        xavier_uniform(self.final.weight)

        # initialize with trained code embeddings if applicable
        if config.init_code_emb:
            if config.embed_size != config.num_filter_maps:
                logger.warn(
                    "Cannot init attention vectors since the dimension differ"
                    "from the dimension of the embedding"
                )
            else:
                self._code_emb_init()

                # also set conv weights to do sum of inputs
                weights = (
                    torch.eye(config.embed_size)
                    .unsqueeze(2)
                    .expand(-1, -1, config.kernel_size)
                    / config.kernel_size
                )
                self.conv.weight.data = weights.clone()
                self.conv.bias.data.zero_()

        # conv for label descriptions as in 2.5
        # description module has its own embedding and convolution layers
        if config.lmbda > 0:
            W = self.embed.weight.data
            self.desc_embedding = nn.Embedding(
                W.size()[0], W.size()[1], padding_idx=0
            )
            self.desc_embedding.weight.data = W.clone()

            self.label_conv = nn.Conv1d(
                config.embed_size,
                config.num_filter_maps,
                kernel_size=config.kernel_size,
                padding=int(floor(config.kernel_size / 2)),
            )
            xavier_uniform(self.label_conv.weight)

            self.label_fc1 = nn.Linear(
                config.num_filter_maps, config.num_filter_maps
            )
            xavier_uniform(self.label_fc1.weight)

            # Pre-process the code description into word idxs
            self.dv_dict = {}
            ind2c = self.dicts["ind2c"]
            w2ind = self.dicts["w2ind"]
            desc_dict = self.dicts["desc"]
            for i, c in ind2c.items():
                desc_vec = [
                    w2ind[w] if w in w2ind else self.unk_idx
                    for w in desc_dict[c]
                ]
                self.dv_dict[i] = desc_vec

    def _code_emb_init(self):
        # In the original CAML repo, this method seems not being called.
        # In this implementation, we compute the AVERAGE word2vec embeddings for
        # each code and initialize the self.U and self.final with it.
        ind2c = self.dicts["ind2c"]
        w2ind = self.dicts["w2ind"]
        desc_dict = self.dicts["desc"]

        weights = torch.zeros_like(self.final.weight)
        for i, c in ind2c.items():
            desc_vec = [
                w2ind[w] if w in w2ind else self.unk_idx
                for w in desc_dict[c].split()
            ]
            weights[i] = self.embed(torch.tensor(desc_vec)).mean(axis=0)
        self.U.weight.data = torch.Tensor(weights).clone()
        self.final.weight.data = torch.Tensor(weights).clone()

    def forward(self, x):
        # get embeddings and apply dropout
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        # apply convolution and nonlinearity (tanh)
        x = F.tanh(self.conv(x).transpose(1, 2))
        # apply attention
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        # document representations are weighted sums using the attention. Can
        # compute all at once as a matmul
        m = alpha.matmul(x)
        # final layer classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        return y
        

    def regularizer(self, labels=None):
        if not self.config.lmbda:
            return 0.0

        # Retrive the description tokens of the labels
        desc_vecs = []
        for label in labels:
            desc_vecs.append(
                [self.dv_dict[i] for i, l in enumerate(label) if l]
            )
        desc_data = [np.array(pad_desc_vecs(dvs)) for dvs in desc_vecs]

        # run descriptions through description module
        b_batch = self.embed_descriptions(desc_data)
        # get l2 similarity loss
        diffs = self._compare_label_embeddings(labels, b_batch, desc_data)
        diff = torch.stack(diffs).mean()

        return diff


@ConfigMapper.map("models", "CNN")
class VanillaConv(BaseModel):
    def __init__(self, config):
        cls_name = self.__class__.__name__
        logger.info(f"Initializing {cls_name}")
        logger.debug(f"Initializing {cls_name} with config: {config}")
        super(VanillaConv, self).__init__(config)

        # initialize conv layer as in 2.1
        self.conv = nn.Conv1d(
            config.embed_size,
            config.num_filter_maps,
            kernel_size=config.kernel_size,
        )
        xavier_uniform(self.conv.weight)

        # linear output
        self.fc = nn.Linear(config.num_filter_maps, self.Y)
        xavier_uniform(self.fc.weight)

    def forward(self, x):
        # embed
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        # conv/max-pooling
        c = self.conv(x)
        x = F.max_pool1d(F.tanh(c), kernel_size=c.size()[2])
        x = x.squeeze(dim=2)

        # linear output
        x = self.fc(x)

        return x

    def construct_attention(self, argmax, num_windows):
        attn_batches = []
        for argmax_i in argmax:
            attns = []
            for i in range(num_windows):
                # generate mask to select indices of conv features where max was
                # i
                mask = (argmax_i == i).repeat(1, self.Y).t()
                # apply mask to every label's weight vector and take the sum to
                # get the 'attention' score
                weights = self.fc.weight[mask].view(-1, self.Y)
                if len(weights.size()) > 0:
                    window_attns = weights.sum(dim=0)
                    attns.append(window_attns)
                else:
                    # this window was never a max
                    attns.append(Variable(torch.zeros(self.Y)).cuda())
            # combine
            attn = torch.stack(attns)
            attn_batches.append(attn)
        attn_full = torch.stack(attn_batches)
        # put it in the right form for passing to interpret
        attn_full = attn_full.transpose(1, 2)
        return attn_full
