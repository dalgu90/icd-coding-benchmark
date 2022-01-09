"""
    CAML model (Mullenbach et al. 2018)
    https://github.com/jamesmullenbach/caml-mimic
"""

import csv
import random
import sys
import time
from collections import defaultdict
from math import floor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors
from torch.autograd import Variable
from torch.nn.init import xavier_uniform

from src.utils.caml_utils import load_embeddings, load_lookups
from src.utils.mapper import ConfigMapper


# From learn/models.py
class BaseModel(nn.Module):
    def __init__(
        self,
        Y,
        dataset_dir,
        embed_file,
        version="mimic3",
        lmbda=0,
        dropout=0.5,
        embed_size=100,
    ):
        super(BaseModel, self).__init__()
        # torch.manual_seed(1337)
        # self.gpu = gpu
        self.Y = Y
        self.embed_size = embed_size
        self.embed_drop = nn.Dropout(p=dropout)
        self.lmbda = lmbda

        self.dicts = load_lookups(Y, dataset_dir, version, desc_embed=lmbda > 0)

        # make embedding layer
        if embed_file:
            print("loading pretrained embeddings...")
            W = torch.Tensor(load_embeddings(embed_file))

            self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.embed.weight.data = W.clone()
        else:
            # add 2 to include UNK and PAD
            vocab_size = len(self.dicts["ind2w"])
            self.embed = nn.Embedding(vocab_size + 2, embed_size, padding_idx=0)

    def _get_loss(self, yhat, target, diffs=None):
        # calculate the BCE
        loss = F.binary_cross_entropy_with_logits(yhat, target)

        # add description regularization loss if relevant
        if self.lmbda > 0 and diffs is not None:
            diff = torch.stack(diffs).mean()
            loss = loss + diff
        return loss

    def embed_descriptions(self, desc_data, gpu):
        # label description embedding via convolutional layer
        # number of labels is inconsistent across instances, so have to iterate
        # over the batch
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
            diffs.append(self.lmbda * diff * bi.size()[0])
        return diffs


@ConfigMapper.map("models", "CAML")
class ConvAttnPool(BaseModel):
    def __init__(
        self,
        num_classes=50,
        dataset_dir=None,
        version="mimic3",
        embed_file=None,
        kernel_size=10,
        num_filter_maps=50,
        lmbda=0.0,
        embed_size=100,
        dropout=0.5,
        code_emb=None,
        **kwargs
    ):
        super(ConvAttnPool, self).__init__(
            Y=num_classes,
            dataset_dir=dataset_dir,
            version=version,
            embed_file=embed_file,
            lmbda=lmbda,
            dropout=dropout,
            embed_size=embed_size,
        )

        # initialize conv layer as in 2.1
        self.conv = nn.Conv1d(
            self.embed_size,
            num_filter_maps,
            kernel_size=kernel_size,
            padding=int(floor(kernel_size / 2)),
        )
        xavier_uniform(self.conv.weight)

        # context vectors for computing attention as in 2.2
        self.U = nn.Linear(num_filter_maps, self.Y)
        xavier_uniform(self.U.weight)

        # final layer: create a matrix to use for the L binary classifiers as in
        # 2.3
        self.final = nn.Linear(num_filter_maps, self.Y)
        xavier_uniform(self.final.weight)

        # initialize with trained code embeddings if applicable
        if code_emb:
            self._code_emb_init(code_emb, self.dicts)
            # also set conv weights to do sum of inputs
            weights = torch.eye(self.embed_size).unsqueeze(2).expand(-1, -1, kernel_size) / kernel_size
            self.conv.weight.data = weights.clone()
            self.conv.bias.data.zero_()

        # conv for label descriptions as in 2.5
        # description module has its own embedding and convolution layers
        if lmbda > 0:
            W = self.embed.weight.data
            self.desc_embedding = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.desc_embedding.weight.data = W.clone()

            self.label_conv = nn.Conv1d(
                self.embed_size,
                num_filter_maps,
                kernel_size=kernel_size,
                padding=int(floor(kernel_size / 2)),
            )
            xavier_uniform(self.label_conv.weight)

            self.label_fc1 = nn.Linear(num_filter_maps, num_filter_maps)
            xavier_uniform(self.label_fc1.weight)

    def _code_emb_init(self, code_emb, dicts):
        code_embs = KeyedVectors.load_word2vec_format(code_emb)
        weights = np.zeros(self.final.weight.size())
        for i in range(self.Y):
            code = dicts["ind2c"][i]
            weights[i] = code_embs[code]
        self.U.weight.data = torch.Tensor(weights).clone()
        self.final.weight.data = torch.Tensor(weights).clone()

    def forward(self, x, target, desc_data=None, get_attention=True):
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

        if desc_data is not None:
            # run descriptions through description module
            b_batch = self.embed_descriptions(desc_data, self.gpu)
            # get l2 similarity loss
            diffs = self._compare_label_embeddings(target, b_batch, desc_data)
        else:
            diffs = None

        # final sigmoid to get predictions
        yhat = y
        loss = self._get_loss(yhat, target, diffs)
        return yhat, loss, alpha


@ConfigMapper.map("models", "CNN")
class VanillaConv(BaseModel):
    def __init__(
        self,
        num_classes=50,
        dataset_dir=None,
        version="mimic3",
        embed_file=None,
        kernel_size=10,
        num_filter_maps=50,
        embed_size=100,
        dropout=0.5,
        **kwargs
    ):
        super(VanillaConv, self).__init__(
            Y=num_classes,
            dataset_dir=dataset_dir,
            version=version,
            embed_file=embed_file,
            dropout=dropout,
            embed_size=embed_size,
        )
        # initialize conv layer as in 2.1
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size)
        xavier_uniform(self.conv.weight)

        # linear output
        self.fc = nn.Linear(num_filter_maps, self.Y)
        xavier_uniform(self.fc.weight)

    def forward(self, x, target, desc_data=None, get_attention=False):
        # embed
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        # conv/max-pooling
        c = self.conv(x)
        if get_attention:
            # get argmax vector too
            x, argmax = F.max_pool1d(F.tanh(c), kernel_size=c.size()[2], return_indices=True)
            attn = self.construct_attention(argmax, c.size()[2])
        else:
            x = F.max_pool1d(F.tanh(c), kernel_size=c.size()[2])
            attn = None
        x = x.squeeze(dim=2)

        # linear output
        x = self.fc(x)

        # final sigmoid to get predictions
        yhat = x
        loss = self._get_loss(yhat, target)
        return yhat, loss, attn

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
