# flake8: noqa

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_


class LabelWiseAttn(nn.Module):
    """
    A Label-wise Attention layer (as implemented in CAML, DCAN, etc.).
    References:
        Papers: https://arxiv.org/abs/1802.05695 (Section 2.2)
        Repository: https://github.com/jamesmullenbach/caml-mimic/blob/master/learn/models.py#L184

    Args:
        input_size (int): The size of the input, i.e., the number of channels
                          if the output is from a convolutional layer/embedding
                          size if the output is from a fully connected layer.
        num_classes (int): The number of classes.
    """

    def __init__(self, input_size, num_classes):
        super(LabelWiseAttn, self).__init__()
        self.U = nn.Linear(input_size, num_classes)
        xavier_uniform_(self.U.weight)

    def forward(self, x):
        att = self.U.weight.matmul(x.transpose(1, 2))  # [bs, Y, seq_len]
        alpha = F.softmax(att, dim=2)
        m = alpha.matmul(x)  # [bs, Y, dim]
        return m
