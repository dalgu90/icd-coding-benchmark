import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_


class LabelWiseAttn(nn.Module):
    def __init__(self, input_size, num_classes):
        self.U = nn.Linear(input_size, num_classes)
        xavier_uniform_(self.U.weight)

    def forward(self, x):
        att = self.U.weight.matmul(x.transpose(1, 2))  # [bs, Y, seq_len]
        alpha = F.softmax(att, dim=2)
        m = alpha.matmul(x)  # [bs, Y, dim]
        return m
