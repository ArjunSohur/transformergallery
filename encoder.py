# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from multi_head_attention import MultiHeadAttention


# ----------------------------------------------------------------------------------------------------------------------
# Methods
# ----------------------------------------------------------------------------------------------------------------------
def add_and_norm(old_tensor: Tensor, new_tensor: Tensor):
    added_tensor = old_tensor + new_tensor

    mean = torch.mean(added_tensor).item()
    std_dev = torch.std(added_tensor).item()

    normalized_tensor = F.normalize(added_tensor, mean, std_dev)

    return normalized_tensor


class Encoder(nn.Module):
    def __init__(self, input_seq_length: int, num_attn_heads: int, embedding_dimension: int, queries_keys_hidden_dimension: int,
                 values_hidden_dimension: int):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(input_seq_length*num_attn_heads, num_attn_heads)

        self.mh_attn = MultiHeadAttention(self, num_attn_heads, embedding_dimension, queries_keys_hidden_dimension,
                 values_hidden_dimension)

