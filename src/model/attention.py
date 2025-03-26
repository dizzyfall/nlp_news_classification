'''
@Project ：NLPNewsClassification 
@File    ：attention.py
@Author  ：DZY
@Date    ：2025/3/11 16:12 
'''
import math

import torch
from torch import nn


def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis.

    Defined in :numref:`sec_attention-scoring-functions`"""

    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0.0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


# print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))
# print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])))

class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(attention_scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_hiddens, num_heads, dropout, use_bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=use_bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=use_bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=use_bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=use_bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))
        # print("queries.shape:",queries.shape,"keys.shape:",keys.shape,"values.shape:",values.shape)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)
        # print("output.shape:",output.shape)

        output_concat = self.transpose_output(output)
        # print("output_concat.shape:",output_concat.shape)

        result = self.W_o(output_concat)
        # print("result.shape:", result.shape)

        return result

    def transpose_qkv(self, X):
        # (batch_size,num_steps,num_heads,num_hiddens)->(batch_size,num_steps,num_heads,num_hiddens/num_heads)
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        # ->(batch_size,num_heads,num_steps,num_hiddens/num_heads)
        X = X.permute(0, 2, 1, 3)
        # ->(batch_size*num_heads,num_steps,num_hiddens/num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        # (batch_size*num_heads,num_steps,num_hiddens/num_heads)->(batch_size,num_heads,num_steps,num_hiddens/num_heads)
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        # ->(batch_size,num_steps,num_heads,num_hiddens/num_heads)
        X = X.permute(0, 2, 1, 3)
        # ->(batch_size,num_steps,num_hiddens)
        return X.reshape(X.shape[0], X.shape[1], -1)

# num_hiddens, num_heads = 100, 5
# attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
#                                num_hiddens, num_heads, 0.5)
# print(attention.eval())
# batch_size, num_queries = 2, 4
# num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
# X = torch.ones((batch_size, num_queries, num_hiddens))
# Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
# print(attention(X, Y, Y, valid_lens).shape)
