'''
@Project ：NLPNewsClassification 
@File    ：encoder.py
@Author  ：DZY
@Date    ：2025/3/11 16:13 
'''
from torch import nn

from src.model import attention


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.layernorm = nn.LayerNorm(normalized_shape)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, Y):
        return self.layernorm(self.dropout(Y) + X)


# add_norm= AddNorm([3,4],0.1)
# print(add_norm(torch.ones((2,3,4)),torch.ones((2,3,4))).shape)
# print(add_norm.forward(torch.ones((2,3,4)),torch.ones((2,3,4))).shape)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_hiddens, normalized_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, dropout, use_bias=False, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.multi_head_attention = attention.MultiHeadAttention(query_size, key_size, value_size, num_hiddens,
                                                                 num_heads, dropout, use_bias)
        self.add_norm1 = AddNorm(normalized_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.add_norm2 = AddNorm(normalized_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.add_norm1(X, self.multi_head_attention(X, X, X, valid_lens))
        return self.add_norm2(Y, self.ffn(Y))
