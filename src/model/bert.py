'''
@Project ：NLPNewsClassification 
@File    ：bert.py
@Author  ：DZY
@Date    ：2025/3/11 21:32 
'''
import torch
from torch import nn

from src.model import encoder, pretrain_tasks


class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, query_size, key_size, value_size, num_hiddens, normalized_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout, max_len=1000, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))
        self.blocks = nn.Sequential()
        for i in range(num_layers):
            self.blocks.add_module(f"TransformerEncoderBlock:{i}",
                                   encoder.TransformerEncoderBlock(query_size, key_size, value_size, num_hiddens,
                                                                   normalized_shape, ffn_num_input, ffn_num_hiddens,
                                                                   num_heads, dropout, True))

    def forward(self, tokens, segments, valid_lens):
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for block in self.blocks:
            X = block(X, valid_lens)
        return X


# vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
# norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
# bertencoder = BERTEncoder(vocab_size, 768, 768, 768, num_hiddens, norm_shape, ffn_num_input,
#                       ffn_num_hiddens, num_heads, num_layers, dropout)
# print(bertencoder)
# tokens = torch.randint(0, vocab_size, (2, 8))
# segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
# encoded_X = bertencoder(tokens, segments, None)
# print(encoded_X.shape)

class BERTLM(nn.Module):
    def __init__(self, vocab_size, query_size, key_size, value_size, num_hiddens, normalized_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, mlm_in_features, mlm_hiddens, nsp_in_features, nsp_hiddens,
                 dropout, max_len=1000, **kwargs):
        super(BERTLM, self).__init__(**kwargs)
        self.encoder = BERTEncoder(vocab_size, query_size, key_size, value_size, num_hiddens, normalized_shape,
                                   ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, max_len=max_len)
        self.mlm = pretrain_tasks.MaskLM(mlm_in_features, mlm_hiddens, vocab_size)
        self.nsp = pretrain_tasks.NextSentencePred(nsp_in_features, nsp_hiddens)

    def forward(self, tokens, segments, valid_lens=None, pred_position=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_position is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_position)
        else:
            mlm_Y_hat = None
        # 下一句子预测只需要<cls>这个特殊标识符的信息
        nsp_Y_hat = self.nsp(encoded_X[:, 0, :])
        return encoded_X, mlm_Y_hat, nsp_Y_hat
