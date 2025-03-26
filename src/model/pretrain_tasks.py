'''
@Project ：NLPNewsClassification 
@File    ：pretrain_tasks.py
@Author  ：DZY
@Date    ：2025/3/12 11:34 
'''
import torch
from torch import nn


class MaskLM(nn.Module):
    def __init__(self, mlm_in_features, mlm_hiddens, vocab_size, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(mlm_in_features, mlm_hiddens), nn.ReLU(), nn.LayerNorm(mlm_hiddens),
                                 nn.Linear(mlm_hiddens, vocab_size))

    def forward(self, X, pred_position):
        # pred_position:(batch_size,num_pred_position)
        # X:(batch_size,num_steps,num_hiddens)
        num_pred_position = pred_position.shape[-1]
        batch_size = X.shape[0]
        # pred_position:(batch_size*num_pred_position)
        pred_position = pred_position.reshape(-1)
        batch_indices = torch.arange(0, batch_size)
        batch_indices = torch.repeat_interleave(batch_indices, num_pred_position)
        # 花式索引
        # masked_X:(batch_size*num_pred_position,num_hiddens)
        # 取的是每个掩蔽位置的词向量
        masked_X = X[batch_indices, pred_position]
        # masked_X:(batch_size,num_pred_position,num_hiddens)
        masked_X = masked_X.reshape((batch_size, num_pred_position, -1))

        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


# vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
# norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
# bertencoder = bert.BERTEncoder(vocab_size, 768, 768, 768, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout)
# tokens = torch.randint(0, vocab_size, (2, 8))
# segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
# encoded_X = bertencoder(tokens, segments, None)
#
#
# mlm = MaskLM(768, num_hiddens, vocab_size)
# mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
# mlm_Y_hat = mlm(encoded_X, mlm_positions)
# print(mlm_Y_hat.shape)
#
# mlm_Y=torch.tensor([[7,8,9],[10,20,30]])
# loss=nn.CrossEntropyLoss(reduction='none')
# mlm_l=loss(mlm_Y_hat.reshape((-1,vocab_size)),mlm_Y.reshape(-1))
# print(mlm_l.shape)

class NextSentencePred(nn.Module):
    def __init__(self, nsp_in_features, nsp_hiddens, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(nsp_in_features, nsp_hiddens), nn.Tanh(), nn.Linear(nsp_hiddens, 2))

    def forward(self, X):
        return self.mlp(X)

# encoded_X = torch.flatten(encoded_X, start_dim=1)
# # NSP的输入形状:(batchsize，num_hiddens)
# nsp = NextSentencePred(encoded_X.shape[-1], 768)
# nsp_Y_hat = nsp(encoded_X)
# print(nsp_Y_hat.shape)
#
# nsp_y = torch.tensor([0, 1])
# nsp_l = loss(nsp_Y_hat, nsp_y)
# print(nsp_l.shape)
