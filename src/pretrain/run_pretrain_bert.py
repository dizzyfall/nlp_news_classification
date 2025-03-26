'''
@Project ：NLPNewsClassification
@File    ：run_pretrain_bert.py
@Author  ：DZY
@Date    ：2025/3/17 17:07
'''
import torch
from torch import nn

from src.model import bert
from src.pretrain import pretrain_data_create, pretrain_bert
from src.utils import environment, scheduler

batch_size, max_len = 128, 256
pretrain_data_relative_path = "../../dataset/train/pretrain/pretrain_sentences_test.txt"
# pretrain_data_relative_path = "../../dataset/train/pretrain/pretrain_sentences_128.txt"
vocab_txt_relative_path = "../../pretrain_results/vocab.txt"
checkpoints_relative_path = "../../pretrain_results/checkpoints"
# checkpoint_relative_path = "../../pretrain_results/checkpoints/checkpoint_step_10/checkpoint_step_10.pth"


pretrain_iter, vocab = pretrain_data_create.load_pretrain_data(pretrain_data_relative_path, vocab_txt_relative_path,
                                                               batch_size, max_len)

initial_size = 768
vocab_size = len(vocab)
query_size = initial_size
key_size = initial_size
value_size = initial_size
num_hiddens = initial_size
normalized_shape = [initial_size]
ffn_num_input = initial_size
ffn_num_hiddens = 3072
num_heads = 12
num_layers = 12
mlm_in_features = initial_size
mlm_hiddens = initial_size
nsp_in_features = initial_size
nsp_hiddens = initial_size
dropout = 0.1
lr = 1e-4
num_pretrain_iter_steps = 30

net = bert.BERTLM(vocab_size, query_size, key_size, value_size, num_hiddens, normalized_shape, ffn_num_input,
                  ffn_num_hiddens, num_heads, num_layers, mlm_in_features, mlm_hiddens, nsp_in_features, nsp_hiddens,
                  dropout, max_len=max_len)

devices = environment.try_all_gpus()
loss = nn.CrossEntropyLoss()

pretrain_optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
pretrain_scheduler = scheduler.BERTScheduler(pretrain_optimizer, num_hiddens, warmup_steps=10000)

step, total_mlm_loss, total_processed_samples, cum_time_list = pretrain_bert.load_pretrain_checkpoint(net,
                                                                                                      pretrain_optimizer,
                                                                                                      devices[0],
                                                                                                      checkpoint_relative_path=None)

pretrainbert = pretrain_bert.PretrainBERT(step, total_mlm_loss, total_processed_samples, cum_time_list)
pretrainbert.pretrain(pretrain_iter, net, loss, vocab_size, pretrain_optimizer, pretrain_scheduler,
                      num_pretrain_iter_steps,
                      checkpoints_relative_path, devices)
