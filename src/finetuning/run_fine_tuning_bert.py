'''
@Project ：NLPNewsClassification 
@File    ：run_fine_tuning_bert.py
@Author  ：DZY
@Date    ：2025/3/18 21:11 
'''
from torch import nn

from src.finetuning import fine_tuning_bert, fine_tuning_data_create
from src.utils import environment

max_len = 256
initial_size = 768
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
devices = environment.try_all_gpus()

pretrained_vocab_relative_path = "../../pretrain_results/vocab.txt"
checkpoints_relative_path = "../../pretrain_results/checkpoints/checkpoint_step_10/checkpoint_step_10.pth"
ft_train_set_relative_path = "../../dataset/train/fine_tuning/train/fine_tuning_train_set_test.txt"
ft_validate_set_relative_path = "../../dataset/train/fine_tuning/validate/fine_tuning_validate_set_test.txt"
ft_checkpoints_relative_path = "../../fine_tuning_results"

pretrained_bert, pretrained_vocab = fine_tuning_bert.load_pretrained_model(query_size, key_size, value_size,
                                                                           num_hiddens, normalized_shape, ffn_num_input,
                                                                           ffn_num_hiddens, num_heads, num_layers,
                                                                           mlm_in_features, mlm_hiddens,
                                                                           nsp_in_features, nsp_hiddens,
                                                                           dropout, max_len, devices,
                                                                           pretrained_vocab_relative_path,
                                                                           checkpoints_relative_path)

batch_size = 32
sequence_length = 256
ft_train_iter = fine_tuning_data_create.create_fine_tuning_iter(batch_size, max_len, sequence_length, pretrained_vocab,
                                                                ft_train_set_relative_path,
                                                                ft_validate_set_relative_path, is_train=True)
ft_valid_iter = fine_tuning_data_create.create_fine_tuning_iter(batch_size, max_len, sequence_length, pretrained_vocab,
                                                                ft_train_set_relative_path,
                                                                ft_validate_set_relative_path, is_train=False)

# for (tokens_X, segments_X, valid_lens_X, Y) in ft_train_iter:
#     print(tokens_X.shape, segments_X.shape, valid_lens_X.shape,Y.shape)
#     break


lr, num_epochs = 2e-5, 3
loss = nn.CrossEntropyLoss(reduction='none')
classifier_num_input, classifier_num_hiddens, classifier_num_output = 768, 3072, 14

net = fine_tuning_bert.BERTClassifier(pretrained_bert, classifier_num_input, classifier_num_hiddens,
                                      classifier_num_output)
fine_tuning_bert.fine_tuning(ft_train_iter, ft_valid_iter, net, loss, lr, num_epochs, devices,
                             ft_checkpoints_relative_path)
