'''
@Project ：NLPNewsClassification 
@File    ：fine_tuning_bert.py
@Author  ：DZY
@Date    ：2025/3/14 12:38 
'''
import os

import torch
from torch import nn

from src.data import vocabulary
from src.model import bert
from src.utils import timer, checkpoints


def _load_pretrained_vocab_txt_file(pretrained_vocab_relative_path):
    """
    加载预训练BERT的词表txt文件

    Returns:
        词元一维列表，每个元素是一个词元

    """
    script_directory = os.getcwd()
    pretrained_vocab_txt_file_absolute_path = os.path.join(script_directory, pretrained_vocab_relative_path)
    with open(pretrained_vocab_txt_file_absolute_path, 'r') as file:
        lines = file.readlines()
    return ' '.join(lines).split()


def _load_pretrained_vocab(pretrained_vocab_relative_path):
    pretrained_vocab_list = _load_pretrained_vocab_txt_file(pretrained_vocab_relative_path)
    pretrained_vocab = vocabulary.Vocab()
    pretrained_vocab.idx_to_token = pretrained_vocab_list
    pretrained_vocab.token_to_idx = {token: idx for idx, token in enumerate(pretrained_vocab.idx_to_token)}
    return pretrained_vocab


def load_pretrained_model(query_size, key_size, value_size, num_hiddens, normalized_shape, ffn_num_input,
                          ffn_num_hiddens, num_heads, num_layers, mlm_in_features, mlm_hiddens, nsp_in_features,
                          nsp_hiddens, dropout, max_len, devices, pretrained_vocab_relative_path,
                          checkpoint_relative_path):
    """

    Args:
        query_size:
        key_size:
        value_size:
        num_hiddens:
        normalized_shape:
        ffn_num_input:
        ffn_num_hiddens:
        num_heads:
        num_layers:
        mlm_in_features:
        mlm_hiddens:
        nsp_in_features:
        nsp_hiddens:
        dropout:
        max_len:
        devices:
        pretrained_vocab_relative_path:
        checkpoint_relative_path:

    Returns:

    """
    pretrained_vocab = _load_pretrained_vocab(pretrained_vocab_relative_path)
    # print("pretrained_vocab",pretrained_vocab.token_to_idx)
    pretrained_bert = bert.BERTLM(len(pretrained_vocab), query_size, key_size, value_size, num_hiddens,
                                  normalized_shape, ffn_num_input,
                                  ffn_num_hiddens, num_heads, num_layers, mlm_in_features, mlm_hiddens, nsp_in_features,
                                  nsp_hiddens,
                                  dropout, max_len=max_len)
    checkpoints.load_pretrained_model_params(pretrained_bert, checkpoint_relative_path, devices[0])
    return pretrained_bert, pretrained_vocab


class BERTClassifier(nn.Module):
    def __init__(self, pretrained_bert, classifier_num_input, classifier_num_hiddens, classifier_num_output):
        super(BERTClassifier, self).__init__()
        self.encoder = pretrained_bert.encoder
        self.classifier = nn.Sequential(nn.Linear(classifier_num_input, classifier_num_hiddens), nn.ReLU(),
                                        nn.Linear(classifier_num_hiddens, classifier_num_output))

    def forward(self, tokens, segments, valid_lens):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        return self.classifier(encoded_X[:, 0, :])


# def accuracy(y_hat, y):
#     """Compute the number of correct predictions.
#
#     Defined in :numref:`sec_utils`"""
#     if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
#         y_hat = d2l.argmax(y_hat, axis=1)
#     cmp = d2l.astype(y_hat, y.dtype) == y
#     return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))


# todo
def get_num_correct_preds(y_hat, y):
    """

    Args:
        y_hat:
        y: 这里y必须是标签列表的下标，比如这里分类是14类，标签是[1,2,3]，分类列表是[1,2,3,...,14]，这里y必须是标签对应分类列表的下标[0,1,2]，这样才能和y_hat对应。
        这里简单的做法是把y中每个元素都减一，后面优化代码时，再在数据上做转换

    Returns:

    """
    preds_max_prob_indices = torch.argmax(y_hat, dim=1)
    num_correct_preds = (preds_max_prob_indices == (y - 1)).sum().item()
    return num_correct_preds


def evaluate_accuracy(net, valid_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    num_samples = 0
    with torch.no_grad():
        for tokens_X, segments_X, valid_lens_X, labels_Y in valid_iter:
            tokens_X = tokens_X.to(device)
            segments_X = segments_X.to(device)
            valid_lens_X = valid_lens_X.to(device)
            labels_Y = labels_Y.to(device)
            Y_hat = net(tokens_X, segments_X, valid_lens_X)
            num_correct_preds = get_num_correct_preds(Y_hat, labels_Y)
            num_samples += tokens_X.shape[0]
    return round(num_correct_preds / num_samples, 4)


def fine_tuning(ft_train_iter, ft_valid_iter, net, loss, lr, num_epochs, devices, ft_checkpoints_relative_path):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    ft_optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    ft_timer = timer.Timer()
    current_ft_iter_step = 0
    total_ft_train_loss = 0.0
    total_ft_train_correct_preds = 0
    total_ft_processed_samples = 0
    total_ft_valid_acc = 0.0
    ft_info_interval = 100

    print("start fine tuning...")
    for epoch in range(num_epochs):
        net.train()
        for tokens_X, segments_X, valid_lens_X, labels_Y in ft_train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_X = valid_lens_X.to(devices[0])
            labels_Y = labels_Y.to(devices[0])

            ft_timer.start()
            ft_optimizer.zero_grad()
            Y_hat = net(tokens_X, segments_X, valid_lens_X)

            total_ft_train_correct_preds += get_num_correct_preds(Y_hat, labels_Y)

            l = loss(Y_hat, labels_Y)
            l.sum().backward()
            ft_optimizer.step()
            ft_timer.stop()

            total_ft_train_loss += l.sum().item()
            total_ft_processed_samples += tokens_X.shape[0]

            if (current_ft_iter_step + 1) % ft_info_interval == 0:
                cum_time_list = ft_timer.get_cumulate_time()
                print(
                    f"Iter Steps: {current_ft_iter_step + 2 - ft_info_interval}-{current_ft_iter_step + 1} ---- "
                    f"Cumulative Avg Loss: {total_ft_train_loss / (current_ft_iter_step + 1):.4f} ---- "
                    f"Cumulative Correct Train Preds/Cumulative Processed Samples: {total_ft_train_correct_preds}/{total_ft_processed_samples} ---- "
                    f"Cumulative Avg Train Acc: {total_ft_train_correct_preds / total_ft_processed_samples:.4f} ---- "
                    f"Cumulative Iter Time: {cum_time_list[current_ft_iter_step]:.4f} sec")

            current_ft_iter_step += 1

        total_ft_valid_acc = evaluate_accuracy(net, ft_valid_iter)
        print(f"epoch {epoch} Avg Train Acc: {total_ft_train_correct_preds / total_ft_processed_samples:.4f}")
        print(f"epoch {epoch} Avg Valid Acc: {total_ft_valid_acc}")

        checkpoint_dir_name = f"checkpoint_epoch_{epoch + 1}"
        checkpoint_file_name = f"checkpoint_epoch_{epoch + 1}.pth"
        checkpoints.save_finetuning_model(net, ft_checkpoints_relative_path, checkpoint_dir_name, checkpoint_file_name)

    ft_total_time = ft_timer.get_total_time()
    print("FT Total Time: ", timer.format_duration(ft_total_time))
    print("FT Total Steps: ", current_ft_iter_step)
    print(f"FT Avg Train Acc: {total_ft_train_correct_preds / total_ft_processed_samples:.4f}")
    print("FT Total Num Processed Samples: ", total_ft_processed_samples)
    print(f"FT Processing Samples Speed: {total_ft_processed_samples / ft_total_time:.4f} samples/sec")
    print("FT Avg Valid Acc: ", total_ft_valid_acc)
