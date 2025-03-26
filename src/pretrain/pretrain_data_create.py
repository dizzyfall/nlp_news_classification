'''
@Project ：NLPNewsClassification 
@File    ：pretrain_data_create.py
@Author  ：DZY
@Date    ：2025/3/12 16:25 
'''
import os
import random

import torch
from torch.utils.data import Dataset

from ..data import data_processing
from ..data import vocabulary


def _get_inputs(samples_tokens, max_len):
    """
    构造所有样本的BERT形式的输入序列

    Args:
        samples_tokens: 所有样本序列，二维列表，每个子列表是一个样本序列，子列表每个元素是一个词元
        max_len: BERT输入序列最大长度

    Returns:
        所有样本的BERT形式的输入序列及其片段索引
        元组列表：[(样本的BERT形式的输入序列,其片段索引),...]

    """
    inputs = []
    for sample_tokens in samples_tokens:
        # 如果原本输入序列加上<cls>'和'<sep>是否会超过max_len
        if (len(sample_tokens) + 2) > max_len:
            # 超过就丢掉
            continue
        tokens, segments = data_processing.get_tokens_and_segments(sample_tokens)
        inputs.append((tokens, segments))
    return inputs


def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab):
    """
    用于掩蔽语言模型替换词元

    Args:
        tokens: 用于MLM的输入序列（BERT形式的输入），一维列表，每个元素是一个词元，包含特殊标识符
        candidate_pred_positions: 候选需要替换（预测）的词元位置下标索引（不包括特殊标识符，特殊标识符不被预测）
        num_mlm_preds: 需要替换（预测）的词元数量
        vocab: 词表

    Returns:
        随机替换后的序列
        替换的位置和替换之前的词元组成的元组列表

    """
    # 复制一份MLM输入序列词元列表
    mlm_input_tokens = [token for token in tokens]
    # 预测的词元位置索引和被替换前的词元
    pred_positions_and_labels = []
    # 随机打乱候选替换位置索引
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) > num_mlm_preds:
            break
        masked_tokens = None
        # 80%时间替换为<mask>
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10%时间保持不变
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10%时间替换为随机词元
            else:
                masked_token = random.choice(vocab.idx_to_token)
        # 替换输入序列中指定位置的词元
        mlm_input_tokens[mlm_pred_position] = masked_token
        # 保存替换的位置和替换之前的词元
        pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


def _get_mlm_data(tokens, vocab):
    """
    生成用于掩蔽语言模型的数据

    Args:
        tokens: 用于MLM的输入序列（BERT形式的输入），一维列表，每个元素是一个词元，包含特殊标识符
        vocab: 词表

    Returns:
        替换后的序列中所有词元的索引下标列表
        预测位置的下标列表
        替换前的词元索引下标列表

    """
    # 候选预测位置索引
    candidate_pred_position = []
    for index, token in enumerate(tokens):
        # 特殊标识符不参与词元预测
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_position.append(index)
    # MLM中只预测15%的词元
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(tokens, candidate_pred_position, num_mlm_preds,
                                                                      vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
    pred_positions = [position_and_label_tuple[0] for position_and_label_tuple in pred_positions_and_labels]
    pred_labels = [position_and_label_tuple[1] for position_and_label_tuple in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[pred_labels]


def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens, = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
                max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
                max_len - len(segments)), dtype=torch.long))
        # valid_lens不包括'<pad>'的计数
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
                max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # 填充词元的预测将通过乘以0权重在损失中过滤掉
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                    max_num_mlm_preds - len(pred_positions)),
                         dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
                max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels)


class PretrainDataset(Dataset):
    def __init__(self, samples, max_len):
        # 原始样本序列按单词分词
        samples_tokens = data_processing.tokenizer(samples)
        self.vocab = vocabulary.Vocab(samples_tokens, min_freq=5, reserved_tokens=['<cls>', '<sep>', '<pad>', '<mask>'])
        # 获取所有BERT形式的输入序列
        examples = []
        examples.extend(_get_inputs(samples_tokens, max_len))
        examples = [(_get_mlm_data(tokens, self.vocab) + (segments,)) for tokens, segments in examples]
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels) = _pad_bert_inputs(examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)


def get_dataloader_workers(num_workers):
    """
    使用指定个数进程来读取数据

    Args:
        num_workers: 进程数

    Returns:
        进程数

    """
    return num_workers


def read_pretrain_data(pretrain_sentences_relative_path):
    script_directory = os.getcwd()
    full_path = os.path.join(script_directory, pretrain_sentences_relative_path)
    with open(full_path, 'r') as file:
        samples = file.readlines()
    return samples


def load_pretrain_data(pretrain_data_relative_path, vocab_txt_relative_path, batch_size, max_len):
    num_workers = get_dataloader_workers(0)
    samples = read_pretrain_data(pretrain_data_relative_path)
    pretrain_set = PretrainDataset(samples, max_len)
    pretrain_set.vocab.create_vocab_txt(vocab_txt_relative_path)
    pretrain_iter = torch.utils.data.DataLoader(pretrain_set, batch_size, shuffle=True, num_workers=num_workers)
    return pretrain_iter, pretrain_set.vocab

# batch_size, max_len = 128, 256
# train_iter, vocab = load_pretrain_data(batch_size, max_len)
#
# for (tokens_X, segments_X, valid_lens_X, pred_positions_X, mlm_weights_X,mlm_Y) in train_iter:
#     print(tokens_X.shape, segments_X.shape, valid_lens_X.shape,
#           pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape)
#     break
