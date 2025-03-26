'''
@Project ：NLPNewsClassification 
@File    ：fine_tuning_data_create.py
@Author  ：DZY
@Date    ：2025/3/18 20:19 
'''
import os

import torch
from torch.utils.data import Dataset

from src.data import data_processing


def load_fine_tuning_set_data(fine_tuning_set_relative_path, sequence_length):
    script_directory = os.getcwd()
    full_path = os.path.join(script_directory, fine_tuning_set_relative_path)
    with open(full_path, 'r') as file:
        lines = file.readlines()
    samples_tokens = []
    labels = []
    for line in lines:
        data = line.split('\t')
        labels.append(int(data[0]))
        sample_tokens = data[1].split()
        num_sample_tokens = len(sample_tokens)
        if num_sample_tokens < sequence_length:
            samples_tokens.append(sample_tokens)
        else:
            samples_tokens.append(sample_tokens[:sequence_length])
    return samples_tokens, labels


def _create_ft_inputs(samples_tokens, max_len):
    """
    构造所有样本的BERT形式的输入序列

    Args:
        samples_tokens: 所有样本序列，二维列表，每个子列表是一个样本序列，子列表每个元素是一个词元
        max_len: BERT输入序列最大长度

    Returns:
        所有样本的BERT形式的输入序列及其片段索引
        元组列表：[(样本的BERT形式的输入序列,其片段索引),...]

    """
    input_data = []
    for sample_tokens in samples_tokens:
        # 如果原本输入序列加上<cls>'和'<sep>是否会超过max_len
        while (len(sample_tokens) + 2) > max_len:
            # 超过就弹出最后一个词
            sample_tokens.pop()
        tokens, segments = data_processing.get_tokens_and_segments(sample_tokens)
        input_data.append((tokens, segments))
    return input_data


def _pad_ft_inputs(input_data, max_len, vocab):
    all_token_ids, all_segments, valid_lens, = [], [], []
    for tokens, segments in input_data:
        all_token_ids.append(torch.tensor(vocab[tokens] + [vocab['<pad>']] * (
                max_len - len(tokens)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
                max_len - len(segments)), dtype=torch.long))
        # valid_lens不包括'<pad>'的计数
        valid_lens.append(torch.tensor(len(tokens), dtype=torch.float32))
    return (all_token_ids, all_segments, valid_lens)


class FineTuningDataset(Dataset):
    def __init__(self, dataset, max_len, vocab):
        samples_tokens = dataset[0]
        self.labels = torch.tensor(dataset[1])
        self.vocab = vocab
        self.max_len = max_len
        input_data = _create_ft_inputs(samples_tokens, max_len)
        (self.all_token_ids, self.all_segments, self.valid_lens) = _pad_ft_inputs(input_data, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.labels[idx])

    def __len__(self):
        return len(self.all_token_ids)


def create_fine_tuning_iter(batch_size, max_len, sequence_length, vocab, ft_train_set_relative_path,
                            ft_validate_set_relative_path, is_train=True):
    if is_train:
        ft_set_relative_path = ft_train_set_relative_path
    else:
        ft_set_relative_path = ft_validate_set_relative_path
    dataset = load_fine_tuning_set_data(ft_set_relative_path, sequence_length)
    ft_dataset = FineTuningDataset(dataset, max_len, vocab)
    ft_iter = torch.utils.data.DataLoader(ft_dataset, batch_size, shuffle=is_train, num_workers=0)
    return ft_iter
