'''
@Project ：NLPNewsClassification 
@File    ：data_create.py
@Author  ：DZY
@Date    ：2025/3/10 12:04 
'''
import torch
from torch.utils.data import TensorDataset, DataLoader

import data_analyse
import data_processing
import vocabulary


def create_data_iterator(data_processed_arrays, batch_size, is_train=True):
    """
    创建数据迭代器

    Args:
        data_processed_arrays:
        batch_size:
        is_train:

    Returns:

    """
    # 手写实现数据生成器方式
    # samples,labels=data_processed_arrays
    # num_samples=len(samples)
    # indices=list(range(num_samples))
    # random.shuffle(indices)
    # for i in range(0,num_samples,batch_size):
    #     batch_indices = torch.tensor(indices[i:min(i+batch_size,num_samples)])
    #     yield samples[batch_indices],labels[batch_indices]

    # 使用pytorch实现
    dataset = TensorDataset(*data_processed_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)


def load_data(batch_size, train_dataset_relative_path, num_steps=768):
    train_data = data_analyse.read_dataset(train_dataset_relative_path)
    train_tokens = data_processing.tokenizer(train_data[0])
    vocab = vocabulary.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
    train_features = torch.tensor(
        [data_processing.sample_truncate_pad(vocab[sample], num_steps, vocab['<pad>']) for sample in train_tokens])

    # torch.Size([200000, 768])
    # print(train_features.shape)

    train_iter = create_data_iterator((train_features, torch.tensor(train_data[1])), batch_size)

    # for X, y in train_iter:
    #     # X: torch.Size([256, 768]) y: torch.Size([256])
    #     print("X:", X.shape, "y:", y.shape)
    #     break
    # # 小批量数目: 782
    # print("小批量数目:", len(train_iter))

    return train_iter, vocab
