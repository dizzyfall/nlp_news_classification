'''
@Project ：NLPNewsClassification 
@File    ：dataset_create.py
@Author  ：DZY
@Date    ：2025/3/13 18:51 
'''
import os
import random

import data_analyse


# 划分用于预训练BERT的数据集和用于微调的数据集
class BERTDataset():
    def __init__(self, dataset_relative_path, pretrain_data_ratio, ft_train_data_ratio):
        self.origin_dataset = data_analyse.read_dataset(dataset_relative_path)
        self.pretrain_data_ratio = pretrain_data_ratio
        self.ft_train_data_ratio = ft_train_data_ratio
        self.num_origin_data = len(self.origin_dataset[0])
        self.origin_samples = self.origin_dataset[0]
        self.origin_labels = self.origin_dataset[1]
        # 将原始数据集的数据随机打乱
        origin_data_indices = list(range(0, self.num_origin_data))
        random.shuffle(origin_data_indices)
        # 随机打乱原始数据的下标
        self.origin_data_indices = origin_data_indices
        # 用于预训练的数据数量
        self.num_pretrain_data = int(self.num_origin_data * self.pretrain_data_ratio)
        # 用于微调的数据数量
        self.num_ft_data = int(self.num_origin_data - self.num_pretrain_data)
        # 微调数据中用于训练的数据数量
        self.num_ft_train_data = int(self.num_ft_data * self.ft_train_data_ratio)

    def get_pretrain_data(self):
        # 随机打乱后的用于预训练的样本
        pretrain_data_samples = [self.origin_samples[self.origin_data_indices[i]] for i in
                                 range(0, self.num_pretrain_data)]
        # 随机打乱后的用于预训练的标签
        pretrain_data_labels = [self.origin_labels[self.origin_data_indices[i]] for i in
                                range(0, self.num_pretrain_data)]
        return pretrain_data_samples, pretrain_data_labels

    def create_pretrain_data_txt_file(self):
        # 获取当前脚本的目录
        script_directory = os.getcwd()

        # 预训练数据集（只有样本）
        # 定义文件名
        pretrain_samples_set_file_name = "pretrain_samples_set" + ".txt"
        # 定义文件夹相对路径
        pretrain_dir_relative_path = "../../dataset/train/pretrain/"
        # 构建完整路径
        pretrain_samples_set_full_path = os.path.join(script_directory, pretrain_dir_relative_path,
                                                      pretrain_samples_set_file_name)

        # 预训练数据集（标签+样本）
        pretrain_set_file_name = "pretrain_set" + ".txt"
        pretrain_set_full_path = os.path.join(script_directory, pretrain_dir_relative_path, pretrain_set_file_name)

        pretrain_data_samples, pretrain_data_labels = self.get_pretrain_data()
        with open(pretrain_samples_set_full_path, "w") as file:
            for sample in pretrain_data_samples:
                file.write(f"{sample}\n")

        with open(pretrain_set_full_path, "w") as file:
            for label, sample in zip(pretrain_data_labels, pretrain_data_samples):
                file.write(f"{label}\t{sample}\n")

    # 读取预训练数据集，并按指定句子长度切分为一个个句子
    def create_pretrain_set_data_txt_file(self, sequence_length):
        script_directory = os.getcwd()
        pretrain_set_relative_path = "../../dataset/train/pretrain/pretrain_samples_set.txt"
        full_path = os.path.join(script_directory, pretrain_set_relative_path)
        with open(full_path, 'r') as file:
            lines = file.readlines()
        pretrain_tokens = ' '.join(lines).split()
        num_pretrain_tokens = len(pretrain_tokens)
        # print(num_pretrain_tokens)
        print(
            f"num_pretrain_tokens: {num_pretrain_tokens} sequence_length: {sequence_length} num_samples: {num_pretrain_tokens / sequence_length}")

        pretrain_sentences = [' '.join(pretrain_tokens[i:min(i + sequence_length, num_pretrain_tokens)]) for i in
                              range(0, num_pretrain_tokens, sequence_length)]
        pretrain_sentences = pretrain_sentences[0:600000]
        print(len(pretrain_sentences))
        pretrain_sentences_file_name = "pretrain_sentences_" + f"{sequence_length}_600000" + ".txt"
        pretrain_dir_relative_path = "../../dataset/train/pretrain/"
        pretrain_sentences_full_path = os.path.join(script_directory, pretrain_dir_relative_path,
                                                    pretrain_sentences_file_name)
        with open(pretrain_sentences_full_path, 'w') as file:
            for sentence in pretrain_sentences:
                file.write(f"{sentence}\n")

    def get_ft_train_and_validate_data(self):
        ft_samples = [self.origin_samples[self.origin_data_indices[i]] for i in
                      range(self.num_pretrain_data, self.num_origin_data)]
        ft_labels = [self.origin_labels[self.origin_data_indices[i]] for i in
                     range(self.num_pretrain_data, self.num_origin_data)]
        ft_train_samples = ft_samples[:self.num_ft_train_data]
        ft_train_labels = ft_labels[:self.num_ft_train_data]
        ft_validate_samples = ft_samples[self.num_ft_train_data:self.num_ft_data]
        ft_validate_labels = ft_labels[self.num_ft_train_data:self.num_ft_data]
        return ft_train_samples, ft_train_labels, ft_validate_samples, ft_validate_labels

    def create_ft_train_and_validate_txt_file(self):
        script_directory = os.getcwd()
        # 微调训练集
        ft_train_set_file_name = "fine_tuning_train_set" + ".txt"
        ft_train_dir_relative_path = "../../dataset/train/fine_tuning/train/"
        ft_train_set_full_path = os.path.join(script_directory, ft_train_dir_relative_path,
                                              ft_train_set_file_name)

        # 微调验证集
        ft_validate_set_file_name = "fine_tuning_validate_set" + ".txt"
        ft_validate_dir_relative_path = "../../dataset/train/fine_tuning/validate/"
        ft_validate_set_full_path = os.path.join(script_directory, ft_validate_dir_relative_path,
                                                 ft_validate_set_file_name)

        ft_train_samples, ft_train_labels, ft_validate_samples, ft_validate_labels = self.get_ft_train_and_validate_data()
        with open(ft_train_set_full_path, "w") as file:
            for label, sample in zip(ft_train_labels, ft_train_samples, ):
                file.write(f"{label}\t{sample}\n")

        with open(ft_validate_set_full_path, "w") as file:
            for label, sample in zip(ft_validate_labels, ft_validate_samples):
                file.write(f"{label}\t{sample}\n")


train_dataset_relative_path = "../../dataset/train/train_set.csv"
BERTDatasetPartition = BERTDataset(train_dataset_relative_path, 0.9, 0.8)
# BERTDatasetPartition.create_pretrain_data_txt_file()
# BERTDatasetPartition.create_ft_train_and_validate_txt_file()
BERTDatasetPartition.create_pretrain_set_data_txt_file(128)
