'''
@Project ：NLPNewsClassification 
@File    ：data_analyse.py
@Author  ：DZY
@Date    ：2025/2/25 15:42 
'''
import pandas as pd
from matplotlib import pyplot as plt


# 获取原始数据集基本属性
def get_dataset_properties(dataset_relative_path):
    dataset_df = pd.read_csv(dataset_relative_path, sep="\t")

    # 查看数据集维度
    print(dataset_df.shape)

    # 查看数据集行属性
    print(dataset_df.index)

    # 查看数据集列属性
    print(dataset_df.columns)

    # 查看数据集第一行数据
    print(dataset_df.iloc[0].values)

    # 查看数据集第一列数据
    # print(dataset_df["label"].values)
    print(dataset_df.iloc[:, 0].values)

    # 查看数据集前20条数据
    print(dataset_df.head(20))

    # 查看数据集是否存在空值
    print(dataset_df.isnull())


def read_dataset(dataset_relative_path):
    """
    读取数据集，提取标签和样本

    Args:
        dataset_relative_path: 数据集相对路径

    Returns:
        标签和样本列表

    """
    # 读取csv文件
    dataset_df = pd.read_csv(dataset_relative_path, sep="\t")
    dataset_dic = dataset_df.to_dict(orient="list")
    samples = [item for item in dataset_dic["text"]]
    labels = [item for item in dataset_dic["label"]]
    return samples, labels


def calculate_word_count(samples):
    """
    词频统计

    Args:
        samples: 样本列表

    Returns:
        按词频降序排序后的词频字典

    """
    words_list = [word for sample in samples for word in sample.split()]
    words_count = {}
    for word in words_list:
        if word not in words_count:
            words_count[word] = 1
        else:
            words_count[word] += 1
    words_count = sorted(words_count.items(), key=lambda x: x[1], reverse=True)
    return words_count


# 统计数据集单个样本序列长度
def calculate_sample_length(sample):
    return len(sample.split())


# 统计数据集所有样本长度，降序排序
def calculate_dataset_all_sample_length(samples):
    return sorted([calculate_sample_length(sample) for sample in samples], reverse=True)


# 前十条样本长度：[57921, 53527, 49186, 47535, 45810, 44665, 43377, 41894, 41105, 39959]
# print("序列:",calculate_dataset_all_sample_length(train_data[0])[:10])

# 绘制样本长度直方图
def show_sequence_length_hist(x_label, y_label, samples, x_start_range=0, x_end_range=5000, X_interval=100):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.hist([calculate_sample_length(line) for line in samples], bins=range(x_start_range, x_end_range, X_interval))
    plt.show()

# show_sequence_length_hist("token sequence","count",train_data[0])
