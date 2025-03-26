'''
@Project ：NLPNewsClassification 
@File    ：data_processing.py
@Author  ：DZY
@Date    ：2025/3/10 11:55 
'''


def tokenizer(samples):
    """
    按照单词分词

    Args:
        samples: 一维列表，每个元素是一个样本

    Returns:
        二维列表，每个子列表是一个样本，子列表中每个元素是一个词元

    Examples:
        samples: ["1 2 3 4","5 6 7 8"]
        Returns: [[1,2,3,4],[5,6,7,8]]

    """
    return [sample.split() for sample in samples]


def sample_truncate_pad(sample_tokens, num_steps, padding_token):
    """
    按num_steps长度截取或填充样本

    Args:
        sample_tokens: 分词后的样本词元列表
        num_steps: 样本要截取或填充的长度
        padding_token: 填充的词元

    Returns:
        截取或填充后的样本词元列表
    """
    sample_length = len(sample_tokens)
    if sample_length > num_steps:
        return sample_tokens[:num_steps]
    else:
        return sample_tokens + [padding_token] * (num_steps - sample_length)


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """
    构造BERT形式的输入序列及片段索引

    Args:
        tokens_a: 样本序列A,一维列表,其中每一个元素是一个词元
        tokens_b: 样本序列B,一维列表,其中每一个元素是一个词元

    Returns:
        BERT输入形式的序列及片段索引

    Examples:
        tokens_a: [1, 2, 3]
        tokens_b: [4, 5, 6]
        Returns:
            tokens: ['<cls>', 1, 2, 3, '<sep>', 4, 5, 6, '<sep>']
            segments: [0, 0, 0, 0, 0, 1, 1, 1, 1]

    """
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments
