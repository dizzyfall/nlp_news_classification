'''
@Project ：NLPNewsClassification 
@File    ：vocabulary.py
@Author  ：DZY
@Date    ：2025/3/10 17:19 
'''
import collections


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        # Flatten a 2D list if needed
        if reserved_tokens is None:
            reserved_tokens = []
        if tokens is None:
            tokens = []
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        counter = collections.Counter(tokens)
        # print("counter:",list(counter.items())[:10])
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # print("token_freqs:",list(self.token_freqs)[:10])
        self.idx_to_token = list(
            sorted(set(['<unk>'] + reserved_tokens + [token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    def create_vocab_txt(self, vocab_txt_relative_path):
        """
        将词表写到txt文件中

        Args:
            vocab_txt_relative_path:

        Returns:
            预训练数据集的词表txt文件
        """
        tokens = []
        indices = []
        for token, idx in self.token_to_idx.items():
            tokens.append(token)
            indices.append(idx)
        with open(vocab_txt_relative_path, 'w') as file:
            for token in tokens:
                # 这里词元后面有一个'\n'，在后续使用词元时要把'\n'去除掉
                file.write(f"{token}\n")

    @property
    def unk(self):
        return self.token_to_idx['<unk>']
