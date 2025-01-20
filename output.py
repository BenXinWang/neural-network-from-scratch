import numpy as np


#通过 predict 计算概率分布，loss 计算损失，diff 计算梯度。
class Softmax:
    def predict(self, X):#X 的 Softmax 概率分布
        exp_scores = np.exp(X)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def loss(self, X, y):
        num_examples = X.shape[0]
        probs = self.predict(X)
        corect_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(corect_logprobs)
        return 1./num_examples * data_loss#1. / 3 是一种 Python 表达式，表示用 浮点数进行除法运算

    def diff(self, X, y):
        num_examples = X.shape[0]
        probs = self.predict(X)
        probs[range(num_examples), y] -= 1
        return probs