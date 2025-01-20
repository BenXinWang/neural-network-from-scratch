import numpy as np

#计算单元

class MultiplyGate:
    def forward(self,W, X):
        return np.dot(X, W)#矩阵乘法

    def backward(self, W, X, dZ):#执行反向传播，计算梯度。
        dW = np.dot(np.transpose(X), dZ)
        dX = np.dot(dZ, np.transpose(W))
        return dW, dX

class AddGate:
    def forward(self, X, b):
        return X + b

    def backward(self, X, b, dZ):
        dX = dZ * np.ones_like(X)
        db = np.dot(np.ones((1, dZ.shape[0]), dtype=np.float64), dZ)
        return db, dX