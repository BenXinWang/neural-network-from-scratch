import matplotlib.pyplot as plt
import numpy as np

# Helper function to plot a decision boundary
def plot_decision_boundary(pred_func, X, y):
    # 设置绘图的范围，并添加适当的边距（padding）
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5  # x 轴的最小值和最大值
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5  # y 轴的最小值和最大值
    h = 0.01  # 网格点的间隔，值越小分辨率越高，图像越精细

    # 生成网格点（覆盖整个绘图范围），用于计算每个点的分类结果
    # np.meshgrid 创建两个矩阵：一个存储所有网格点的 x 坐标，一个存储 y 坐标
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),  # x 方向的点，从 x_min 到 x_max，间隔为 h
        np.arange(y_min, y_max, h)  # y 方向的点，从 y_min 到 y_max，间隔为 h
    )

    # 将网格点展平（ravel），组合成二维坐标，用于输入到分类模型中
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])  # 调用预测函数，计算每个网格点的分类结果
    Z = Z.reshape(xx.shape)  # 将预测结果 Z 的形状还原为网格点的形状

    # 绘制决策边界（用颜色区分不同的分类区域）
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)  # 填充轮廓图，显示决策区域，颜色使用 Spectral 映射
    # 绘制训练数据点
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)  # 训练数据点，颜色根据标签 y 确定
