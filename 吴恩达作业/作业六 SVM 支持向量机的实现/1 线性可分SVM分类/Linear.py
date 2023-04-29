import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
def plot_scatter(x1, x2, y):
    plt.scatter(x1, x2, c=y.flatten())#为各个点标记颜色
    plt.xlabel("x1")
    plt.ylabel("X2")
data1 = loadmat("C:\\Users\\Administrator\\Desktop\\吴恩达作业\\作业六 SVM 支持向量机的实现\\题目\\ex6\\ex6data1.mat")
X = data1["X"]
y = data1["y"]
model = svm.SVC(C=1, kernel='linear')
model.fit(X, y.ravel())#y.ravel将y变为一位数组，model.fit对给定的训练集和标签进行训练
def plot_boundary(model, X, title):
    x_max, x_min = np.max(X[..., 0]), np.min(X[..., 0])#获取最小值和最大值
    y_max, y_min = np.max(X[..., 1]), np.min(X[..., 1])
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000), np.linspace(y_min, y_max, 1000))#生成坐标矩阵，即xx为横坐标的网点矩阵，而yy为纵坐标的网点矩阵，xx与yy一一对应
    #concatenate为对多个数组进行拼接，axis=1时为按列拼接，axis=0时为按对应行拼接。
    p = model.predict(np.concatenate((xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)), axis=1))#对上述的网格点进行标记，reshape(-1, 1)表示转换为一列
    print(p)
    plt.contour(xx, yy, p.reshape(xx.shape))#xx.shape表示其矩阵大小，将p转化为xx的矩阵大小,实际上就是形成一个面，然后对其进行标记,使用contourf可以用颜色标记不同等高线
    plt.title(title)
plot_boundary(model, X, "SVM Decision Boundary with C = 1 (Example Dataset 1)")
plot_scatter(X[..., 0], X[..., 1], y)
plt.show()