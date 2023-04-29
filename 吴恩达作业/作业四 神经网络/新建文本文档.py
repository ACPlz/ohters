#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib
import scipy.optimize as opt
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
data = loadmat("C:\\Users\\Administrator\\Desktop\\吴恩达作业\\作业四 神经网络\\ex4data1.mat")
X = data['X']
y = data['y']
print(X.shape, y.shape)

#可视化数据部分
def display(x):
    (m, n) = x.shape   #100*400
    width = np.round(np.sqrt(n)).astype(int)
    height = (n / width).astype(int)

    gap = 1  #展示图像间的距离
    display_array = -np.ones((gap + 10 * (width + gap), gap + 10 * (height + gap)))
    # 将样本填入到display矩阵中
    curr_ex = 0
    for j in range(10):
        for i in range(10):
            if curr_ex > m:
                break
            # Get the max value of the patch
            max_val = np.max(np.abs(x[curr_ex]))
            display_array[gap + j * (height + gap) + np.arange(height),
                          gap + i * (width + gap) + np.arange(width)[:, np.newaxis]] = \
                x[curr_ex].reshape((height, width)) / max_val
            curr_ex += 1
        if curr_ex > m:
            break
    plt.figure()
    plt.imshow(display_array, cmap='gray', extent=[-1, 1, -1, 1])
    plt.show()

# 随机抽取100个训练样本 进行可视化
m = y.size
rand_indices = np.random.permutation(range(m))  # 获取0-4999 5000个无序随机索引
selected = X[rand_indices[0:100], :]  # 获取前100个随机索引对应的整条数据的输入特征
print(selected.shape)
display(selected)
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)
print(y_onehot.shape)
#定义前向传播函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#定义前向传播函数
def forward_propagate(X, theta1, theta2):
    m = X.shape[0]

    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)

    return a1, z2, a2, z3, h
#计算sigmoid函数的梯度
def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

#定义前向传播与后向传播
def backprop(params, input_size, hidden_size, num_labels, X, y, l):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # 将参数数组重新塑造为每个层的参数矩阵
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # 运行前向传递
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # 赋初值
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)

    # 计算代价
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)

    J = J / m

    # 添加代价函数正则化项
    J += (float(l) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))

    # 执行反向传播
    for t in range(m):
        a1t = a1[t, :]  # (1, 401)
        z2t = z2[t, :]  # (1, 25)
        a2t = a2[t, :]  # (1, 26)
        ht = h[t, :]  # (1, 10)
        yt = y[t, :]  # (1, 10)

        d3t = ht - yt  # (1, 10)

        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)

        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    # 添加梯度正则化项
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * l) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * l) / m

    # 将剃度矩阵分解为数组
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return J, grad

# 初始化设置
input_size = 400
hidden_size = 25
num_labels = 10
l = 1

# 随机初始化完整网络参数大小的参数数组
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25

m = X.shape[0]
X = np.matrix(X)
y = np.matrix(y)

# 将参数数组解开为每个层的参数矩阵
theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

print(theta1.shape, theta2.shape)
a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
print(a1.shape, z2.shape, a2.shape, z3.shape, h.shape)
# print(cost(params, input_size, hidden_size, num_labels, X, y_onehot, l))
J, grad = backprop(params, input_size, hidden_size, num_labels, X, y_onehot, l)
print(J, grad.shape)

from scipy.optimize import minimize
#最小化目标函数
fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, l),
                method='TNC', jac=True, options={'maxiter' : 250})
print(fmin)

X = np.matrix(X)
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
y_pred = np.array(np.argmax(h, axis=1) + 1)
y_pred

correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))