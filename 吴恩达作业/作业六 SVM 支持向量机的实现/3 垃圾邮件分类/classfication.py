import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
train_data = loadmat("C:\\Users\\Administrator\\Desktop\\吴恩达作业\\作业六 SVM 支持向量机的实现\\题目\\ex6\\spamTrain.mat")
train_X = train_data['X']  # (4000,1899)
train_y = train_data['y']  # (4000,1)
test_data = loadmat("C:\\Users\\Administrator\\Desktop\\吴恩达作业\\作业六 SVM 支持向量机的实现\\题目\\ex6\\spamTest.mat")
test_X = test_data['Xtest']  # (1000,1899)
test_y = test_data['ytest']  # (1000,1)
model = svm.SVC(kernel='linear')  # 这里的n比较大，选用线性核函数效果好
model.fit(train_X, train_y.ravel())
print(model.score(test_X, test_y.ravel()))#返回模型的评估得分，也就是准确度