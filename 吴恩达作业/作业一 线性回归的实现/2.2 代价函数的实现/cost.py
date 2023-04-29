import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def computeCost(x,y,theta):
    total = np.power(((x*theta.T)-y),2) #power指的是求其平方
    return np.sum(total)/(2*len(x))
path='C:\\Users\\Administrator\\Desktop\\吴恩达作业\\作业一 线性回归的实现\\题目\\ex1\\ex1data1.txt'
data = pd.read_csv(path,header = 0, names = ['Population','Profit'])
data.insert(0,'Ones',1)#insert函数的第一个为在第0个位置插入，第二个为插入的对象，格式如上所示，第二个为头名，第三个为数字。此处添加偏置。
cols = data.shape[1]#读取矩阵的列数，shape[0]就是读取行数
X = data.iloc[:,0:cols-1]#读取第0-倒数第二列，iloc[ : , : ]前面取函数，后面取列数，最后一列不取，左闭右开
Y = data.iloc[:,cols-1:cols]#读取倒数第一列。
X = np.matrix(X.values)#设置为矩阵,value返回其具体值（去掉头名）
Y = np.matrix(Y.values)
theta = np.matrix(np.array([0,0]))#不知道里面套个array有什么用，为0是因为题目要求
print(computeCost(X,Y,theta))