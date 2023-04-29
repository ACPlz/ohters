import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def computeCost(x,y,theta):#计算代价函数
    total = np.power(((x*theta.T)-y),2) #power指的是求其平方
    return np.sum(total)/(2*len(x))
def gradientDescent(x,y,theta,alpha,iters):#alpha指的是学习速率，而iters指的是迭代次数
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])#ravel函数将矩阵降为一维，int函数将其转化为某个进制，为填写默认为十进制
    cost = np.zeros(iters)

    for i in range(iters):
        error = (x*theta.T) - y
        for j in range(parameters):
            term = np.multiply(error,x[:,j])#np.multiply为矩阵的对应位置相乘
            temp[0,j] = theta[0,j] - ((alpha/len(x))*np.sum(term))#此处没有1/2是因为已经偏导了
            theta = temp
            cost[i]=computeCost(x,y,theta)
    return theta,cost
def normalEquation(x,y):#正规方程法来选择多项式进行拟合，事实上用归一化更好？
    result = np.linalg.inv(x.T@x)@x.T@y
    return result
path = 'C:\\Users\\Administrator\\Desktop\\吴恩达作业\\作业一 线性回归的实现\\题目\\ex1\\ex1data2.txt'
data = pd.read_csv(path,header=None,names = ['Size','Bedroom','Price'])
data = (data - data.mean())/data.std()#特征值标准化！！！
print(data.head())
data.insert(0,'x0',1)
cols = data.shape[1]
print(cols)
X = data.iloc[:,0:cols-1]
Y = data.iloc[:,cols-1:cols]
X = np.matrix(X.values)
Y = np.matrix(Y.values)
theta = np.matrix(np.zeros(data.shape[1]-1))#多变量回归的θ变成其特征数量了
alpha = 0.01
iter = 1000
g, cost = gradientDescent(X,Y,theta,alpha,iter)
#x1 = np.linspace(data.size.min(),data.size.max(),100)
#x2 = np.linspace(data.Bedroom.min(),data.Bedroom.max(),100)
x1 = 1.257476
x2 = 1.090417
f = g[0,0]+g[0,1]*x1+g[0,2]*x2#实际上计算g的shape然后乘一个向量x应该就搞定了
print(f)
#theta = normalEquation(X,Y)#使用归一化并且梯度下降后不需要用到正规方程了
print(g)
fig, ax = plt.subplots(figsize=(10,8))
ax.plot(np.arange(iter),cost,'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Coat')
ax.set_title('Error vs. Training Epoch')
plt.show()