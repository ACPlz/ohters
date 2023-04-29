import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
path='C:\\Users\\Administrator\\Desktop\\吴恩达作业\\作业一 线性回归的实现\\题目\\ex1\\ex1data1.txt'
data = pd.read_csv(path,header = 0, names = ['Population','Profit'])
data.insert(0,'Ones',1)#insert函数的第一个为在第0个位置插入，第二个为插入的对象，格式如上所示，第二个为头名，第三个为数字。此处添加偏置。
cols = data.shape[1]#读取矩阵的列数，shape[0]就是读取行数
X = data.iloc[:,0:cols-1]#读取第0-倒数第二列，iloc[ : , : ]前面取行数，后面取列数，最后一列不取，左闭右开
Y = data.iloc[:,cols-1:cols]#读取倒数第一列。
X = np.matrix(X.values)#设置为矩阵,value返回其具体值（去掉头名）
Y = np.matrix(Y.values)
theta = np.matrix(np.array([0,0]))#不知道里面套个array有什么用，为0是因为题目要求
alpha = 0.01
iters = 1000
g,cost = gradientDescent(X,Y,theta,alpha,iters)
x = np.linspace(data.Population.min(),data.Population.max(),100)
f = g[0,0]+g[0,1]*x#其实就是第0行第0列作为常数项、第0行第一列作为一次方系数
fig, ax = plt.subplots(figsize=(10,8))
ax.plot(x,f,'r',label='Prediction')
ax.scatter(data.Population,data.Profit,label='Training Data')
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predict Profit vs. Population Size')
plt.show()
theta0_vals = np.linspace(-10,20,20)
theta1_vals = np.linspace(-1,4,20)
J_vals = np.zeros((theta0_vals.shape[0],theta1_vals.shape[0]))
for i in range(theta0_vals.shape[0]):
    for j in range(theta1_vals.shape[0]):
        t = np.matrix([theta0_vals[i],theta1_vals[j]])#此处将array改成matrix就不会报错了，具体为啥没搞懂
        J_vals[i,j]=computeCost(X,Y,t)
fig = plt.figure()
ax = Axes3D(fig)
theta0_vals,theta1_vals = np.meshgrid(theta0_vals,theta1_vals)
plt.title('Visualizing Graph')
ax.plot_surface(theta0_vals,theta1_vals,J_vals,cmap='rainbow')
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('J')
plt.show()
plt.contourf(theta0_vals,theta1_vals,J_vals,10,cmap='rainbow')
C = plt.contour(theta0_vals,theta1_vals,J_vals,10,colors = 'black')
plt.clabel(C,inline=True,fontsize=10)
plt.plot(g[0,0], g[0,1], c='r', marker="x")
plt.show()