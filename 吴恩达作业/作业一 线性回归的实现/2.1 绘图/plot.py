import pandas as pd
import matplotlib.pyplot as plt
path = 'C:\\Users\\Administrator\\Desktop\\吴恩达作业\\作业一 线性回归的实现\\题目\\ex1\\ex1data1.txt'
data = pd.read_csv(path,header = 0, names = ['Population','Profit'])
print(data)
data.plot(x = 'Population',y = 'Profit',kind = 'scatter',figsize = (10,8))
plt.show()