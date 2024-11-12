import numpy as np
import matplotlib.pyplot as plot



#梯度下降法

X = [0.01 * x for x in range(100)]
Y = [2*x**2 + 3*x + 4 for x in X]

#选定模型结构
def func(x):
    y = w1*x**2 + w2*x + w3
    return y

#计算损失函数
def loss(y_pred, y_true):
    return (y_pred - y_true)**2

#初始化参数
w1, w2, w3 = 1, 0, -1

#设定学习率 
lr = 0.1    

#开始训练，1000轮
for epoch in range(10000):
    batch_size = 20
    grad_w1 = 0
    grad_w2 = 0
    grad_w3 = 0
    counter = 0
    epoch_loss = 0 #每一轮的loss
    for x, y_true in zip(X,Y):
        #计算预测值
        counter += 1
        y_pred= func(x)
        #计算loss
        epoch_loss += loss(y_pred, y_true)
        #计算梯度,求导链式法则
        grad_w1 += 2*(y_pred - y_true)*x**2
        grad_w2 += 2*(y_pred - y_true) * x
        grad_w3 += 2*(y_pred - y_true) 
        
        #更新梯度
        if(counter == batch_size):
            w1 = w1 - lr*grad_w1/batch_size
            w2 = w2 - lr*grad_w2/batch_size
            w3 = w3 - lr*grad_w3/batch_size
            counter = 0
            grad_w1 = 0
            grad_w2 = 0
            grad_w3 = 0
    epoch_loss /= len(X)
    # plot.scatter(epoch,epoch_loss)
    # plot.show()
    print("第%d轮的损失值为：%f" %(epoch,epoch_loss))
    if(epoch_loss <= 0.0001):
        break

#训练后的参数值为：
print(f"训练后的参数值，w1:{w1},w2:{w2},w3:{w3}")
#最终训练的函数输出的值
Y_p = [func(i)for i in X]
#最终训练出来的图像
plot.scatter(X,Y_p,color = "red")
plot.scatter(X,Y,color = "yellow")
plot.show()
