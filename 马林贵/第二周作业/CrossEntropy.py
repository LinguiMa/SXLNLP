import torch.nn as nn
import numpy as np
import torch



#手动实现交叉熵


#预测的分类
pred = np.array([[1.1,2,3],
          [2,3,4],
          [2,5,7]])

#模型的交叉熵函数
ce_loss = nn.CrossEntropyLoss()

#目标分类
target = np.array([1,2,0])

#交叉熵算出的损失值：
cross_loss = ce_loss(torch.FloatTensor(pred),torch.LongTensor(target))
print("torch模型中的交叉熵损失值：",cross_loss)


#自定义实现交叉熵

#softmax函数，归一化预测值
def softmax(pred):
    pred = np.exp(pred)/np.sum(np.exp(pred),axis=1,keepdims=True)
    return pred


#根据目标分类创建目标矩阵
def to_one_hot(target,shape):
    t = np.zeros(shape)
    for i,j in enumerate(target):
        t[i][j] = 1
    
    return t

#实现交叉熵
def CrossEntropy(pred,target):
    pred = softmax(pred)
    target = to_one_hot(target,pred.shape)
    batch_size,class_num = pred.shape
    loss_1 = -np.sum(target * np.log(pred),axis = 1)
    loss_cross = sum(loss_1)/batch_size
    return loss_cross


print("自定义softmax函数的loss值为：",CrossEntropy(pred,target))