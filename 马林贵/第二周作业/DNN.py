#用pytorch框架和自己搭建的两层模型进行对比
import torch
import torch.nn as nn
import numpy as np



#搭建两层模型 输入为

class TorchModel(nn.Module):
    def __init__(self, put_size, hidden_size1,hidden_size2):
        super(TorchModel,self).__init__()
        self.layer1 = nn.Linear(put_size,hidden_size1) # 1*3 3*5
        self.layer2 = nn.Linear(hidden_size1,hidden_size2)
    def forward(self,x):
        x = self.layer1(x)
        y = self.layer2(x)
        return y



#自己构建模型

class DiyModel():
    def __init__(self,w1,w2,b1,b2) -> None:
        self.w1 = w1
        self.w2 = w2
        self.b1 = b1
        self.b2 = b2
    
    def forward(self,x):
        x1 = np.dot(x,self.w1.T) + self.b1
        y_pred = np.dot(x1,self.w2.T)+self.b2
        return y_pred
    
input_x = np.array([[2,3,4],
                    [4,5,6]])

input_size = 3
hidden1_size = 3
hidden2_size = 5                    
torchModel = TorchModel(input_size,hidden1_size,hidden2_size)
#看参数
print(torchModel.state_dict())

w1 = torchModel.state_dict()["layer1.weight"].numpy()
w2 = torchModel.state_dict()["layer2.weight"].numpy()
b1 = torchModel.state_dict()["layer1.bias"].numpy()
b2 = torchModel.state_dict()["layer2.bias"].numpy()
 
tensor_input = torch.FloatTensor(input_x)
torch_pred = torchModel.forward(tensor_input)
print("Torch模型的输出为：",torch_pred)

#构造自定义模型
my_Model = DiyModel(w1,w2,b1,b2)
diy_pred = my_Model.forward(input_x)
print("my_Model的预测输出为：",diy_pred)