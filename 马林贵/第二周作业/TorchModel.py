import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt


#建立一个模型，进行五分类，输入一个五维向量，最大值的下标为几，就为第几类

#构建模型

class TorchModel(nn.Module):
    def __init__(self,input_size,hidden1_size):
        super(TorchModel,self).__init__()
        self.layer = nn.Linear(input_size,hidden1_size)
        self.loss = nn.functional.cross_entropy
    def forward(self,x,y_act=None):
        y_pred = self.layer(x)
        if y_act is  None:
            return y_pred
        else:
            return self.loss(y_pred,y_act)


#构建训练样本:
def build_sample():
    #输入数据
    x = np.random.random(5)
    i = np.argmax(x)
    return x,i

#构建数据集：
def build_dataset(size):
    X = []
    Y = []
    for i in range(size):
        x,y = build_sample()
        #print("x为：\n",x)
        #print("y为：\n",y)
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X),torch.LongTensor(Y)        

#训练模型
def train(model):
    #模型处于训练模式
   
    #优化器的选择
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    #训练集的个数：5000
    train_num = 5000
    x,y_act = build_dataset(train_num)
    #batch_size:批次
    batch_size = 20
    #训练轮数
    epoch_num = 100
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        #每一轮需要训练的批次：train_num/batch_size
        for i in range(train_num//batch_size):
            x_in = x[i*batch_size : (i+1)*batch_size]
            y = y_act[i*batch_size : (i+1)*batch_size]
            #预测输出
            #y_pred = model(x)
            #计算loss,此时loss值应该为累加和，20个样本的累加和
            #print(x_in)
            #print(y)
            loss = model(x_in,y)
            watch_loss.append(loss.item())
            #反向传播,计算梯度，梯度也是20个样本的梯度和
            loss.backward()
            #梯度更新,按批次更新 w1 = w1 - lr *gradw1/batch_size
            optim.step()
            #梯度归零
            optim.zero_grad()
        print("第%d轮训练的损失值为：%f"%(epoch+1,np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

#评估模型
def evaluate(model):
    #测试模型
    model.eval()
    #测试集100个
    test_x,test_y = build_dataset(100)
    pred_y = model(test_x)
    correct = 0
    wrong = 0
    with torch.no_grad():
        for y_p,y_t in zip(pred_y,test_y):
            if(torch.argmax(y_p) == int(y_t)):
                correct += 1
            else:
                wrong += 1
    print("测试集的准确率为：%f"%(correct/(correct+wrong)))
    return correct/(correct+wrong)

#利用训练好的模型做预测：
def predict(model_path,vec):
    model = TorchModel(5,5)
    #加载训练好的权重
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())
    model.eval()
    #用于测试
    with torch.no_grad():
        y_pred = model.forward(torch.FloatTensor(vec))
    for vec_t, y_t in zip(vec,y_pred):
        print("预测的数据为%s,预测的类别为：%s"%(vec_t,y_t))



if __name__ == "__main__":
    #模型初始化
    #model = TorchModel(5,5)
    #训练模型
    #train(model)
    #预测模型
    vec,y = build_dataset(5)
    print("要预测的数据为：%s"%(vec))
    predict('model.pt',vec)
