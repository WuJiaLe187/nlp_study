import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self,input_size=5):
        super(TorchModel,self).__init__()
        self.linear = nn.Linear(input_size,5)        #返回的应该是5个，每个位置的概率
        # 也就是 假设我们的batch_size 为1  1*5 的阵 乘以 5*5矩阵转置  还是 1*5 [0.1,0.2,0.3,0.4]
        self.loss = nn.functional.cross_entropy

    def forward(self,x,y=None):
        x = self.linear(x)
        #y_pred = self.activation(x,dim=1)
        # 内部会自动实现
        if y is not None:
            return self.loss(x,y)
        # 这个怎么实现的呢，    x已经是经过矩阵运算后的了，然后就是进行softmax() 然后就是根据真实的y进行onehot转换，然后它来进行相乘返回的是模型预测为正确类别的概率
        # axis=1 求和得到正确类别的概率，然后对这个概率取负对数，返回的才是损失函数
        # 为啥取反，-log(x)的图像，是当我们的概率越接近1，我们的损失函数越小
        else:
            return torch.softmax(x,dim=1)
#def softmax(matrix):
 #   return np.exp(matrix)/np.sum(np.exp(matrix),axis=1,keepdims = True)
#else里面跟这个差不多


def create_data():
    x = np.random.random(5)  #生成5个随机数  ，python模块生成一个随机数
    # print(len(x))
    for i in range(len(x)):
        if x[i] == max(x):
            return x,i


def create_dataList(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x,y = create_data()
        X.append(x)
        Y.append(y)

    return torch.FloatTensor(X),torch.LongTensor(Y)     #y必须是整形，因为它相当于一个索引，就是告诉我们第几个位置是最大的
#

# X,Y=create_dataList(5)
# print(X)
# print("-------------------------")
# print(Y)

def evaluate(model):
    model.eval()
    total_sample_num = 100
    x,y = create_dataList(total_sample_num)
    correct,wrong = 0,0
    with torch.no_grad():
        y_pred = model(x)
        for y_p,y_t in zip(y_pred,y):
            if y_p.argmax() == y_t:
                correct += 1
            else:
                wrong += 1
    print(f"正确预测个数{correct},正确率{correct / (correct + wrong)}")
    return correct / (correct + wrong)

def main():
    input_size = 5
    total_sample_num = 5000
    epoch_num = 50
    batch_size = 20
    learning_rate = 0.05

    #模型初始化
    model = TorchModel(input_size)  #这个必须放在优化器上面，因为优化器中要使用
    log = []

    #选择优化器
    optim = torch.optim.Adam(model.parameters(),lr = learning_rate)

    # #模型初始化
    # model = TorchModel(input_size)

    for epoch in range(epoch_num):
        model.train()

        #生成数据集
        X,Y = create_dataList(total_sample_num)
        watch_loss = []
        for batch_index in range(total_sample_num//batch_size):
            x = X[batch_index*batch_size:(batch_index+1)*batch_size]
            y = Y[batch_index*batch_size:(batch_index+1)*batch_size]
            loss = model(x,y)
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            optim.zero_grad()    #梯度清零
            watch_loss.append(loss.item())
        acc = evaluate(model)
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        log.append([acc,float(np.mean(watch_loss))])
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

if __name__=="__main__":
    main()
