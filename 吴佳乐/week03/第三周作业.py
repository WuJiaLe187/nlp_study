import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

"""
    尝试在nlpdemo中使用rnn模型训练
    判断特定字符在文本中的位置。
"""

class TorchRNN(nn.Module):
    """
    vector_dim: 向量维度
    sentence_length: 语句长度
    vocab: 词表
    """
    def __init__(self,vector_dim,vocab,hidden_size,sentence_len):
        super(TorchRNN,self).__init__()

        # 保存参数以便forward中使用
        self.hidden_size = hidden_size
        self.sentence_len = sentence_len

        # Embedding 层
        self.Embedding = nn.Embedding(len(vocab),vector_dim,padding_idx=0)
        # 池化层
        # self.pool = nn.AvgPool1d(sentence_len)
        # 在RNN加池化层RNN就没有意义
        # RNN层
        self.rnn_layer = nn.RNN(vector_dim, hidden_size,bias=False,batch_first=True)
        # 线性层
        self.linear = nn.Linear(hidden_size,sentence_len)
        # softmax
        #self.activation = torch.softmax
        # loss函数交叉熵
        self.loss = nn.functional.cross_entropy


    def forward(self,x,y = None):
        # 向量化
        # [bathc_size,sentence_len] -> [batch_size,sentence_len,vector_dim]
        x = self.Embedding(x)

        #x = x.transpose(1,2)
        # 池化
        #x = self.pool(x)
        # 去维
        #x = x.squeeze()

        #ht = np.zeros(self.hidden_size)

        # ht = tanh(b+Ux + Wh)
        # rnn_output记录每次运算后的结果,h_n记录最后的结果
        # rnn_output [batch_size,sentence_len,hidden_size]
        # hn ：[1,batch_size,hidden_size]
        rnn_output,h_n = self.rnn_layer(x)

        h_n = h_n.squeeze(0)
        # print("==============")
        # print(rnn_output)
        # print("==============")
        # print(h_n)
        # print("==============")
        y_pred = self.linear(h_n)

        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred

def build_vocab():
    chars = "吴佳乐是帅哥abcdef"
    vocab = {"pad":0}
    for index,char in enumerate(chars):
        vocab[char] = index + 1
    vocab["unk"] = len(vocab)
    return vocab

def build_sample(vocab,sentence_len):
    while True:
       x = [random.choice(list(vocab.keys())) for _ in range(sentence_len)]
       # 假设我找 a
       if "a" in x:
          y = x.index("a")
          break
    return x,y

def build_dataSet(vocab,sentence_len,batch_size):
    dataset_x = []
    dataset_y = []

    for i in range(batch_size):
        x,y = build_sample(vocab,sentence_len)
        x = [vocab.get(word,"unk") for word in x]
        dataset_x.append(x)
        dataset_y.append(y)

    return torch.LongTensor(dataset_x),torch.LongTensor(dataset_y)


def evalue(model,vocab,sentence_len,sample_num=200):
    #开启 dropout层
    model.eval()
    correct = 0
    with torch.no_grad():
        x, y = build_dataSet(vocab, sentence_len, sample_num)
        y_pred = model(x)
        pred_idx = torch.argmax(y_pred, dim=1)  # 取概率最大的位置
        correct = (pred_idx == y).sum().item()
    acc = correct / sample_num
    print(f"准确率：{acc:.2%}")
    return acc


def main():
    # 超参数
    epoch_num = 30
    sample_num = 1000
    batch_size = 50
    sentence_len = 5
    vector_dim = 10
    hidden_size = 16
    lr = 0.005

    vocab = build_vocab()
    vocab_size = len(vocab)

    model = TorchRNN(vector_dim, vocab, hidden_size, sentence_len)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    print("开始训练……\n")
    for epoch in range(epoch_num):
        model.train()
        loss_list = []

        for _ in range(sample_num // batch_size):
            x, y = build_dataSet(vocab, sentence_len, batch_size)
            optim.zero_grad()  # 必须加括号
            loss = model(x, y)
            loss.backward()
            optim.step()
            loss_list.append(loss.item())  # 必须加括号

        avg_loss = np.mean(loss_list)
        print(f"第{epoch+1:2d}轮  平均loss：{avg_loss:.4f}", end="  ")
        evalue(model, vocab, sentence_len)



if __name__ == "__main__":
    main()


