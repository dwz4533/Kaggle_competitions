import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
from torch.utils import data

batch_size, lr, in_dim, out_dim = 32, 0.01, 64, 99

# 读取数据
train_raw_data = pd.read_csv('train.csv')
test_raw_data = pd.read_csv('test.csv')

# 得到数字和label之间的映射
r_train = 0.9
num_to_label = {}
label_to_num = {}

labels = np.unique(train_raw_data.iloc[:, 1])
for i in range(len(labels)):
    num_to_label[str(i)] = labels[i]
    label_to_num[labels[i]] = i

# 划分数据集
n_train = int(len(train_raw_data) * r_train)
train_dataset = torch.tensor(train_raw_data.iloc[:n_train, 2:].values, dtype=torch.float32)
train_labels = torch.tensor([label_to_num[label] for label in train_raw_data.iloc[:n_train, 1].values], dtype=torch.long)
train_dataset = data.TensorDataset(train_dataset, train_labels)

eval_dataset = torch.tensor(train_raw_data.iloc[n_train:, 2:].values, dtype=torch.float32)
eval_labels = torch.tensor([label_to_num[label] for label in train_raw_data.iloc[n_train:, 1].values], dtype=torch.long)
eval_dataset = data.TensorDataset(eval_dataset, eval_labels)

test_dataset = torch.tensor(test_raw_data.iloc[:, 1:].values, dtype=torch.float32)
test_idx = torch.tensor(test_raw_data.iloc[:, 0].values, dtype=torch.int32)
test_dataset = data.TensorDataset(test_dataset, test_idx)

# 设定iteration
train_iter = data.DataLoader(
    dataset=train_dataset, shuffle=True, batch_size=batch_size
)
eval_iter = data.DataLoader(
    dataset=eval_dataset, shuffle=False, batch_size=batch_size
)
test_iter = data.DataLoader(
    dataset=test_dataset, shuffle=False, batch_size=batch_size
)

# 定义网络
class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.f1 = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
        )
        self.f2 = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
        )
        self.f3 = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
        )
        self.fusion = nn.Linear(3 * out_dim, out_dim)

    def forward(self, X):
        X1, X2, X3 = torch.split(X, 64, dim=1)
        y1 = self.f1(X1)
        y2 = self.f2(X2)
        y3 = self.f3(X3)
        Y = torch.cat([y1, y2, y3], dim=1)
        return self.fusion(Y)
net = Net(in_dim, out_dim)

# 定义损失函数和优化器
loss = nn.CrossEntropyLoss(reduction='mean')
trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)

# 训练
epochs = 80
# for epoch in range(epochs):
#     net.train()
#     total_samples = 0
#     total_acc = 0
#     total_loss = 0
#
#     for X, Y in train_iter:
#         y_hat = net(X)
#         trainer.zero_grad()
#         l = loss(y_hat, Y)
#         l.backward()
#         trainer.step()
#
#         ac_batch = Y.size(0)
#         correct = (y_hat.argmax(axis=1) == Y).sum().item()
#         total_samples += ac_batch
#         total_acc += correct
#         total_loss += l
#
#     if epoch == 0 or epoch % 10 == 0:
#         train_loss = total_loss / total_samples
#         train_acc = total_acc / total_samples
#         print(f'epoch {epoch}th. train_loss is {train_loss}, train_acc is {train_acc}')
#
# net.eval()
# total_samples = 0
# total_acc = 0
# for X, Y in eval_iter:
#     y_hat = net(X)
#     total_acc += (y_hat.argmax(axis=1) == Y).sum().item()
#     total_samples += Y.size(0)
# print(f'eval acc is {total_acc / total_samples}')
#
# torch.save(net.state_dict(), 'model_params.pth')

state_dict = torch.load('model_params.pth')
net.load_state_dict(state_dict)

results = []

net.eval()
with torch.no_grad():
    for X, id_batch in test_iter:  # id_batch 形状 (batch_size,)
        # 模型输出 logits，转为概率 (batch_size, 99)
        probs = F.softmax(net(X), dim=1)  # 显式指定 dim=1，消除警告

        # 逐个样本处理
        for idx, prob in zip(id_batch, probs):
            # idx 是标量或0维张量，转为 Python 整数
            row = [int(idx.item())] + prob.cpu().numpy().tolist()
            results.append(row)

# 构建 DataFrame，列名：id + 类别名
columns = ['id'] + list(label_to_num.keys())
df = pd.DataFrame(results, columns=columns)
df.to_csv('submission.csv', index=False)

# df.to_csv('output.csv', index=False)