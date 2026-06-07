import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
import pandas as pd
import numpy as np

batch_size, lr, in_dim, out_dim = 32, 0.01, 192, 99

label_to_num = {}
num_to_label = {}

# 读取文件信息
train_raw_data = pd.read_csv('train.csv')
test_raw_data = pd.read_csv('test.csv')


r_train = 0.9
labels = np.unique(train_raw_data.iloc[:, 1].values)
for i in range(len(labels)):
    num_to_label[str(i)] = labels[i]
    label_to_num[labels[i]] = i

# 划分训练集和验证集
train_features = torch.tensor(train_raw_data.iloc[:int(r_train * len(train_raw_data)), 2:].values, dtype=torch.float32)
train_labels = torch.tensor([label_to_num[label] for label in train_raw_data.iloc[:int(r_train * len(train_raw_data)), 1].values], dtype=torch.long)

eval_features = torch.tensor(train_raw_data.iloc[int(r_train * len(train_raw_data)):, 2:].values, dtype=torch.float32)
eval_labels = torch.tensor([label_to_num[label] for label in train_raw_data.iloc[int(r_train * len(train_raw_data)):, 1].values], dtype=torch.long)

print(len(train_labels), len(eval_labels))

test_features = torch.tensor(test_raw_data.iloc[:, 1:].values, dtype=torch.float32)
test_id = torch.tensor(test_raw_data.iloc[:, 0].values, dtype=torch.int32)

train_dataset = data.TensorDataset(train_features, train_labels)
test_dataset = data.TensorDataset(test_features, test_id)
eval_dataset = data.TensorDataset(eval_features, eval_labels)

# 制作数据批次迭代器
train_iter = data.DataLoader(
    dataset=train_dataset, batch_size=batch_size ,shuffle=True 
)
test_iter = data.DataLoader(
    dataset=test_dataset, batch_size=batch_size ,shuffle=False 
)
eval_iter = data.DataLoader(
    dataset=eval_dataset, batch_size=batch_size ,shuffle=False 
)

# 定义网络
# net = nn.Sequential(
#     nn.Linear(in_dim, 1024), nn.BatchNorm1d(1024),  nn.ReLU(), nn.Dropout(0.1), 
#     nn.Linear(1024, 512), nn.BatchNorm1d(512),  nn.ReLU(),  nn.Dropout(0.2),
#     nn.Linear(512, 256), nn.BatchNorm1d(256),  nn.ReLU(), nn.Dropout(0.3),
#     nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),  
#     nn.Linear(128, 100), nn.BatchNorm1d(100), nn.ReLU(), 
#     nn.Linear(100, out_dim)
# )

class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.f1 = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.BatchNorm1d(1024),  nn.ReLU(), 
            nn.Linear(1024, 512), nn.BatchNorm1d(512),  nn.ReLU(),
            nn.Linear(512, out_dim)
        )
        
        self.f2 = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512),  nn.ReLU(), 
            nn.Linear(512, 256), nn.BatchNorm1d(256),  nn.ReLU(),
            nn.Linear(256, out_dim)
        )
        
        self.f3 = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512),  nn.ReLU(), 
            nn.Linear(512, 256), nn.BatchNorm1d(256),  nn.ReLU(),
            nn.Linear(256, out_dim)
        )
        self.fusion = nn.Linear(3 * out_dim, out_dim)
        
    def forward(self, X):
        X1 = X[:, :64]
        X2 = X[:, 64:128]
        X3 = X[:, 128:]
        y1 = self.f1(X1)
        y2 = self.f2(X2)
        y3 = self.f3(X3)
        combined = torch.cat([y1, y2, y3], axis=1)
        # Y = y1 + y2 + y3
        Y = self.fusion(combined)
        return Y
net = Net(in_dim=64, out_dim=99)
      
# 定义损失函数和训练器
trainer = torch.optim.SGD(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction='mean')

# 开始训练
epochs = 80
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net.to(device)
for epoch in range(epochs):
    net.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for X, Y in train_iter:
        X, Y = X.to(device), Y.to(device)
        y_hat = net(X)
        trainer.zero_grad()
        l = loss(y_hat, Y)        
        l.backward()
        trainer.step()
        
        # 累计正确数和样本数
        batch_correct = (y_hat.argmax(dim=1) == Y).sum().item()
        batch_size_actual = Y.size(0)

        total_correct += batch_correct
        total_samples += batch_size_actual
        total_loss += l.item() * batch_size_actual   # 恢复该 batch 的总损失

    epoch_loss_avg = total_loss / total_samples   # 平均 loss
    epoch_acc = total_correct / total_samples     # 准确率

    if epoch == 0 or epoch % 10 == 0:
        print(f'epoch {epoch} training loss: {epoch_loss_avg:.6f}')
        print(f'training acc: {epoch_acc:.4f}')

total_correct = 0
total_samples = 0
for X, Y in eval_iter:
    net.eval()
    y_hat = net(X)
    total_correct += (y_hat.argmax(dim=1) == Y).sum().item()
    batch_size = Y.size(0)
    
    total_samples += batch_size
    
print('eval acc is ', total_correct / total_samples)

torch.save(net.state_dict(), 'test_new.pth')

net.to(device)
# state_dict = torch.load('test.pth', map_location=device)
# net.load_state_dict(state_dict)

data_rows = []
with torch.no_grad():
    for X, id_ in test_iter:
        y_hat = F.softmax(net(X)).numpy()          # (batch, 99) float
        id_np = id_.reshape(-1, 1).numpy().astype(int)  # (batch, 1) int
        # 逐行组合
        for i in range(id_np.shape[0]):
            row = [int(id_np[i, 0])] + y_hat[i].tolist()
            data_rows.append(row)

columns = ['id'] + list(label_to_num.keys())
df = pd.DataFrame(data_rows, columns=columns)
df.to_csv('submission.csv', index=False)   # id 列会保存为整数

    
