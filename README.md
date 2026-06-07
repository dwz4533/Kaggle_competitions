# Leaf Classification – Kaggle Competition

基于 PyTorch 的多分支神经网络，利用树叶的 margin、shape、texture 特征进行分类。

## 项目结构
.<br>
├── train.csv # 训练集<br>
├── test.csv # 测试集<br>
├── sample_submission.csv # 提交样例<br>
├── model.py # 训练与预测脚本<br>
├── submission.csv # 最终的输出
└── README.md<br>

## 数据说明

- 特征：192 维，分为 margin、shape、texture 三类，每类 64 维
- 标签：99 种树叶物种
- 评估指标：多分类对数损失（Multi-class Log Loss）

## 模型架构

三分支结构，分别处理三类特征，最后通过一层线性层进行融合。

## 环境依赖

```
pip install torch pandas numpy
```
## 使用方法
训练模型
```
python model.py --mode train
```
预测并生成提交文件
```
python model.py --mode predict --model_path best_model.pth
```
输出 submission.csv。

|训练参数名|实值|
| --- | --- |
|损失函数|CrossEntropyLoss|
|优化器|SGD (lr=0.01)|
|批大小|32|
|轮数|80|
## 结果
验证集准确率：~99.8%<br>
验证集对数损失：<0.05

## License
MIT
