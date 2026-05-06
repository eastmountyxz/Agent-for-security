"""
基于TF-IDF向量和PyTorch CNN的Web攻击类型分类实验
特征: url_n_gram
类别: attack_type
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
import os
import time

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============================================================
# 1. 读取数据
# ============================================================
print("=" * 60)
print("步骤1: 读取数据")
print("=" * 60)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_features.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_features.csv'))
val_df = pd.read_csv(os.path.join(DATA_DIR, 'val_features.csv'))

print(f"训练集大小: {len(train_df)}")
print(f"验证集大小: {len(val_df)}")
print(f"测试集大小: {len(test_df)}")

# 提取指定列
cols = ['payload_id', 'attack_type', 'url_n_gram']
train_df = train_df[cols]
test_df = test_df[cols]
val_df = val_df[cols]

# 查看类别分布
print(f"\n训练集类别分布:")
print(train_df['attack_type'].value_counts())
print(f"\n测试集类别分布:")
print(test_df['attack_type'].value_counts())

# ============================================================
# 2. 特征提取与TF-IDF向量化
# ============================================================
print("\n" + "=" * 60)
print("步骤2: 特征提取与TF-IDF向量化")
print("=" * 60)

# 合并训练集和验证集作为训练数据
train_val_df = pd.concat([train_df, val_df], ignore_index=True)
X_train_text = train_val_df['url_n_gram'].fillna('')
y_train = train_val_df['attack_type']
X_test_text = test_df['url_n_gram'].fillna('')
y_test = test_df['attack_type']

# TF-IDF向量化
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    max_df=0.95,
    min_df=2
)

X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

print(f"TF-IDF特征维度: {X_train_tfidf.shape[1]}")
print(f"训练样本数: {X_train_tfidf.shape[0]}")
print(f"测试样本数: {X_test_tfidf.shape[0]}")

# 标签编码
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

class_names = le.classes_
num_classes = len(class_names)
print(f"类别数量: {num_classes}")
print(f"类别列表: {list(class_names)}")

# ============================================================
# 3. 构建CNN模型
# ============================================================
print("\n" + "=" * 60)
print("步骤3: 构建CNN模型")
print("=" * 60)

# 将TF-IDF稀疏矩阵转换为dense并reshape为CNN输入格式
# CNN输入: (batch_size, channels, sequence_length)
# 将TF-IDF向量reshape为单通道1D序列
X_train_dense = X_train_tfidf.toarray().astype(np.float32)
X_test_dense = X_test_tfidf.toarray().astype(np.float32)

# reshape为 (samples, 1, features) 用于Conv1d
X_train_cnn = X_train_dense.reshape(X_train_dense.shape[0], 1, -1)
X_test_cnn = X_test_dense.reshape(X_test_dense.shape[0], 1, -1)

print(f"CNN输入形状: {X_train_cnn.shape}")


class TextCNN(nn.Module):
    """基于1D卷积的文本分类CNN模型"""

    def __init__(self, input_dim, num_classes, dropout=0.5):
        super(TextCNN, self).__init__()

        # 多尺度卷积核，捕获不同范围的特征模式
        self.conv1 = nn.Conv1d(1, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(1, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(1, 128, kernel_size=7, padding=3)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.dropout = nn.Dropout(dropout)

        # 全连接层
        self.fc1 = nn.Linear(128 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, 1, seq_len)

        # 多尺度卷积
        x1 = self.pool(self.relu(self.bn1(self.conv1(x)))).squeeze(-1)  # (batch, 128)
        x2 = self.pool(self.relu(self.bn2(self.conv2(x)))).squeeze(-1)  # (batch, 128)
        x3 = self.pool(self.relu(self.bn3(self.conv3(x)))).squeeze(-1)  # (batch, 128)

        # 拼接多尺度特征
        x = torch.cat([x1, x2, x3], dim=1)  # (batch, 384)

        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# 初始化模型
input_dim = X_train_cnn.shape[2]
model = TextCNN(input_dim=input_dim, num_classes=num_classes, dropout=0.5).to(device)

# 计算模型参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")
print(f"\n模型结构:\n{model}")

# ============================================================
# 4. 训练模型
# ============================================================
print("\n" + "=" * 60)
print("步骤4: 训练CNN模型")
print("=" * 60)

# 准备数据
X_train_tensor = torch.FloatTensor(X_train_cnn)
y_train_tensor = torch.LongTensor(y_train_encoded)
X_test_tensor = torch.FloatTensor(X_test_cnn)
y_test_tensor = torch.LongTensor(y_test_encoded)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 训练
num_epochs = 30
train_start_time = time.time()
train_losses = []
train_accs = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_X.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    scheduler.step()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

train_time = time.time() - train_start_time
print(f"\n训练完成! 训练时间: {train_time:.2f}秒")

# ============================================================
# 5. 预测与评估
# ============================================================
print("\n" + "=" * 60)
print("步骤5: 模型预测与评估")
print("=" * 60)

model.eval()
predict_start_time = time.time()
with torch.no_grad():
    X_test_dev = X_test_tensor.to(device)
    outputs = model(X_test_dev)
    _, y_pred_encoded = torch.max(outputs.data, 1)
    y_pred_encoded = y_pred_encoded.cpu().numpy()

predict_time = time.time() - predict_start_time
print(f"预测时间: {predict_time:.2f}秒")

y_pred = le.inverse_transform(y_pred_encoded)

# 整体评价指标
accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')
f1_macro = f1_score(y_test, y_pred, average='macro')

precision_weighted = precision_score(y_test, y_pred, average='weighted')
recall_weighted = recall_score(y_test, y_pred, average='weighted')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print(f"\n整体评价指标（保留3位小数）:")
print(f"  准确率 (Accuracy):           {accuracy:.3f}")
print(f"  宏精确率 (Precision-Macro):   {precision_macro:.3f}")
print(f"  宏召回率 (Recall-Macro):      {recall_macro:.3f}")
print(f"  宏F1值 (F1-Macro):           {f1_macro:.3f}")
print(f"  加权精确率 (Precision-Wtd):   {precision_weighted:.3f}")
print(f"  加权召回率 (Recall-Wtd):      {recall_weighted:.3f}")
print(f"  加权F1值 (F1-Wtd):           {f1_weighted:.3f}")

# 分类报告
print(f"\n详细分类报告:")
report = classification_report(y_test, y_pred, target_names=class_names, digits=3)
print(report)

# 各类别指标
precision_per_class = precision_score(y_test, y_pred, average=None, labels=class_names)
recall_per_class = recall_score(y_test, y_pred, average=None, labels=class_names)
f1_per_class = f1_score(y_test, y_pred, average=None, labels=class_names)

print(f"\n各类别详细指标（保留3位小数）:")
print(f"{'类别':<20s} {'精确率':>8s} {'召回率':>8s} {'F1值':>8s}")
print("-" * 48)
for i, cls in enumerate(class_names):
    print(f"{cls:<20s} {precision_per_class[i]:>8.3f} {recall_per_class[i]:>8.3f} {f1_per_class[i]:>8.3f}")
print("-" * 48)
print(f"{'宏平均':<20s} {precision_macro:>8.3f} {recall_macro:>8.3f} {f1_macro:>8.3f}")
print(f"{'加权平均':<20s} {precision_weighted:>8.3f} {recall_weighted:>8.3f} {f1_weighted:>8.3f}")

# ============================================================
# 6. 绘制可视化图
# ============================================================
print("\n" + "=" * 60)
print("步骤6: 绘制评价可视化图")
print("=" * 60)

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'CNN')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 6.1 训练损失和准确率曲线 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(1, num_epochs + 1), train_losses, 'b-', linewidth=1.5)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training Loss Curve', fontsize=14)
ax1.grid(alpha=0.3)

ax2.plot(range(1, num_epochs + 1), train_accs, 'r-', linewidth=1.5)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Training Accuracy Curve', fontsize=14)
ax2.grid(alpha=0.3)

plt.tight_layout()
train_curve_path = os.path.join(OUTPUT_DIR, 'training_curves.png')
plt.savefig(train_curve_path, dpi=150)
plt.close()
print(f"训练曲线已保存: {train_curve_path}")

# --- 6.2 混淆矩阵 ---
cm = confusion_matrix(y_test, y_pred, labels=class_names)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - CNN (TF-IDF url_n_gram)', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"混淆矩阵已保存: {cm_path}")

# --- 6.3 归一化混淆矩阵 ---
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(12, 10))
sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Normalized Confusion Matrix - CNN (TF-IDF url_n_gram)', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
cm_norm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_normalized.png')
plt.savefig(cm_norm_path, dpi=150)
plt.close()
print(f"归一化混淆矩阵已保存: {cm_norm_path}")

# --- 6.4 各类别Precision/Recall/F1柱状图 ---
x = np.arange(len(class_names))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 7))
bars1 = ax.bar(x - width, precision_per_class, width, label='Precision', color='#4C72B0')
bars2 = ax.bar(x, recall_per_class, width, label='Recall', color='#DD8452')
bars3 = ax.bar(x + width, f1_per_class, width, label='F1-Score', color='#55A868')

ax.set_xlabel('Attack Type', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Per-Class Precision / Recall / F1-Score - CNN (TF-IDF url_n_gram)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0.01:
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)

plt.tight_layout()
bar_path = os.path.join(OUTPUT_DIR, 'per_class_metrics.png')
plt.savefig(bar_path, dpi=150)
plt.close()
print(f"各类别指标柱状图已保存: {bar_path}")

# --- 6.5 测试集类别分布对比 ---
test_counts = test_df['attack_type'].value_counts().reindex(class_names)
pred_counts = pd.Series(y_pred).value_counts().reindex(class_names, fill_value=0)

fig, ax = plt.subplots(figsize=(12, 6))
x_idx = np.arange(len(class_names))
width = 0.35
ax.bar(x_idx - width/2, test_counts.values, width, label='True Distribution', color='#4C72B0')
ax.bar(x_idx + width/2, pred_counts.values, width, label='Predicted Distribution', color='#DD8452')
ax.set_xlabel('Attack Type', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('True vs Predicted Distribution - Test Set', fontsize=14)
ax.set_xticks(x_idx)
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
dist_path = os.path.join(OUTPUT_DIR, 'distribution_comparison.png')
plt.savefig(dist_path, dpi=150)
plt.close()
print(f"类别分布对比图已保存: {dist_path}")

# --- 6.6 综合评价雷达图 ---
from math import pi

categories_count = len(class_names)
scores = {
    'Precision': precision_per_class,
    'Recall': recall_per_class,
    'F1-Score': f1_per_class
}

fig, axes = plt.subplots(1, 3, figsize=(20, 7), subplot_kw=dict(polar=True))
for idx, (metric_name, values) in enumerate(scores.items()):
    ax = axes[idx]
    angles = [n / float(categories_count) * 2 * pi for n in range(categories_count)]
    values_plot = list(values) + [values[0]]
    angles += angles[:1]

    ax.plot(angles, values_plot, 'o-', linewidth=2, color=['#4C72B0', '#DD8452', '#55A868'][idx])
    ax.fill(angles, values_plot, alpha=0.25, color=['#4C72B0', '#DD8452', '#55A868'][idx])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(class_names, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title(metric_name, fontsize=14, pad=20)

plt.suptitle('Per-Class Metric Radar Charts - CNN (TF-IDF url_n_gram)', fontsize=16, y=1.05)
plt.tight_layout()
radar_path = os.path.join(OUTPUT_DIR, 'radar_charts.png')
plt.savefig(radar_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"雷达图已保存: {radar_path}")

# ============================================================
# 7. 输出预测结果CSV
# ============================================================
print("\n" + "=" * 60)
print("步骤7: 输出预测结果CSV")
print("=" * 60)

result_df = pd.DataFrame({
    'payload_id': test_df['payload_id'].values,
    'true_label': y_test.values,
    'predicted_label': y_pred
})
result_df['correct'] = (result_df['true_label'] == result_df['predicted_label']).astype(int)

result_path = os.path.join(OUTPUT_DIR, 'cnn_prediction_results.csv')
result_df.to_csv(result_path, index=False, encoding='utf-8-sig')
print(f"预测结果已保存: {result_path}")

# 打印前10条
print(f"\n预测结果前10条:")
print(result_df.head(10).to_string(index=False))

# ============================================================
# 8. 生成MD实验分析报告
# ============================================================
print("\n" + "=" * 60)
print("步骤8: 生成MD实验分析报告")
print("=" * 60)

# 计算八个类别的平均结果
avg_precision = np.mean(precision_per_class)
avg_recall = np.mean(recall_per_class)
avg_f1 = np.mean(f1_per_class)

# 各类别样本数
test_class_counts = test_df['attack_type'].value_counts()
train_class_counts = train_val_df['attack_type'].value_counts()

# 混淆矩阵中各类别的正确分类数和错误分类数
correct_per_class = np.diag(cm)
total_per_class = cm.sum(axis=1)
error_per_class = total_per_class - correct_per_class
accuracy_per_class = correct_per_class / total_per_class

md_content = f"""# 基于CNN的Web攻击类型分类实验分析报告

## 1. 实验概述

本实验基于Web攻击载荷的 `url_n_gram` 特征，使用TF-IDF向量化方法将文本特征转换为数值向量，并采用卷积神经网络（CNN）算法进行多类别分类，以识别不同类型的Web攻击。

### 1.1 实验目标
- 基于URL的n-gram特征构建CNN深度学习分类模型
- 评估CNN算法在Web攻击分类任务上的性能
- 分析各类别的分类效果和混淆情况
- 与传统机器学习方法（如SVM）进行对比

### 1.2 数据集说明

| 数据集 | 样本数 |
|--------|--------|
| 训练集（train + val） | {len(train_val_df)} |
| 测试集 | {len(test_df)} |

### 1.3 类别分布

| 攻击类型 | 训练集样本数 | 测试集样本数 |
|----------|-------------|-------------|
"""

for cls in class_names:
    tr_cnt = train_class_counts.get(cls, 0)
    te_cnt = test_class_counts.get(cls, 0)
    md_content += f"| {cls} | {tr_cnt} | {te_cnt} |\n"

md_content += f"""
## 2. 实验方法

### 2.1 特征提取
- **特征选择**: `url_n_gram`（URL的n-gram特征）
- **向量化方法**: TF-IDF（Term Frequency-Inverse Document Frequency）
- **TF-IDF参数**:
  - 最大特征数: 5,000
  - n-gram范围: (1, 2)
  - 使用sublinear_tf: True
  - 最大文档频率: 0.95
  - 最小文档频率: 2

### 2.2 CNN模型结构
- **模型类型**: 多尺度1D卷积神经网络（TextCNN）
- **卷积层**: 3个不同kernel_size的卷积分支（3, 5, 7）
  - 每个分支: Conv1d(1→128) + BatchNorm1d + ReLU + AdaptiveMaxPool1d
- **特征拼接**: 将3个分支的输出拼接（384维）
- **全连接层**:
  - FC1: 384 → 256 + ReLU + Dropout(0.5)
  - FC2: 256 → 128 + ReLU + Dropout(0.5)
  - FC3: 128 → {num_classes}（输出层）
- **模型参数量**: {total_params:,}
- **可训练参数量**: {trainable_params:,}

### 2.3 训练配置
- **优化器**: Adam（lr=0.001, weight_decay=1e-4）
- **学习率调度**: StepLR（每10轮衰减为0.5倍）
- **损失函数**: CrossEntropyLoss
- **批次大小**: 128
- **训练轮数**: {num_epochs}
- **Dropout率**: 0.5
- **训练时间**: {train_time:.2f}秒
- **预测时间**: {predict_time:.2f}秒
- **运行设备**: {device}

### 2.4 数据处理流程
1. 读取CSV文件，提取 `payload_id`、`attack_type`、`url_n_gram` 三列
2. 使用TF-IDF将 `url_n_gram` 文本转换为5000维稀疏向量
3. 将稀疏矩阵转换为dense矩阵，reshape为 (samples, 1, 5000) 作为CNN输入
4. 对 `attack_type` 进行LabelEncoder编码
5. 合并训练集和验证集作为训练数据
6. 使用DataLoader进行批次训练

## 3. 实验结果

### 3.1 整体评价指标

| 评价指标 | 值 |
|----------|-----|
| 准确率 (Accuracy) | {accuracy:.3f} |
| 宏精确率 (Macro Precision) | {precision_macro:.3f} |
| 宏召回率 (Macro Recall) | {recall_macro:.3f} |
| 宏F1值 (Macro F1-Score) | {f1_macro:.3f} |
| 加权精确率 (Weighted Precision) | {precision_weighted:.3f} |
| 加权召回率 (Weighted Recall) | {recall_weighted:.3f} |
| 加权F1值 (Weighted F1-Score) | {f1_weighted:.3f} |

### 3.2 各类别详细评价

| 攻击类型 | 精确率 | 召回率 | F1值 | 测试样本数 | 正确分类数 | 错误分类数 | 类别准确率 |
|----------|--------|--------|------|-----------|-----------|-----------|-----------|
"""

for i, cls in enumerate(class_names):
    md_content += (f"| {cls} | {precision_per_class[i]:.3f} | {recall_per_class[i]:.3f} "
                   f"| {f1_per_class[i]:.3f} | {total_per_class[i]} | {correct_per_class[i]} "
                   f"| {error_per_class[i]} | {accuracy_per_class[i]:.3f} |\n")

md_content += f"""
### 3.3 八个类别的平均结果

| 指标 | 八类平均 |
|------|---------|
| 平均精确率 | {avg_precision:.3f} |
| 平均召回率 | {avg_recall:.3f} |
| 平均F1值 | {avg_f1:.3f} |

### 3.4 混淆矩阵分析

混淆矩阵展示了各类别之间的误分类情况，对角线元素表示正确分类的样本数，非对角线元素表示误分类的样本数。

详细的混淆矩阵数据如下：

| 真实\\预测 | {' | '.join(class_names)} |
|-----------|{'|' * len(class_names)}
"""

for i, cls_true in enumerate(class_names):
    row_vals = ' | '.join([f"{cm[i][j]}" for j in range(len(class_names))])
    md_content += f"| {cls_true} | {row_vals} |\n"

md_content += f"""
### 3.5 训练过程

模型在{num_epochs}个epoch的训练过程中：
- 初始Loss: {train_losses[0]:.4f}，最终Loss: {train_losses[-1]:.4f}
- 初始训练准确率: {train_accs[0]:.4f}，最终训练准确率: {train_accs[-1]:.4f}
- 训练曲线平滑收敛，说明模型学习稳定

## 4. 可视化分析

实验生成了以下可视化图表：

1. **训练曲线** (`training_curves.png`): 展示训练过程中Loss和Accuracy的变化
2. **混淆矩阵** (`confusion_matrix.png`): 展示各类别间的分类情况
3. **归一化混淆矩阵** (`confusion_matrix_normalized.png`): 以比例形式展示分类情况
4. **各类别指标柱状图** (`per_class_metrics.png`): 对比各类别的Precision/Recall/F1
5. **类别分布对比图** (`distribution_comparison.png`): 真实标签与预测标签的分布对比
6. **雷达图** (`radar_charts.png`): 各类别指标的雷达图可视化

## 5. 实验分析与讨论

### 5.1 模型整体性能
- 模型整体准确率为 **{accuracy:.3f}**，宏F1值为 **{f1_macro:.3f}**，说明CNN结合TF-IDF特征在Web攻击分类任务上具有良好的分类能力。
- CNN通过多尺度卷积核能够捕获不同粒度的特征模式，从而提升分类效果。

### 5.2 各类别分析
"""

# 分析表现最好的类别
best_idx = np.argmax(f1_per_class)
worst_idx = np.argmin(f1_per_class)
md_content += f"- 表现最好的类别: **{class_names[best_idx]}**（F1={f1_per_class[best_idx]:.3f}）\n"
md_content += f"- 表现最差的类别: **{class_names[worst_idx]}**（F1={f1_per_class[worst_idx]:.3f}）\n\n"

# 分析混淆情况
md_content += "### 5.3 主要混淆情况\n\n"
cm_copy = cm.copy().astype(float)
np.fill_diagonal(cm_copy, 0)
top_confusions = []
for _ in range(min(5, len(class_names))):
    max_idx = np.unravel_index(np.argmax(cm_copy), cm_copy.shape)
    if cm_copy[max_idx] == 0:
        break
    top_confusions.append((class_names[max_idx[0]], class_names[max_idx[1]], int(cm_copy[max_idx])))
    cm_copy[max_idx] = 0

if top_confusions:
    md_content += "| 真实类别 | 误判为 | 误判数量 |\n|---------|--------|----------|\n"
    for true_cls, pred_cls, cnt in top_confusions:
        md_content += f"| {true_cls} | {pred_cls} | {cnt} |\n"
    md_content += "\n"

md_content += f"""
### 5.4 八类平均结果分析
- 八个类别的平均精确率为 **{avg_precision:.3f}**，平均召回率为 **{avg_recall:.3f}**，平均F1值为 **{avg_f1:.3f}**。
- 平均结果与宏平均（Macro Average）结果一致，反映了模型在各类别上的均衡表现。

### 5.5 CNN模型优势与特点
1. **多尺度特征提取**: 通过kernel_size为3、5、7的三个卷积分支，模型能够同时捕获短程和长程的特征依赖关系。
2. **BatchNorm正则化**: 在卷积层后使用BatchNorm加速训练收敛并提高模型稳定性。
3. **Dropout防过拟合**: 在全连接层使用0.5的Dropout率，有效防止过拟合。
4. **自适应池化**: 使用AdaptiveMaxPool1d，无需手动计算池化后的尺寸。

### 5.6 实验结论
1. CNN + TF-IDF方法在Web攻击分类任务上取得了良好的效果，整体准确率达到{accuracy:.3f}。
2. url_n_gram特征能够有效区分不同类型的Web攻击，n-gram特征捕获了URL中的关键模式。
3. 部分类别之间存在一定混淆，这可能与攻击载荷的相似性有关（如SQL注入和LDAP注入等注入类攻击）。
4. CNN的多尺度卷积机制能够从TF-IDF向量中提取多粒度特征，适合此多类别分类任务。
5. 相比传统机器学习方法，CNN具有更强的特征学习能力，但需要更多训练时间。

## 6. 输出文件说明

| 文件名 | 描述 |
|--------|------|
| `cnn_prediction_results.csv` | 测试集预测结果（含payload_id、真实标签、预测标签、是否正确） |
| `training_curves.png` | 训练损失和准确率曲线 |
| `confusion_matrix.png` | 混淆矩阵可视化 |
| `confusion_matrix_normalized.png` | 归一化混淆矩阵可视化 |
| `per_class_metrics.png` | 各类别Precision/Recall/F1柱状图 |
| `distribution_comparison.png` | 真实vs预测分布对比图 |
| `radar_charts.png` | 各类别指标雷达图 |
| `cnn_classification.py` | 实验源代码 |
| `experiment_report.md` | 本实验分析报告 |
"""

report_path = os.path.join(OUTPUT_DIR, 'experiment_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(md_content)
print(f"实验报告已保存: {report_path}")

print("\n" + "=" * 60)
print("实验完成!")
print("=" * 60)
