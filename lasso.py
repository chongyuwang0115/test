import scanpy as sc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score
import random
import matplotlib.pyplot as plt
from LPA2 import label_propagation  # 导入自定义的 LPA 函数
from scipy.sparse import csr_matrix
from svm import train_svm, calculate_ARI  # 导入 SVM 模型
from decision_tree import train_decision_tree
from neural_network import train_neural_network

# 路径设置
data_path = r"C:\Users\18176\Desktop\LPA\dataset.h5ad"

# VAE 模型定义
class VAE(nn.Module):
    def __init__(self, input_size, output_size, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_size, h_dim)
        self.batch_norm1 = nn.BatchNorm1d(h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.batch_norm2 = nn.BatchNorm1d(h_dim)
        self.fc5 = nn.Linear(h_dim, output_size)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = self.batch_norm1(h)
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        h = self.batch_norm2(h)
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

# 数据预处理与候选集生成
def Gen_TrainSet(h5adFile):
    adata = sc.read_h5ad(h5adFile)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor='seurat', inplace=True)
    sc.pp.pca(adata, n_comps=50, use_highly_variable=True, svd_solver='arpack')
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    candidates = pd.DataFrame(index=adata.obs_names)
    for res in range(1, 10, 2):
        lores = res * 0.1
        sc.tl.leiden(adata, key_added='clusters', resolution=lores)
        tmp_can = pd.DataFrame(
            np.array([adata.obs['clusters'] == x for x in adata.obs['clusters'].unique()]).T,
            index=adata.obs_names)
        candidates = pd.concat([candidates, tmp_can], axis=1)
    return adata, candidates

# 标签遮蔽与错误标签生成
def Gen_maskSet(candidate: pd.DataFrame, errRate=0.20):
    """
    根据候选集生成一维标签数组（适配 LabelPropagation）
    """
    # 初始化标签为 -1（未知）
    labels = np.full(candidate.shape[0], -1, dtype=int)

    # 随机选择一些样本为已知标签
    known_indices = random.sample(range(candidate.shape[0]), int(candidate.shape[0] * (1 - errRate)))

    for idx in known_indices:
        # 找到行中第一个 True 的列索引作为标签
        row = candidate.iloc[idx]
        if row.any():
            labels[idx] = row.idxmax()

    return labels

# 使用 VAE 训练
def train_VAE(candidates, h_dim=400, z_dim=20, num_epochs=50, learning_rate=1e-3, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = candidates.shape[1]
    model = VAE(input_size=input_size, output_size=input_size, h_dim=h_dim, z_dim=z_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    candidates_np = candidates.to_numpy(dtype=np.float32)
    dataset_size = candidates_np.shape[0]

    for epoch in range(num_epochs):
        # 按批次处理数据
        for start in range(0, dataset_size, batch_size):
            end = min(start + batch_size, dataset_size)
            batch_candidates = candidates_np[start:end]

            # 修复遮蔽矩阵维度
            mask_candidates = np.zeros_like(batch_candidates, dtype=np.float32)
            known_indices = random.sample(range(batch_candidates.shape[0]), int(batch_candidates.shape[0] * 0.95))
            mask_candidates[known_indices] = batch_candidates[known_indices]

            # 转换为张量
            mask_x = torch.tensor(mask_candidates, dtype=torch.float32).to(device)
            origin_x = torch.tensor(batch_candidates, dtype=torch.float32).to(device)

            # 前向传播
            reconst_x, mu, log_var = model(mask_x)
            reconst_loss = F.binary_cross_entropy(reconst_x, origin_x, reduction='mean')
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = reconst_loss + kl_div

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

    return model

def to_one_hot(y, n_classes):
    """将标签转换为 one-hot 编码"""
    return np.eye(n_classes)[y]

# 标签传播训练 (使用自定义的 label_propagation)
def train_LPA(candidate, adata, errRate=0.05):
    """
    使用自定义的标签传播算法训练并预测标签
    """
    X = adata.obsm['X_pca']  # 特征矩阵
    y = Gen_maskSet(candidate, errRate)  # 生成一维标签

    # 获取有标签的类别数
    n_classes = len(np.unique(y[y != -1]))
    if n_classes == 0:
        raise ValueError("No labeled data available.")

    # 将标签转换为 one-hot 编码，未标记的样本保持为全0
    y_one_hot = np.zeros((len(y), n_classes), dtype=np.float32)
    labeled_indices = y != -1
    y_one_hot[labeled_indices, y[labeled_indices]] = 1.0

    print(f"X shape: {X.shape}, y_one_hot shape: {y_one_hot.shape}")

    # 使用 adata 的邻接矩阵作为 W
    W = adata.obsp['connectivities'].copy()

    print(f"W shape: {W.shape}, Y shape: {y_one_hot.shape}")

    y_pred = label_propagation(W, y_one_hot)  # 使用自定义 LPA

    return y_pred

# 计算 ARI
def calculate_ARI(true_labels, predicted_labels):
    ari_score = adjusted_rand_score(true_labels, predicted_labels)
    print(f"ARI Score: {ari_score:.4f}")
    return ari_score

# 可视化结果
def visualize_result(adata, true_labels, predicted_labels):
    adata.obs["True Labels"] = true_labels
    adata.obs["Predicted Labels"] = predicted_labels
    sc.pl.umap(adata, color=["True Labels", "Predicted Labels"])

# 主函数
if __name__ == "__main__":
    # 数据路径
    # 1. 生成数据集和候选标签
    adata, candidates = Gen_TrainSet(data_path)
    print("Candidates Generated.")

    # 2. 使用 VAE 模型进行特征学习
    vae_model = train_VAE(candidates)
    print("VAE Training Complete.")

    # 3. 使用标签传播算法 (LPA)
    y_pred_lpa = train_LPA(candidates, adata)
    print("Label Propagation Training Complete.")

    # 4. 使用 SVM 模型进行训练和预测
    X = adata.obsm['X_pca']  # 获取特征矩阵
    y = Gen_maskSet(candidates)  # 生成遮蔽标签

    # 训练 SVM 并预测标签
    y_pred_svm = train_svm(X, y)
    print("SVM Training Complete.")

    # 5. 使用决策树模型进行训练和预测
    from decision_tree import train_decision_tree
    y_pred_tree = train_decision_tree(X, y)
    print("Decision Tree Training Complete.")

    # 6. 使用神经网络模型进行训练和预测
    from neural_network import train_neural_network
    y_pred_nn = train_neural_network(X, y)
    print("Neural Network Training Complete.")

    # 7. 获取真实标签 (假设存储在 adata.obs['clusters'] 中)
    true_labels = adata.obs["clusters"].astype(int)

    # 8. 计算 ARI 指标
    print("Performance Metrics:")
    ari_lpa = calculate_ARI(true_labels, y_pred_lpa)  # 标签传播的 ARI
    ari_svm = calculate_ARI(true_labels, y_pred_svm)  # SVM 的 ARI
    ari_tree = calculate_ARI(true_labels, y_pred_tree)  # 决策树的 ARI
    ari_nn = calculate_ARI(true_labels, y_pred_nn)  # 神经网络的 ARI

    # 9. 可视化结果
    visualize_result(adata, true_labels, y_pred_lpa)
    visualize_result(adata, true_labels, y_pred_svm)
    visualize_result(adata, true_labels, y_pred_tree)
    visualize_result(adata, true_labels, y_pred_nn)

    # 输出最终结果
    print(f"Final ARI Score (LPA): {ari_lpa:.4f}")
    print(f"Final ARI Score (SVM): {ari_svm:.4f}")
    print(f"Final ARI Score (Decision Tree): {ari_tree:.4f}")
    print(f"Final ARI Score (Neural Network): {ari_nn:.4f}")

