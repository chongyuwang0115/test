import cupy as cp
import numpy as np
from scipy.sparse import coo_matrix
import math
from scipy.sparse import csr_matrix

def label_propagation(X, y_label, alpha=0.5, max_iter=1000):
    """标签传播算法的GPU加速版本"""
    n_samples = X.shape[0]
    n_classes = y_label.shape[1]
    
    # 一次性将所有数据转移到GPU
    with cp.cuda.Device(0):
        # 检查 y_label 是否已经是 one-hot 编码，如果不是，进行转换
        if len(y_label.shape) == 1 or y_label.shape[1] != n_classes:
            y_label = np.eye(n_classes)[y_label]  # 转换为 one-hot 编码

        # 将 X 转换为稀疏矩阵格式
        if not isinstance(X, csr_matrix):
            X = csr_matrix(X)  # 将 X 转换为 CSR 格式的稀疏矩阵

        # 先将COO矩阵转换为CSR格式
        X_csr = X.tocsr()
        
        # 构建相似度矩阵W (使用CSR格式)
        W = cp.sparse.csr_matrix(
            (cp.exp(-alpha * cp.array(X_csr.data) ** 2),
             cp.array(X_csr.indices),
             cp.array(X_csr.indptr)),
            shape=X.shape
        )
        
        # 将标签转换为概率分布格式
        Y = cp.array(y_label, dtype=cp.float32)
        Y_old = cp.zeros_like(Y)
        y_label_gpu = cp.array(y_label)
        
        # 对未标记的样本初始化为均匀分布
        unlabeled_mask = ~(y_label_gpu != -1).any(axis=1)
        Y[unlabeled_mask] = 1.0 / n_classes
        
        # 主循环
        for _ in range(max_iter):
            Y_old = Y.copy()
            
            # 标签传播
            Y = W.dot(Y)
            
            # 按行归一化
            row_sums = Y.sum(axis=1).reshape(-1, 1)
            Y = Y / cp.maximum(row_sums, 1e-12)
            
            # 保持已标记样本不变
            labeled_mask = (y_label_gpu != -1).any(axis=1)
            Y[labeled_mask] = y_label_gpu[labeled_mask]
            
            # 检查收敛
            diff = cp.abs(Y - Y_old).sum() / n_samples
            if _ > 0 and diff < 1e-5:
                break
        
        # 获取预测结果（转换为单一标签）
        y_pred = cp.zeros(n_samples, dtype=cp.int32)
        for i in range(n_samples):
            if cp.any(Y[i] > 0):  # 只处理有效的预测
                y_pred[i] = cp.argmax(Y[i])
            else:
                y_pred[i] = 0  # 默认标签
                
        return cp.asnumpy(y_pred)

def data_process(y_old, preserved=0.8, changed=0.1, masked=0.1):
    """数据预处理函数"""
    n_samples, n_classes = y_old.shape
    y_new = np.full_like(y_old, -1, dtype=np.float32)
    
    # 只处理有标签的样本
    labeled_mask = (y_old != -1).any(axis=1)
    labeled_indices = np.where(labeled_mask)[0]
    
    for idx in labeled_indices:
        r = np.random.random()
        if r < preserved:
            # 保持原标签
            y_new[idx] = y_old[idx]
        elif r < preserved + changed:
            # 随机选择一个不同的标签
            current_label = np.argmax(y_old[idx])
            possible_labels = list(range(n_classes))
            possible_labels.remove(current_label)
            if possible_labels:
                new_label = np.random.choice(possible_labels)
                y_new[idx] = np.zeros(n_classes)
                y_new[idx, new_label] = 1
    
    return y_new
