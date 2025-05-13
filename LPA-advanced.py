import cupy as cp
import numpy as np
from cupyx.scipy.sparse import csr_matrix as gpu_csr_matrix
from scipy.sparse import csr_matrix

def label_propagation(X, y_label, alpha=0.5, max_iter=1000, tol=1e-5, block_size=1000):
    """标签传播算法的优化 GPU 版本"""
    n_samples = X.shape[0]
    n_classes = y_label.shape[1]

    with cp.cuda.Device(0):
        # 检查 y_label 是否为 one-hot 编码
        if len(y_label.shape) == 1 or y_label.shape[1] != n_classes:
            y_label = np.eye(n_classes)[y_label]  # 转换为 one-hot 编码

        # 将 X 转换为稀疏矩阵
        if not isinstance(X, csr_matrix):
            X = csr_matrix(X)  # 将 X 转换为 CSR 格式的稀疏矩阵

        # 将 CSR 矩阵转移到 GPU
        X_gpu = gpu_csr_matrix(X)

        # 构建相似度矩阵 W
        W = gpu_csr_matrix(
            (cp.exp(-alpha * cp.array(X.data) ** 2),
             cp.array(X.indices),
             cp.array(X.indptr)),
            shape=X.shape
        )

        # 初始化标签分布
        Y = cp.array(y_label, dtype=cp.float32)
        Y_old = cp.zeros_like(Y)
        y_label_gpu = cp.array(y_label)

        # 对未标记样本初始化为邻居的加权平均值
        unlabeled_indices = cp.where(~(y_label_gpu != -1).any(axis=1))[0]
        for idx in unlabeled_indices:
            row = W[idx]
            if row.nnz > 0:
                neighbor_labels = Y[row.indices]
                neighbor_weights = row.data
                Y[idx] = (neighbor_labels.T @ neighbor_weights) / cp.sum(neighbor_weights)
            else:
                Y[idx] = 1.0 / n_classes

        # 标签传播主循环
        for iteration in range(max_iter):
            Y_old = Y.copy()

            # 动态调整 alpha
            dynamic_alpha = alpha / (1 + 0.1 * iteration)

            # 标签传播（分块计算）
            for i in range(0, n_samples, block_size):
                end = min(i + block_size, n_samples)
                Y[i:end] = W[i:end].dot(Y)

            # 按行归一化
            row_sums = Y.sum(axis=1).reshape(-1, 1)
            Y = Y / cp.maximum(row_sums, 1e-12)

            # 保持已标记样本不变
            labeled_mask = (y_label_gpu != -1).any(axis=1)
            Y[labeled_mask] = y_label_gpu[labeled_mask]

            # 检查收敛
            diff = cp.abs(Y - Y_old).sum() / n_samples
            if iteration > 0 and diff < tol:
                print(f"Converged in {iteration + 1} iterations with diff: {diff:.6f}")
                break

        # 获取预测结果
        y_pred = cp.zeros(n_samples, dtype=cp.int32)
        for i in range(n_samples):
            if cp.any(Y[i] > 0):  # 只处理有效预测
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

def rectify(X, y_label, y_pred):
    """结果修正函数，基于邻居标签的加权统计"""
    with cp.cuda.Device(0):
        X_gpu = gpu_csr_matrix(X)
        y_pred_gpu = cp.array(y_pred)
        y_new = cp.array(y_pred.copy())

        # 获取需要修正的样本
        labeled_mask = cp.array(y_label != -1).any(axis=1)
        labeled_indices = cp.where(labeled_mask)[0]

        for i in labeled_indices:
            row = X_gpu[int(i)]
            if row.nnz > 0:
                # 获取邻居标签及其权重
                neighbor_labels = y_pred_gpu[row.indices]
                neighbor_weights = row.data

                # 统计加权标签
                unique_labels, counts = cp.unique(neighbor_labels, return_counts=True)
                weighted_counts = cp.zeros_like(unique_labels, dtype=cp.float32)
                for label, weight in zip(neighbor_labels, neighbor_weights):
                    weighted_counts[label] += weight

                # 选择权重最高的标签
                max_idx = cp.argmax(weighted_counts)
                y_new[int(i)] = unique_labels[max_idx]

        return cp.asnumpy(y_new)
