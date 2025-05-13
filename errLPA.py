# evaluation the robustness of LPA
import h5py
import scipy.sparse
import time
import random
import numpy as np
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
import LPA  # 导入新的LPA模块

df = pd.DataFrame(columns=['MR', 'ARI_o', 'ARI_r', 'Time'])

start = time.time()
# 读取数据集
with h5py.File(r"D:\Desktop\机器学习\大作业\LPA\dataset.h5ad", 'r') as f:
    group = f['obsp']['connectivities']
    data = group['data'][:]
    indices = group['indices'][:]
    indptr = group['indptr'][:]
    shape = (f['obsp']['connectivities'].attrs['shape'][0],
            f['obsp']['connectivities'].attrs['shape'][1])
    
    # 创建稀疏矩阵
    mat = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
    coo = mat.tocoo()

# 读取标注数据
with h5py.File(r"D:\Desktop\机器学习\大作业\LPA\dataset.h5ad", 'r') as h5file:
    obs_group = h5file['obs']
    if "codes" in obs_group['annotation']:
        annotations = obs_group['annotation']['codes'][:]
    else:
        annotations = obs_group['annotation'][:]

# 构建标签映射
unique_labels = np.unique(annotations)
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
n_classes = len(unique_labels)

k1 = 1.0
while True:
    for _ in range(100):
        # 准备稀疏矩阵X
        X = scipy.sparse.coo_matrix((coo.data, (coo.row, coo.col)), 
                                  shape=(shape[0], shape[0]))
        
        # 准备标签矩阵
        y_label = np.full((shape[0], n_classes), -1, dtype=np.float32)
        
        # 随机选择10%的样本作为已标记数据
        random_indices = random.sample(range(shape[0]), int(shape[0] * 0.1))
        for idx in random_indices:
            label_idx = label_to_idx[annotations[idx]]
            y_label[idx] = np.zeros(n_classes)
            y_label[idx, label_idx] = 1
            
        # 数据处理
        start_time = time.perf_counter()
        y_new = LPA.data_process(y_label, preserved=k1, changed=(1-k1), masked=0)
        
        # 标签传播
        y_pred = LPA.label_propagation(X, y_new, alpha=0.5, max_iter=1000)
        
        # 结果修正
        y_res = LPA.rectify(X, y_label, y_pred)
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # 计算ARI分数
        ari_o = adjusted_rand_score(annotations, y_pred)
        ari_r = adjusted_rand_score(annotations, y_res)
        
        # 记录结果
        item = [round((1-k1), 2), ari_o, ari_r, round(execution_time, 5)]
        print(item)
        df.loc[len(df)] = item
        
    k1 -= 0.05
    print(f"k1: {k1}")
    if k1 < 0:
        break

# 保存结果
df.to_csv("mistake.tsv", sep='\t', header=True, index=True)
end = time.time()
print(f"Total time: {end-start}")
