from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import adjusted_rand_score
import numpy as np

def train_decision_tree(X, y, errRate=0.20):
    """
    使用决策树进行训练和预测
    :param X: 特征矩阵
    :param y: 标签数组（包含部分未标记样本）
    :param errRate: 错误率，影响已标记样本数量
    :return: 预测标签数组
    """
    # 仅使用有标签的样本进行训练
    labeled_indices = y != -1
    X_train = X[labeled_indices]
    y_train = y[labeled_indices]

    if len(np.unique(y_train)) < 2:
        raise ValueError("At least two classes are required for Decision Tree training.")

    # 初始化决策树分类器并训练
    clf = DecisionTreeClassifier(max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    # 对所有样本进行预测
    y_pred = clf.predict(X)
    return y_pred

def calculate_ARI(true_labels, predicted_labels):
    """
    计算 Adjusted Rand Index (ARI)
    """
    ari_score = adjusted_rand_score(true_labels, predicted_labels)
    print(f"ARI Score (Decision Tree): {ari_score:.4f}")
    return ari_score
