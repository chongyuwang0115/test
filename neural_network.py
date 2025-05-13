import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import adjusted_rand_score
import numpy as np

class SimpleNN(nn.Module):
    """
    简单的全连接神经网络
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_neural_network(X, y, num_epochs=50, learning_rate=0.0005, hidden_size=256):
    # 仅使用有标签的样本进行训练
    labeled_indices = y != -1
    X_train = X[labeled_indices]
    y_train = y[labeled_indices]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = X_train.shape[1]
    output_size = len(np.unique(y_train))
    model = SimpleNN(input_size, hidden_size, output_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)

    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    X_all = torch.tensor(X, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        y_pred = torch.argmax(model(X_all), dim=1).cpu().numpy()

    return y_pred
