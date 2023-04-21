import numpy as np
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.preprocess import *
from src.features_extraction import *
from sklearn.preprocessing import StandardScaler


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

def neo_network(data, labels):
    output = np.max(labels)
    print(output)
    data = np.array(data)
    model = MLP(input_dim=len(data[0]), hidden_dim=100, output_dim=output+1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    num_epochs = 60000
    X = torch.Tensor(data)
    Y = torch.Tensor(labels)
    Y = Y.long()
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(X)
        loss = criterion(outputs, Y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印损失值
        if (epoch + 1) % 100 == 0:
            print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item()))

    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)

    print("Accuracy: {:.2f}%".format((predicted == Y).sum().item() / len(Y) * 100))


def random_forest(data, labels):
    X = data
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)
    return y_pred


if __name__ == '__main__':
    config.set_config(config.speciesType.human, config.chainType.beta)
    config.set_label(config.labelType.epitope)
    config.set_fe_method(config.feMethodType.giana_features)
    config.set_distance_method(config.distanceMethodType.giana)

    data = load_data().iloc[:10000, :]
    data, label = do_preprocess(data)
    feature_matrix = do_features_extraction(data)
    print(f'data: \n{data}')
    print(f'feature matrix: \n{feature_matrix}')
    neo_network(feature_matrix, data['label'])






