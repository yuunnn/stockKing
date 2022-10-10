import os
import time
import warnings
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings('ignore')
SEQUENCE_SIZE = 20


class PreprocessedDataset(Dataset):
    def __init__(self, data_path, training=True):

        number = 0
        with open(data_path, "r") as f:
            # 获得训练数据的总行数
            for _ in tqdm(f, desc="load training dataset"):
                number += 1
        self.number = number
        self.fopen = open(data_path, 'r')
        self.sequence_size = SEQUENCE_SIZE
        columns = np.array([['open_price-{}'.format(i), 'high_price-{}'.format(i),
                             'low_price-{}'.format(i), 'close_price-{}'.format(i),
                             'period_volume-{}'.format(i)] for i in range(20, 0, -1)]).reshape(1, -1).squeeze()
        if training:
            columns = np.append(columns, ['sc', 'label'])
        else:
            columns = np.append(columns, ['sc'])
        self.columns = columns
        self.training = training

    def __len__(self):
        return self.number

    def __getitem__(self, index):
        line = self.fopen.__next__().strip()
        _data = list(map(float, line.split(',')[:100]))
        _data = torch.tensor(_data).resize(20, 5)
        if self.training:
            _label = torch.tensor(int(line.split(',')[-1]))
            return _data, _label
        return _data


class sequenceModel(nn.Module):
    def __init__(self, step_input_size, hidden_size, sequence_size=SEQUENCE_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(input_size=step_input_size, hidden_size=hidden_size, num_layers=1,
                            batch_first=True)
        self.pre_bn = nn.BatchNorm1d(step_input_size)

        _fc1 = nn.Linear(hidden_size * sequence_size, 64)
        # _fc1 = nn.Linear(hidden_size, 64)
        _fc2 = nn.Linear(64, 4)
        self.mlp = nn.Sequential(
            _fc1,
            nn.PReLU(),
            _fc2
            # nn.Softmax()
        )

    def forward(self, x):
        x = torch.transpose(x, dim0=1, dim1=2)
        x = self.pre_bn(x)
        x = torch.transpose(x, dim0=1, dim1=2)
        x_seq_output, (hn, cn) = self.lstm(x)
        # x = self.mlp(x_seq_output[:, -1, :])
        x = self.mlp(x_seq_output.flatten(start_dim=1))
        # x = torch.transpose(hn, dim0=0, dim1=1).flatten(start_dim=1)
        return x

def train(lr=0.001, batch_size=128, epoch=5):
    model = sequenceModel(5, 5)
    optim = Adam(model.parameters(), lr=lr)
    ts = int(time.time())
    ce = nn.CrossEntropyLoss()
    for e in range(epoch):
        total_loss = 0
        batch_count = 1
        for file in os.listdir('./trainset'):
            if len(file.split('.')) == 1:
                _dataset = PreprocessedDataset(os.path.join('./trainset', file))
                loader = DataLoader(_dataset, batch_size=batch_size, shuffle=True)
                pbar = tqdm(loader)
                pbar.set_description("[Epoch {}, File {}]".format(e, file))
                for _data, _label in pbar:
                    softmax_res = model(_data)
                    # loss = -torch.take(softmax_res, _label).log().sum()
                    loss = ce(softmax_res, _label)
                    total_loss += loss.item()
                    optim.zero_grad()
                    loss.backward()
                    # 更新参数
                    optim.step()
                    pbar.set_postfix(avg_loss=total_loss / batch_count)
                    batch_count += 1
        torch.save(model, os.path.join('./models', f'model_{ts}.pkl'))  # save entire net


if __name__ == "__main__":
    train()
