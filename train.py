import os
import time
import warnings
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import get_device, to_device
from config import SEQUENCE_LENGTH, EMB_DIM

warnings.filterwarnings('ignore')


class PreprocessedDataset(Dataset):
    def __init__(self, data_path, training=True, input_size=5):

        number = 0
        with open(data_path, "r") as f:
            # 获得训练数据的总行数
            for _ in tqdm(f, desc="load training dataset"):
                number += 1
        self.number = number
        self.fopen = open(data_path, 'r')
        self.sequence_size = SEQUENCE_LENGTH
        columns = np.array([['open_price-{}'.format(i), 'high_price-{}'.format(i),
                             'low_price-{}'.format(i), 'close_price-{}'.format(i),
                             'period_volume-{}'.format(i)] for i in range(SEQUENCE_LENGTH, 0, -1)]).reshape(1,
                                                                                                            -1).squeeze()
        if training:
            columns = np.append(columns, ['sc', 'indices', 'industry', 'label'])
        else:
            columns = np.append(columns, ['sc', 'indices', 'industry'])
        self.columns = columns
        self.training = training
        self.input_size = input_size

    def __len__(self):
        return self.number

    def __getitem__(self, index):
        line = self.fopen.__next__().strip()
        _data = list(map(float, line.split(',')[:SEQUENCE_LENGTH * self.input_size]))
        _data = torch.tensor(_data).resize(SEQUENCE_LENGTH, self.input_size)
        if self.training:
            _indices = list(map(int, line.split(',')[-3].split('-')))
            _indices_len = len(_indices)
            tmp = 30 - _indices_len
            tmp = [0] * tmp
            _mask = [1] * _indices_len + tmp
            _indices = _indices + tmp
            _indices = torch.tensor(_indices)
            _mask = torch.tensor(_mask)
            _industry = torch.tensor(int(line.split(',')[-2].replace('.0', '')))

            _label = torch.tensor(int(line.split(',')[-1]))
            return _data, _indices, _mask, _industry, _label

        _indices = list(map(int, line.split(',')[-2].split('-')))
        _indices_len = len(_indices)
        tmp = 30 - _indices_len
        tmp = [0] * tmp
        _mask = [1] * _indices_len + tmp
        _indices = _indices + tmp
        _indices = torch.tensor(_indices)
        _mask = torch.tensor(_mask)
        _industry = torch.tensor(int(line.split(',')[-1]))

        _sc = line.split(',')[-3]
        return _data, _indices, _mask, _industry, _sc


class DeviceDataLoader:
    def __init__(self, data_loader, device):
        self.dl = data_loader
        self.device = device

    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.dl)


class Attention(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # self.wq = nn.Linear(emb_dim, emb_dim)
        # self.wk = nn.Linear(emb_dim, emb_dim)
        # self.wv = nn.Linear(emb_dim, emb_dim)
        self.wq = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.PReLU()
        )
        self.wk = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.PReLU()
        )
        self.wv = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.PReLU()
        )
        self.emb_dim = emb_dim

    @torch.jit.export
    def forward(self, x):
        qx = self.wq(x)
        kx = self.wk(x)
        vx = self.wv(x)

        qk = torch.matmul(qx, kx.transpose(dim0=1, dim1=2)) / (self.emb_dim ** 0.5)
        qk_softmax = torch.softmax(qk, dim=2)

        # qkv = torch.einsum('tab,tbc -> tac', qk_softmax, vx)
        qkv = torch.matmul(qk_softmax, vx)
        return qkv


class sequenceModel(nn.Module):
    def __init__(self, step_input_size, hidden_size, sequence_size=SEQUENCE_LENGTH):
        super().__init__()
        self.lstm = nn.GRU(input_size=step_input_size, hidden_size=hidden_size, num_layers=1,
                           batch_first=True)
        self.pre_bn = nn.BatchNorm1d(step_input_size)
        self.att = Attention(hidden_size)

        self.indices_emb = nn.Embedding(100, EMB_DIM)
        self.industry_emb = nn.Embedding(100, EMB_DIM)

        _fc1 = nn.Linear(2 * sequence_size * hidden_size+EMB_DIM*2, hidden_size)
        # _fc1 = nn.Linear(hidden_size, int(hidden_size/2))
        _fc2 = nn.Linear(hidden_size, 4)
        self.mlp = nn.Sequential(
            _fc1,
            nn.PReLU(),
            _fc2
            # nn.Softmax()
        )

    def forward(self, x, x_indices, x_mask, x_indusry):
        x_indices = self.indices_emb(x_indices)
        x_indices = torch.einsum('abc,ab->abc', x_indices, x_mask).sum(1)
        x_indusry = self.industry_emb(x_indusry)

        x = torch.transpose(x, dim0=1, dim1=2)
        x = self.pre_bn(x)
        x = torch.transpose(x, dim0=1, dim1=2)
        x_seq_output, hn = self.lstm(x)
        # x = self.mlp(x_seq_output[:, -1, :])
        x_att = self.att(x_seq_output)
        # x = torch.cat([x_seq_output.sum(2), x_att.sum(2)], 1)
        x = torch.cat([x_seq_output.flatten(start_dim=1), x_att.flatten(start_dim=1), x_indices, x_indusry], 1)
        x = self.mlp(x)
        # x = torch.transpose(hn, dim0=0, dim1=1).flatten(start_dim=1)
        return x


def train(lr=0.001, batch_size=128, epoch=8):
    device = get_device()
    # device = 'cpu'
    model = sequenceModel(5, 12).to(device)
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
                loader = DeviceDataLoader(loader, device)
                pbar = tqdm(loader)
                pbar.set_description("[Epoch {}, File {}]".format(e, file))
                for _data, _indices, _mask, _indusry, _label in pbar:
                    softmax_res = model(_data, _indices, _mask, _indusry)
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
