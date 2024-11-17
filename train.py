import os
import subprocess
import time
import warnings
import torch
from torch import nn
from torch.optim import RAdam, SGD, AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import get_device, to_device
from config import SEQUENCE_LENGTH, EMB_DIM, INPUT_SIZE, HIDDEN_SIZE, NHEAD

warnings.filterwarnings('ignore')


class PreprocessedDataset(Dataset):
    def __init__(self, data_path, training=True, backtest=False, input_size=5):

        number = 0
        with open(data_path, "r") as f:
            # 获得训练数据的总行数
            for _ in tqdm(f, desc="load training dataset"):
                number += 1
        self.number = number
        self.fopen = open(data_path, 'r')
        self.sequence_size = SEQUENCE_LENGTH
        self.training = training
        self.input_size = input_size
        self.backtest = backtest

    def __len__(self):
        return self.number

    def __getitem__(self, index):
        line = self.fopen.__next__().strip()
        _data = list(map(float, line.split(',')[:SEQUENCE_LENGTH * self.input_size]))
        _data = torch.tensor(_data).resize(SEQUENCE_LENGTH, self.input_size)
        if self.training:
            _indices = list(map(int, line.split(',')[-4].split('-')))
            _indices_len = len(_indices)
            tmp = 30 - _indices_len
            tmp = [0] * tmp
            _mask = [1] * _indices_len + tmp
            _indices = _indices + tmp
            _indices = torch.tensor(_indices)
            _mask = torch.tensor(_mask)
            _industry = torch.tensor(int(line.split(',')[-3].replace('.0', '')))
            _hour = torch.tensor(int(line.split(',')[-2]))

            _label = torch.tensor(float(line.split(',')[-1]))
            return _data, _indices, _mask, _industry, _hour, _label

        if self.backtest:
            _indices = list(map(int, line.split(',')[-6].split('-')))
            _indices_len = len(_indices)
            tmp = 30 - _indices_len
            tmp = [0] * tmp
            _mask = [1] * _indices_len + tmp
            _indices = _indices + tmp
            _indices = torch.tensor(_indices)
            _mask = torch.tensor(_mask)
            _industry = torch.tensor(int(line.split(',')[-5].replace('.0', '')))
            _hour = torch.tensor(int(line.split(',')[-4]))
            _sc = line.split(',')[-7]
            _openprice = torch.tensor(float(line.split(',')[-3]))
            _closeprice = torch.tensor(float(line.split(',')[-2]))
            _datetime = line.split(',')[-1]
            return _data, _indices, _mask, _industry, _hour, _openprice, _closeprice,_sc, _datetime

        _indices = list(map(int, line.split(',')[-3].split('-')))
        _indices_len = len(_indices)
        tmp = 30 - _indices_len
        tmp = [0] * tmp
        _mask = [1] * _indices_len + tmp
        _indices = _indices + tmp
        _indices = torch.tensor(_indices)
        _mask = torch.tensor(_mask)
        _industry = torch.tensor(int(line.split(',')[-2].replace('.0', '')))
        _hour = torch.tensor(int(line.split(',')[-1]))

        _sc = line.split(',')[-4]
        return _data, _indices, _mask, _industry, _hour, _sc


class DeviceDataLoader:
    def __init__(self, data_loader, device):
        self.dl = data_loader
        self.device = device

    def __iter__(self):
        # batch = yield from self.dl
        for batch in self.dl:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.dl)


class sequenceModel(nn.Module):
    def __init__(self, step_input_size, hidden_size=HIDDEN_SIZE, sequence_size=SEQUENCE_LENGTH, nhead=NHEAD,
                 num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size

        # 输入嵌入层，将输入映射到 hidden_size 维度
        self.input_proj = nn.Linear(step_input_size, hidden_size)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size * 4,
                                                   dropout=0.1, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 类别特征嵌入层
        self.indices_emb = nn.Embedding(100, EMB_DIM)
        self.industry_emb = nn.Embedding(100, EMB_DIM)
        self.hour_emb = nn.Embedding(4, EMB_DIM)

        # 全连接层
        total_feature_dim = hidden_size + EMB_DIM * 3
        self.fc = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_size * 4),
            nn.BatchNorm1d(hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, 1)
        )

    def forward(self, x, x_indices, x_mask, x_industry, x_hour):
        # x: (batch_size, sequence_length, step_input_size)
        batch_size, seq_length, _ = x.size()

        # 输入嵌入
        x = self.input_proj(x)  # (batch_size, sequence_length, hidden_size)

        # 准备 Transformer 输入，调整形状为 (sequence_length, batch_size, hidden_size)
        x = x.permute(1, 0, 2)

        # 创建注意力掩码，防止 Transformer 关注未来的信息（如果需要）
        src_mask = None  # 或者根据需要创建形状为 (sequence_length, sequence_length) 的掩码矩阵

        # Transformer Encoder
        transformer_output = self.transformer_encoder(x, mask=src_mask)  # (sequence_length, batch_size, hidden_size)

        # 池化操作，将序列维度的信息聚合
        transformer_output = transformer_output.mean(dim=0)  # (batch_size, hidden_size)

        # 类别特征嵌入
        x_indices = self.indices_emb(x_indices)  # (batch_size, indices_length, EMB_DIM)
        x_indices = x_indices * x_mask.unsqueeze(-1)  # 应用掩码
        x_indices = x_indices.sum(dim=1)  # (batch_size, EMB_DIM)

        x_industry = self.industry_emb(x_industry)  # (batch_size, EMB_DIM)
        x_hour = self.hour_emb(x_hour)  # (batch_size, EMB_DIM)

        # 特征拼接
        x = torch.cat([transformer_output, x_indices, x_industry, x_hour], dim=1)  # (batch_size, total_feature_dim)

        # 全连接层
        x = self.fc(x)  # (batch_size, 1)

        return x.squeeze(-1)  # 输出形状为 (batch_size,)


def train(lr=0.007, batch_size=128, epoch=8):
    device = get_device()
    # device = 'cpu'
    model = sequenceModel(INPUT_SIZE).to(device)
    optim = SGD(model.parameters(), lr=lr, momentum=0.95)
    # optim = AdamW(model.parameters(), lr=lr)
    ts = int(time.time())
    # ce = nn.CrossEntropyLoss()
    # distance_array = torch.asarray([0, 1, 2, 3, 4]).to(device)
    # alpha = torch.tensor([0.1, 0.2, 0.3, 0.4])
    loss_fn = nn.MSELoss()
    for e in range(epoch):
        total_loss = 0
        # total_focal_loss = 0
        # total_distance_loss = 0
        batch_count = 1
        for file in os.listdir('./trainset'):
            if len(file.split('.')) == 1:
                file = os.path.join('./trainset', file)
                subprocess.run(f"shuf {file} -o {file}", shell=True)
                _dataset = PreprocessedDataset(file, input_size=INPUT_SIZE)
                loader = DataLoader(_dataset, batch_size=batch_size)
                loader = DeviceDataLoader(loader, device)
                pbar = tqdm(loader)
                pbar.set_description("[Epoch {}, File {}]".format(e, file))
                for _data, _indices, _mask, _indusry, _hour, _label in pbar:
                    res = model(_data, _indices, _mask, _indusry, _hour)
                    # 回归损失
                    loss = loss_fn(res, _label)
                    total_loss += loss.item()
                    # total_focal_loss += focal_loss.item()
                    # total_distance_loss += distance_loss.item()
                    optim.zero_grad()
                    loss.backward()
                    # 更新参数
                    optim.step()
                    pbar.set_postfix(
                        avg_loss=total_loss / batch_count,
                        # focal_loss=total_focal_loss / batch_count,
                        # distance_loss=total_distance_loss / batch_count
                    )
                    batch_count += 1
        torch.save(model, os.path.join('./models', f'model_{ts}.pkl'))  # save entire net


if __name__ == "__main__":
    train()
