import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from train import sequenceModel

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
        _sc = line.split(',')[-1]
        return _data, _sc


def predict(model_file, predicts_file):
    model = torch.load(model_file)
    _dataset = PreprocessedDataset(predicts_file, training=False)
    loader = DataLoader(_dataset, batch_size=32)

    pbar = tqdm(loader)
    stocks = []
    res = []
    for _data, sc in pbar:
        softmax_res = nn.Softmax()(model(_data))
        stocks.append(sc)
        res.append(softmax_res)
    return torch.cat(res), np.concatenate(stocks)


if __name__ == "__main__":
    probs, stock_codes = predict('./models/model_1665421818.pkl', './predictset/latest.csv')
    df = pd.DataFrame(probs.detach().numpy())
    df.columns = ['不持有', '买入', '持有', '卖出']
    df['code'] = stock_codes
    df.to_csv('tmp.csv', index=False)