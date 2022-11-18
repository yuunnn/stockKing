import datetime
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
from train import sequenceModel, Attention, PreprocessedDataset, DeviceDataLoader
from config import SEQUENCE_LENGTH, INPUT_SIZE
from utils import get_device

warnings.filterwarnings('ignore')


def predict(model_file, predicts_file):
    device = get_device()
    model = torch.load(model_file).to(device)
    _dataset = PreprocessedDataset(predicts_file, training=False, input_size=INPUT_SIZE)
    loader = DataLoader(_dataset, batch_size=32)
    loader = DeviceDataLoader(loader, device)

    pbar = tqdm(loader)
    stocks = []
    res = []
    for _data, _indices, _mask, _indusry, _hour, sc in pbar:
        softmax_res = nn.Softmax()(model(_data, _indices, _mask, _indusry, _hour))
        stocks.append(sc)
        res.append(softmax_res)
    return torch.cat(res).cpu(), np.concatenate(stocks)


if __name__ == "__main__":
    probs, stock_codes = predict('./models/model_1668783764.pkl', './predictset/latest.csv')
    df = pd.DataFrame(probs.detach().numpy())
    df.columns = ['减持', '中性', '谨慎增持', '增持']
    df['code'] = stock_codes
    dt = datetime.date.today().strftime('%Y%m%d')
    df.to_csv('./predictset/output{}.csv'.format(dt), index=False)
