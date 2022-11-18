from config import SEQUENCE_LENGTH, FUTURE_CHANCE_LENGTH, TRAIN_LENGTH
from functools import reduce
import datetime

import numpy as np
import pandas as pd
import os
import sqlalchemy
import warnings
from sortedcontainers import SortedList
from numba import jit

warnings.filterwarnings("ignore")


@jit
def ts_rank(x, n):
    res = x.get(x.index[-1])
    sl = SortedList(x)
    return sl.bisect_left(res) / n


def datetime_hour_to_index(x):
    if x.hour == 10:
        return 0
    if x.hour == 11:
        return 1
    if x.hour == 14:
        return 2
    if x.hour == 15:
        return 3
    return 'fuck the datetime is wrong'


def get_alpha(x):
    x['hour'] = x['datetime'].apply(datetime_hour_to_index)

    x['ma'] = (x['high_price'] + x['low_price']) / 2
    x = x.sort_values(by=['stock_code', 'datetime'])
    x['ma_change'] = x.groupby(['stock_code'])['ma'].diff().fillna(0).tolist()
    # do not shift 1
    x['ma15day'] = x.groupby(['stock_code'])['ma'].rolling(60).mean().reset_index()['ma'].tolist()
    x['volume_15day'] = x.groupby(['stock_code'])['period_volume'].rolling(60).mean().reset_index()['period_volume'].tolist()
    # x['volume_ts_15'] = x.groupby(['stock_code'])['period_volume'].rolling(60).apply(
    #     lambda a: ts_rank(a, 60)).reset_index()['period_volume']
    x = x.dropna()
    x['ma_change_rate'] = x['ma_change'] / x['ma']
    x['ma_change_rate_rank'] = x.groupby(['datetime'])['ma_change_rate'].rank().tolist()
    x['double_ma_rate'] = x['ma'] / x['ma15day']
    x['double_ma_rate_rank'] = x.groupby(['datetime'])['double_ma_rate'].rank().tolist()
    x['double_v_rate'] = x['period_volume'] / x['volume_15day']
    # x['volume_change_15day'] = x['period_volume'] / x['volume_15day']
    # x['volume_change_15day_rank'] = x.groupby(['datetime'])[['volume_change_15day']].rank()

    # ['ma', 'ma_change', 'ma15day', 'ma_change_rate', 'ma_change_rate_rank', 'double_ma_rate', 'double_ma_rate_rank',
    # volume_rate_rank, 'volume_change_15day', 'volume_change_15day_rank', 'volume_ts_15]
    return x


def get_label(_data, output_file, basic_info=None):
    _data = _data.sort_values(by=['stock_code', 'datetime'])
    _data['label'] = 0
    _data['next_current_price'] = _data.groupby('stock_code')['close_price'].shift(-1).tolist()
    _data['future_high_price'] = _data.groupby('stock_code')['close_price'].shift(-FUTURE_CHANCE_LENGTH).rolling(
        FUTURE_CHANCE_LENGTH).max().tolist()
    _data['future_low_price'] = _data.groupby('stock_code')['low_price'].shift(-FUTURE_CHANCE_LENGTH).rolling(
        FUTURE_CHANCE_LENGTH).min().tolist()
    _data = _data.dropna()
    _data.loc[_data['future_high_price'] / _data['high_price'] >= 1.04, 'label'] = 1
    _data.loc[(_data['future_high_price'] / _data['high_price'] >= 1.1) & (
            _data['future_low_price'] / _data['high_price'] >= 0.95), 'label'] = 2
    _data.loc[(_data['future_high_price'] / _data['high_price'] >= 1.16) & (
            _data['future_low_price'] / _data['high_price'] >= 0.97), 'label'] = 3
    _data.loc[_data['next_current_price'] / _data['high_price'] <= 1, 'label'] = 0

    _data['period_volume'] /= 1000000
    _data['volume_15day'] /= 1000000
    _data['ma_change_rate_rank'] = 100 / _data['ma_change_rate_rank']
    _data['double_ma_rate_rank'] = 100 / _data['double_ma_rate_rank']
    _data['ma_change_rate'] *= 100
    _data['open_price'] = np.log1p(_data['open_price'])
    _data['high_price'] = np.log1p(_data['high_price'])
    _data['low_price'] = np.log1p(_data['low_price'])
    _data['close_price'] = np.log1p(_data['close_price'])

    df_res = _data

    if basic_info is not None:
        df_res = pd.merge(df_res, basic_info[['stock_code', 'indices', 'industry']], on='stock_code', how='left')
        df_res = df_res.fillna('99')

    step_col = []
    for i in range(SEQUENCE_LENGTH, 0, -1):
        for col in ['open_price', 'high_price', 'low_price', 'close_price', 'period_volume',
                    'ma_change_rate', 'ma_change_rate_rank', 'double_ma_rate', 'double_v_rate',
                    'double_ma_rate_rank']:
            _step_col = '{}{}'.format(col, i)
            step_col.append(_step_col)
            df_res[_step_col] = df_res.groupby('stock_code')[col].shift(i).tolist()

    df_res = df_res.dropna()
    step_col.append('stock_code')
    if basic_info is not None:
        step_col.extend(['indices', 'industry', 'hour'])
    step_col.append('label')
    print(df_res['datetime'].max())
    df_res[step_col].to_csv(output_file, index=None, header=None)
    return


if __name__ == '__main__':
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database')
    engine = sqlalchemy.create_engine('sqlite:///{}'.format(os.path.join(database_path, 'StockKing.db')))
    df = pd.read_sql_table('ma60m', engine)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[df['datetime'].isin(sorted(df['datetime'].unique())[-TRAIN_LENGTH - SEQUENCE_LENGTH:])]
    df = get_alpha(df)
    dt = datetime.date.today().strftime('%Y%m%d')
    emb_info = pd.read_sql_table('emb_info', engine)
    get_label(df, './trainset/train_set{}.csv'.format(dt), emb_info)
