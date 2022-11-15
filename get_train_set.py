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
    x['ma_change'] = x.groupby(['stock_code'])['ma'].diff().fillna(0)
    # do not shift 1
    x['ma15day'] = x.groupby(['stock_code'])['ma'].rolling(60).mean().reset_index()['ma']
    x['volume_15day'] = x.groupby(['stock_code'])['period_volume'].rolling(60).mean().reset_index()['period_volume']
    # x['volume_ts_15'] = x.groupby(['stock_code'])['period_volume'].rolling(60).apply(
    #     lambda a: ts_rank(a, 60)).reset_index()['period_volume']
    x = x.dropna()
    x['ma_change_rate'] = x['ma_change'] / x['ma']
    x['ma_change_rate_rank'] = x.groupby(['datetime'])[['ma_change_rate']].rank()
    x['double_ma_rate'] = x['ma'] / x['ma15day']
    x['double_ma_rate_rank'] = x.groupby(['datetime'])[['double_ma_rate']].rank()
    x['double_v_rate'] = x['period_volume'] / x['volume_15day']
    # x['volume_change_15day'] = x['period_volume'] / x['volume_15day']
    # x['volume_change_15day_rank'] = x.groupby(['datetime'])[['volume_change_15day']].rank()

    # ['ma', 'ma_change', 'ma15day', 'ma_change_rate', 'ma_change_rate_rank', 'double_ma_rate', 'double_ma_rate_rank',
    # volume_rate_rank, 'volume_change_15day', 'volume_change_15day_rank', 'volume_ts_15]
    return x


def get_bieo(x):
    i = x.index[0]
    while i < x.index[-1]:
        current_price = x.loc[i, 'high_price']
        next_current_price = x.loc[i + 1, 'close_price']
        if (current_price / next_current_price) >= 0.99:
            i += 1
            continue
        try:
            future_high_price = x.loc[range(i + 1, i + FUTURE_CHANCE_LENGTH + 1), 'close_price'].max()
        except KeyError:
            future_high_price = x.loc[range(i + 1, x.index[-1] + 1), 'close_price'].max()
        future_high_price_index = x.loc[range(i + 1, x.index[-1] + 1)][x['close_price'] == future_high_price].index[0]
        if future_high_price / current_price >= 1.21:
            x.loc[i, 'label'] = 1
            x.loc[future_high_price_index, 'label'] = 3
            x.loc[range(i + 1, future_high_price_index), 'label'] = 2
            i = future_high_price_index + 1
        else:
            i += 1
    return x


def get_label(_data, output_file, basic_info=None):
    _data['label'] = 0
    df_res = []
    for sc in _data['stock_code'].unique():
        tmp = _data[_data['stock_code'] == sc].sort_values(by='datetime').reset_index().drop('index', axis=1)
        tmp = get_bieo(tmp)
        df_res.append(tmp)

    df_res = pd.concat(df_res)
    df_res = df_res.reset_index().drop('index', axis=1)
    if basic_info is not None:
        df_res = pd.merge(df_res, basic_info[['stock_code', 'indices', 'industry']], on='stock_code', how='left')
        df_res = df_res.fillna('99')

    for i in df_res.index[:-1]:
        if df_res.loc[i + 1, 'close_price'] >= df_res.loc[i, 'close_price'] and \
                df_res.loc[i, 'label'] == 3 and df_res.loc[i + 1, 'label'] == 0 and \
                df_res.loc[i, 'stock_code'] == df_res.loc[i + 1, 'stock_code']:
            df_res.loc[i, 'label'] = 2
            df_res.loc[i + 1, 'label'] = 3

    for i in df_res.index[:-1]:
        if df_res.loc[i, 'label'] == 3 and df_res.loc[i + 1, 'label'] == 1 and \
                df_res.loc[i, 'stock_code'] == df_res.loc[i + 1, 'stock_code']:
            df_res.loc[i, 'label'] = 2
            df_res.loc[i + 1, 'label'] = 2

    columns = np.array([['open_price-{}'.format(i), 'high_price-{}'.format(i),
                         'low_price-{}'.format(i), 'close_price-{}'.format(i),
                         'period_volume-{}'.format(i)] for i in range(SEQUENCE_LENGTH, 0, -1)]).reshape(1, -1).squeeze()
    with open(output_file, 'a') as f:

        for sc in df_res['stock_code'].unique():
            tmp = df_res[df_res['stock_code'] == sc].sort_values(by='datetime').reset_index().drop('index', axis=1)
            for i in tmp.index[SEQUENCE_LENGTH:]:
                _step = np.array(tmp.loc[range(i - SEQUENCE_LENGTH, i),
                                         ['open_price', 'high_price', 'low_price', 'close_price', 'period_volume',
                                          'ma_change_rate', 'ma_change_rate_rank', 'double_ma_rate', 'double_v_rate',
                                          'double_ma_rate_rank']]) \
                    .reshape(1, -1).squeeze()
                if basic_info is not None:
                    emb = tmp.loc[i, ['indices', 'industry', 'hour', 'label']].astype(str)
                    if len(emb) == 0:
                        continue
                    _step = np.append(np.append(_step, sc), emb)
                else:
                    _step = np.append(_step, [sc, tmp.loc[i, 'label']])
                f.write(','.join(_step))
                f.write('\n')

    return


if __name__ == '__main__':
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database')
    engine = sqlalchemy.create_engine('sqlite:///{}'.format(os.path.join(database_path, 'StockKing.db')))
    df = pd.read_sql_table('ma60m', engine)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[~df['datetime'].isin(sorted(df['datetime'].unique())[-16:])]
    df = df[df['datetime'].isin(sorted(df['datetime'].unique())[-TRAIN_LENGTH:])]
    df = get_alpha(df)
    dt = datetime.date.today().strftime('%Y%m%d')
    emb_info = pd.read_sql_table('emb_info', engine)
    get_label(df, './trainset/train_set{}.csv'.format(dt), emb_info)
