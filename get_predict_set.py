import numpy as np
import pandas as pd
import os
import sqlalchemy
import warnings
from numba import jit
from sortedcontainers import SortedList
from config import SEQUENCE_LENGTH

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


def rollingRankArgSort(array):
    try:
        return array.size - array.argsort().argsort()[-1]
    except KeyError:
        return np.nan


def get_alpha(x):
    x['hour'] = x['datetime'].apply(datetime_hour_to_index)

    x['ma'] = (x['high_price'] + x['low_price']) / 2
    x = x.sort_values(by=['stock_code', 'datetime'])
    x['ma_change'] = x.groupby(['stock_code'])['ma'].diff().fillna(0)
    # do not shift 1
    x['ma15day'] = x.groupby(['stock_code'])['ma'].rolling(60).mean().reset_index()['ma']
    x['volume_15day'] = x.groupby(['stock_code'])['period_volume'].rolling(60).mean().reset_index()['period_volume']
    x['volume_ts_15'] = x.groupby(['stock_code'])['period_volume'].rolling(15).apply(
        lambda a: ts_rank(a, 15)).reset_index()['period_volume']
    x = x.dropna()
    x['ma_change_rate'] = x['ma_change'] / x['ma']
    x['ma_change_rate_rank'] = x.groupby(['datetime'])[['ma_change_rate']].rank()
    x['double_ma_rate'] = x['ma'] / x['ma15day']
    x['double_ma_rate_rank'] = x.groupby(['datetime'])[['double_ma_rate']].rank()
    x['volume_change_15day'] = x['period_volume'] / x['volume_15day']
    x['volume_change_15day_rank'] = x.groupby(['datetime'])[['volume_change_15day']].rank()

    # ['ma', 'ma_change', 'ma15day', 'ma_change_rate', 'ma_change_rate_rank', 'double_ma_rate', 'double_ma_rate_rank',
    # volume_rate_rank, 'volume_change_15day', 'volume_change_15day_rank', 'volume_ts_15]
    return x


def get_data(_data, output_file, basic_info=None):
    with open(output_file, 'w') as f:
        for sc in _data['stock_code'].unique():
            tmp = _data[_data['stock_code'] == sc].sort_values(by='datetime').reset_index().drop('index', axis=1)
            try:
                _step = np.array(tmp.loc[range(SEQUENCE_LENGTH),
                                         ['open_price', 'high_price', 'low_price', 'close_price', 'period_volume',
                                          'ma_change_rate', 'ma_change_rate_rank', 'double_ma_rate', 'volume_ts_15',
                                          'double_ma_rate_rank', 'volume_change_15day', 'volume_change_15day_rank'
                                          ]]) \
                    .reshape(1, -1).squeeze()
            except KeyError:
                continue
            if basic_info is not None:
                emb = basic_info.loc[basic_info['stock_code'] == sc][['indices', 'industry']].astype(str)
                hour = str(list(tmp['hour'])[-1])
                if len(emb) == 0:
                    continue
                _step = np.append(np.append(np.append(_step, [sc]), emb), hour)
            else:
                _step = np.append(_step, [sc])
            f.write(','.join(_step))
            f.write('\n')
    return


if __name__ == '__main__':
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database')
    engine = sqlalchemy.create_engine('sqlite:///{}'.format(os.path.join(database_path, 'StockKing.db')))
    df = pd.read_sql_table('ma60m', engine)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = get_alpha(df)
    df = df[df['datetime'].isin(sorted(df['datetime'].unique())[-SEQUENCE_LENGTH:])]
    emb_info = pd.read_sql_table('emb_info', engine)
    get_data(df, './predictset/latest.csv', emb_info)
