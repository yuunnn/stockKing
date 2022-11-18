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
    x['ma_change'] = x.groupby(['stock_code'])['ma'].diff().fillna(0).tolist()
    # do not shift 1
    x['ma15day'] = x.groupby(['stock_code'])['ma'].rolling(60).mean().reset_index()['ma'].tolist()
    x['volume_15day'] = x.groupby(['stock_code'])['period_volume'].rolling(60).mean().reset_index()[
        'period_volume'].tolist()
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


def get_data(_data, output_file, basic_info=None):
    _data = _data.sort_values(by=['stock_code', 'datetime'])
    df_res = _data

    if basic_info is not None:
        df_res = pd.merge(_data, basic_info[['stock_code', 'indices', 'industry']], on='stock_code', how='left')
        df_res = df_res.fillna('99')

    df_res['period_volume'] /= 1000000
    df_res['volume_15day'] /= 1000000
    df_res['ma_change_rate_rank'] = 1 / df_res['ma_change_rate_rank']
    df_res['double_ma_rate_rank'] = 1 / df_res['double_ma_rate_rank']
    df_res['ma_change_rate'] *= 100
    df_res['open_price'] = np.log1p(df_res['open_price'])
    df_res['high_price'] = np.log1p(df_res['high_price'])
    df_res['low_price'] = np.log1p(df_res['low_price'])
    df_res['close_price'] = np.log1p(df_res['close_price'])

    step_col = []
    for i in range(SEQUENCE_LENGTH, 0, -1):
        for col in ['open_price', 'high_price', 'low_price', 'close_price', 'period_volume',
                    'ma_change_rate', 'ma_change_rate_rank', 'double_ma_rate', 'double_v_rate',
                    'double_ma_rate_rank']:
            _step_col = '{}{}'.format(col, i)
            step_col.append(_step_col)
            df_res[_step_col] = df_res.groupby('stock_code')[col].shift(i).tolist()

    df_res = df_res.dropna()
    df_res = df_res[df_res['datetime'] == df_res['datetime'].max()]

    step_col.append('stock_code')
    if basic_info is not None:
        step_col.extend(['indices', 'industry', 'hour'])

    df_res[step_col].to_csv(output_file, index=None, header=None)
    return


if __name__ == '__main__':
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database')
    engine = sqlalchemy.create_engine('sqlite:///{}'.format(os.path.join(database_path, 'StockKing.db')))
    df = pd.read_sql_table('ma60m', engine)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[df['datetime'].isin(sorted(df['datetime'].unique())[-200:])]
    df = get_alpha(df)
    emb_info = pd.read_sql_table('emb_info', engine)
    get_data(df, './predictset/latest.csv', emb_info)
