import numpy as np
import pandas as pd
import os
import sqlalchemy
import warnings
from numba import jit, njit
from sortedcontainers import SortedList
from config import SEQUENCE_LENGTH

warnings.filterwarnings("ignore")


@jit
def ts_rank(x, n):
    res = x.get(x.index[-1])
    sl = SortedList(x)
    return sl.bisect_left(res) / n


@njit
def ts_rank_numba(a):
    return np.argsort(np.argsort(a))[-1] / (len(a) - 1)


def datetime_hour_to_index(x):
    if x.hour == 10:
        return 0
    if x.hour == 11:
        return 1
    if x.hour == 14:
        return 2
    if x.hour == 15:
        return 3
    else:
        # 返回默认值或抛出异常
        return -1  # 或者 raise ValueError(f"Unexpected hour value: {x.hour}")


def rollingRankArgSort(array):
    try:
        return array.size - array.argsort().argsort()[-1]
    except KeyError:
        return np.nan


def get_alpha(x):
    x['hour'] = x['datetime'].apply(datetime_hour_to_index)

    x = x.sort_values(by=['stock_code', 'datetime'])
    # 计算MA均线
    x['ma'] = (x['high_price'] + x['low_price']) / 2
    x['ma_change'] = x.groupby(['stock_code'])['ma'].diff().fillna(0)

    # 计算MA15天均线
    x['ma15day'] = x.groupby('stock_code')['ma'].transform(lambda s: s.rolling(window=60).mean())
    x['volume_15day'] = x.groupby('stock_code')['period_volume'].transform(lambda s: s.rolling(window=60).mean())
    x['volume_ts_15'] = x.groupby('stock_code')['period_volume'].transform(
        lambda s: s.rolling(window=60).apply(ts_rank_numba, raw=True))

    # 计算其他技术指标
    x['ma_change_rate'] = x['ma_change'] / x['ma']
    x['ma_change_rate_rank'] = x.groupby('datetime')['ma_change_rate'].rank()
    x['double_ma_rate'] = x['ma'] / x['ma15day']
    x['double_ma_rate_rank'] = x.groupby('datetime')['double_ma_rate'].rank()
    x['double_v_rate'] = x['period_volume'] / x['volume_15day']

    # 计算动量、均值回归和量比
    x['momentum'] = (x.groupby('stock_code')['close_price'].shift(1) - x.groupby('stock_code')['close_price'].shift(
        11)) / x.groupby('stock_code')['close_price'].shift(11)
    x['mean_reversion'] = (x['close_price'] - x.groupby('stock_code')['close_price'].transform(
        lambda s: s.rolling(window=20).mean())) / x.groupby('stock_code')['close_price'].transform(
        lambda s: s.rolling(window=20).std())
    x['volume_ratio'] = x['period_volume'] / x.groupby('stock_code')['period_volume'].transform(
        lambda s: s.rolling(window=10).mean())

    # 添加MACD指标
    x['ema12'] = x.groupby('stock_code')['close_price'].transform(lambda s: s.ewm(span=6, adjust=False).mean())
    x['ema26'] = x.groupby('stock_code')['close_price'].transform(lambda s: s.ewm(span=16, adjust=False).mean())
    x['macd'] = x['ema12'] - x['ema26']
    x['signal_line'] = x.groupby('stock_code')['macd'].transform(lambda s: s.ewm(span=5, adjust=False).mean())
    x['macd_hist'] = x['macd'] - x['signal_line']

    # 添加RSI指标
    delta = x.groupby('stock_code')['close_price'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    avg_gain = up.groupby(x['stock_code']).transform(lambda s: s.rolling(window=10).mean())
    avg_loss = down.groupby(x['stock_code']).transform(lambda s: s.rolling(window=10).mean())
    rs = avg_gain / avg_loss
    x['rsi'] = 100 - (100 / (1 + rs))

    # 添加布林带指标
    rolling_mean = x.groupby('stock_code')['close_price'].transform(lambda s: s.rolling(window=10).mean())
    rolling_std = x.groupby('stock_code')['close_price'].transform(lambda s: s.rolling(window=10).std())
    x['bollinger_upper'] = rolling_mean + (rolling_std * 2)
    x['bollinger_lower'] = rolling_mean - (rolling_std * 2)
    x['bollinger_width'] = x['bollinger_upper'] - x['bollinger_lower']

    # 特征标准化
    features_to_scale = ['ma_change_rate', 'double_ma_rate', 'double_v_rate', 'momentum',
                         'mean_reversion', 'volume_ratio', 'macd', 'macd_hist', 'rsi',
                         'bollinger_width']
    for feature in features_to_scale:
        mean = x[feature].mean()
        std = x[feature].std()
        x[feature] = (x[feature] - mean) / std

    # 处理异常值（用中位数填充）
    x = x.fillna(x.median(numeric_only=True))

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
    df_res['momentum'] = (df_res['close_price'].shift(1) - df_res['close_price'].shift(11)) / df_res[
        'close_price'].shift(11)
    df_res['mean_reversion'] = (df_res['close_price'].iloc[-1] - df_res['close_price'].rolling(window=20).mean().iloc[
        -1]) / df_res['close_price'].rolling(window=20).std().iloc[-1]
    df_res['volume_ratio'] = df_res['period_volume'].iloc[-1] / df_res['period_volume'].rolling(window=10).mean().iloc[
        -1]

    step_col = []
    for i in range(SEQUENCE_LENGTH, 0, -1):
        for col in ['open_price', 'high_price', 'low_price', 'close_price', 'period_volume',
                    'ma_change_rate', 'ma_change_rate_rank', 'double_ma_rate', 'double_v_rate',
                    'double_ma_rate_rank', 'volume_ts_15', 'momentum', 'mean_reversion', 'volume_ratio']:
            _step_col = '{}{}'.format(col, i)
            step_col.append(_step_col)
            df_res[_step_col] = df_res.groupby('stock_code')[col].shift(i).tolist()

    df_res = df_res.dropna()
    print(df_res['datetime'].max())
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
