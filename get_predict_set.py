import numpy as np
import pandas as pd
import os
import sqlalchemy
import warnings

from config import SEQUENCE_LENGTH

warnings.filterwarnings("ignore")


def get_data(_data, output_file):
    with open(output_file, 'a') as f:
        for sc in _data['stock_code'].unique():
            tmp = _data[_data['stock_code'] == sc].sort_values(by='datetime').reset_index().drop('index', axis=1)
            try:
                _step = np.array(tmp.loc[range(20), ['open_price', 'high_price', 'low_price',
                                                     'close_price', 'period_volume']]).reshape(1, -1).squeeze()
            except KeyError:
                continue
            _step = np.append(_step, [sc])
            f.write(','.join(_step))
            f.write('\n')
    return


if __name__ == '__main__':
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database')
    engine = sqlalchemy.create_engine('sqlite:///{}'.format(os.path.join(database_path, 'StockKing.db')))
    df = pd.read_sql_table('ma60m', engine)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[df['datetime'].isin(sorted(df['datetime'].unique())[-SEQUENCE_LENGTH:])]
    get_data(df, './predictset/latest.csv')
