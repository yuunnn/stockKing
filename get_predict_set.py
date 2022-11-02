import numpy as np
import pandas as pd
import os
import sqlalchemy
import warnings

from config import SEQUENCE_LENGTH

warnings.filterwarnings("ignore")


def get_data(_data, output_file, basic_info=None):
    with open(output_file, 'w') as f:
        for sc in _data['stock_code'].unique():
            tmp = _data[_data['stock_code'] == sc].sort_values(by='datetime').reset_index().drop('index', axis=1)
            try:
                _step = np.array(tmp.loc[range(SEQUENCE_LENGTH),
                                         ['open_price', 'high_price', 'low_price', 'close_price', 'period_volume']]) \
                    .reshape(1, -1).squeeze()
            except KeyError:
                continue
            if basic_info is not None:
                emb = basic_info.loc[basic_info['stock_code'] == sc][['indices', 'industry']].astype(str)
                if len(emb) == 0:
                    continue
                _step = np.append(np.append(_step, [sc]), emb)
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
    df = df[df['datetime'].isin(sorted(df['datetime'].unique())[-SEQUENCE_LENGTH:])]
    emb_info = pd.read_sql_table('emb_info', engine)
    get_data(df, './predictset/latest.csv', emb_info)
