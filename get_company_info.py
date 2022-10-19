import requests
import json
import timeit
import io
import os
import pandas as pd
import sqlalchemy
import tushare as ts

def get_company_info_from_tushare():
    def code_util(x):
        _code, _exchange = x.split('.')
        return _exchange.lower() + str(_code)

    pro = ts.pro_api('7e0b71553d2355108c8c429dfe48bd42f20fa82a80d08b92b1128426')

    # 拉取数据
    df = pro.stock_company(**{
        "ts_code": "",
        "exchange": "",
        "status": "",
        "limit": "",
        "offset": ""
    }, fields=[
        "ts_code",
        "introduction",
        "main_business",
        "business_scope",
        "employees"
    ])
    df['ts_code'] = df['ts_code'].apply(code_util)

    df.columns = ['stock_code', 'introduction', 'main_business', 'business_scope', 'employees']
    return df


def data_to_sqlite(export_path, database, table, all_quotes):
    engine = sqlalchemy.create_engine('sqlite:///{}'.format(os.path.join(export_path, database)))
    _df = pd.DataFrame(all_quotes)
    _df.to_sql(table, engine, if_exists='replace')


if __name__ == "__main__":
    file = open('./database/symbols.json', encoding='utf-8').readlines()
    symbols = json.loads(file[0])
    df_symbols = pd.DataFrame(symbols)
    df_symbols.columns = ['stock_code', 'name']
    df_company_info = get_company_info_from_tushare()
    df = pd.merge(df_symbols, df_company_info, on='stock_code')
    export_path = './database'
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database')
    data_to_sqlite(database_path, 'StockKing.db', 'company_info', df)