from functools import reduce

import requests
import json
import timeit
import io
import os
import pandas as pd
import sqlalchemy
import tushare as ts
import akshare as ak


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

    df.columns = ['stock_code', 'introduction', 'main_business', 'employees', 'business_scope']
    return df


def get_company_info_from_akshare(symbols):
    res = []
    for s in symbols:
        try:
            stock_profile_cninfo_df = ak.stock_profile_cninfo(symbol=s[2:])
            res.append(stock_profile_cninfo_df)
        except Exception:
            continue
    return pd.concat(res)


def get_basic_info(export_path, database, table):
    engine = sqlalchemy.create_engine('sqlite:///{}'.format(os.path.join(export_path, database)))

    df = pd.read_sql_table('company_info_basic', engine)
    df['indices'] = df['indices'].fillna('-1')
    indices = df['indices'].unique().tolist()
    f = lambda x, y: x + y
    indices = list(set(list(reduce(f, [i.split(',') for i in indices]))))
    indices = {k: i for i, k in enumerate(indices)}
    df['indices'] = df['indices'].apply(lambda x: x.split(','))
    df['indices'] = df['indices'].apply(lambda x: '-'.join([str(indices[i]) for i in x]))

    df['industry'] = df['industry'].fillna('01')
    industry = df['industry'].unique().tolist()
    industry = {k: i for i, k in enumerate(industry)}
    df['industry'] = df['industry'].apply(lambda x: industry[x])
    df[['stock_code', 'indices', 'industry']].to_sql(table, engine, if_exists='replace')

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

    # tushare 数据
    df_company_info = get_company_info_from_tushare()
    df = pd.merge(df_symbols, df_company_info, on='stock_code')
    export_path = './database'
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database')
    data_to_sqlite(database_path, 'StockKing.db', 'company_info', df)

    # akshare 数据
    df_company_info_basic = get_company_info_from_akshare(df_symbols['stock_code'])
    df_symbols['code'] = df_symbols['stock_code'].apply(lambda x: x[2:])
    df = pd.merge(df_symbols, df_company_info_basic, left_on='code', right_on=['A股代码'])
    df = df[['name', 'stock_code', '曾用简称', 'A股简称', '入选指数', '所属行业', '成立日期', '上市日期',
             '官方网站', '电子邮箱', '主营业务', '经营范围', '机构简介']]

    df.columns = ['name', 'stock_code', 'used_abbr', 'abbr', 'indices', 'industry', 'setup_date', 'listing_date',
                  'website', 'email', 'main_business', 'business_scope', 'introduction']
    data_to_sqlite(database_path, 'StockKing.db', 'company_info_basic', df)

    get_basic_info(database_path, 'StockKing.db', 'emb_info')
