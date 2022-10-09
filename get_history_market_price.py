import json
import requests
import pandas as pd
import os

import sqlalchemy
import random


def requests_headers():
    """
    Random UA  for every request && Use cookie to scan
    """
    user_agent = [
        'Mozilla/5.0 (Windows; U; Win98; en-US; rv:1.8.1) Gecko/20061010 Firefox/2.0',
        'Mozilla/5.0 (Windows; U; Windows NT 5.0; en-US) AppleWebKit/532.0 (KHTML, like Gecko) Chrome/3.0.195.6 '
        'Safari/532.0',
        'Mozilla/5.0 (Windows; U; Windows NT 5.1 ; x64; en-US; rv:1.9.1b2pre) Gecko/20081026 Firefox/3.1b2pre',
        'Opera/10.60 (Windows NT 5.1; U; zh-cn) Presto/2.6.30 Version/10.60',
        'Mozilla/5.0 (Windows; U; Windows NT 5.1; ; rv:1.9.0.14) Gecko/2009082707 Firefox/3.0.14',
        'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.106 Safari/537.36',
        'Mozilla/5.0 (Windows; U; Windows NT 6.0; fr; rv:1.9.2.4) Gecko/20100523 Firefox/3.6.4 ( .NET CLR 3.5.30729)',
        'Mozilla/5.0 (Windows; U; Windows NT 6.0; fr-FR) AppleWebKit/533.18.1 (KHTML, like Gecko) Version/5.0.2 '
        'Safari/533.18.5',
        'Mozilla/5.0 (compatible; Bytespider; https://zhanzhang.toutiao.com/) AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/70.0.0.0 Safari/537.36']
    UA = random.choice(user_agent)
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'User-Agent': UA, 'Upgrade-Insecure-Requests': '1', 'Connection': 'keep-alive', 'Cache-Control': 'max-age=0',
        'Accept-Encoding': 'gzip, deflate, sdch', 'Accept-Language': 'zh-CN,zh;q=0.8',
        "Referer": "http://www.baidu.com/link?url=www.so.com&url=www.soso.com&&url=www.sogou.com"}
    return headers


def get_market_price(_api, _symbols, _scale):
    k = 0
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database')
    engine = sqlalchemy.create_engine('sqlite:///{}'.format(os.path.join(database_path, 'StockKing.db')))
    for symbol in _symbols:
        symbol = symbol['Symbol']
        if 'bj' in symbol:
            continue
        headers = requests_headers()
        r = requests.get(_api.format(symbol, _scale), headers=headers)
        stock_ma_info = pd.DataFrame(r.json()).rename(columns={
            'day': 'datetime',
            'open': 'open_price',
            'high': 'high_price',
            'low': 'low_price',
            'close': 'close_price',
            'volume': 'period_volume'
        })
        stock_ma_info['stock_code'] = symbol
        stock_ma_info['open_price'] = stock_ma_info['open_price'].astype('double')
        stock_ma_info['high_price'] = stock_ma_info['high_price'].astype('double')
        stock_ma_info['low_price'] = stock_ma_info['low_price'].astype('double')
        stock_ma_info['close_price'] = stock_ma_info['close_price'].astype('double')
        stock_ma_info['period_volume'] = stock_ma_info['period_volume'].astype('int64')

        data_to_sqlite('ma{}m'.format(scale), stock_ma_info, engine)
        print(k, symbol)
        k += 1


def data_to_sqlite(table, _data, engine):
    _data.to_sql(table, engine, if_exists='append', index=False)


if __name__ == '__main__':
    api_url = "https://quotes.sina.cn/cn/api/json_v2.php/" \
              "CN_MarketDataService.getKLineData?symbol={}&scale={}&ma=yes&datalen=2023"
    file = open('./database/symbols.json', encoding='utf-8').readlines()
    symbols = json.loads(file[0])
    scale = 60
    get_market_price(api_url, symbols, scale)
