import requests
import json
import timeit
import io
import os
import pandas as pd
import sqlalchemy


def load_all_quote_symbol():
    print("load_all_quote_symbol start..." + "\n")
    start = timeit.default_timer()
    all_quotes = []
    all_quotes_url = 'http://money.finance.sina.com.cn/d/api/openapi_proxy.php'
    try:
        count = 1
        while count < 100:
            para_val = '[["hq","hs_a","",0,' + str(count) + ',500]]'
            r_params = {'__s': para_val}
            r = requests.get(all_quotes_url, params=r_params)
            if len(r.json()[0]['items']) == 0:
                break
            for item in r.json()[0]['items']:
                quote = {}
                code = item[0]
                name = item[2]
                # convert quote code
                if code.find('sh') > -1:
                    code = 'sh' + code[2:]
                elif code.find('sz') > -1:
                    code = 'sz' + code[2:]
                if code.startswith('sz30') or code.startswith('sh68') or code.startswith('bj'):
                    continue
                # convert quote code end
                quote['Symbol'] = code
                quote['Name'] = name
                all_quotes.append(quote)
            count += 1
    except Exception as e:
        print("Error: Failed to load all stock symbol..." + "\n")
        print(e)
    print("load_all_quote_symbol end... time cost: " + str(round(timeit.default_timer() - start)) + "s" + "\n")
    print("total " + str(len(all_quotes)) + " quotes are loaded..." + "\n")
    return all_quotes


def data_export(export_path, all_quotes, file_name):
    start = timeit.default_timer()
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    if all_quotes is None or len(all_quotes) == 0:
        print("no data to export...\n")
    print("start export to JSON file...\n")
    f = io.open(export_path + '/' + file_name + '.json', 'w', encoding='utf-8')
    json.dump(all_quotes, f, ensure_ascii=False)
    print("export is complete... time cost: " + str(round(timeit.default_timer() - start)) + "s" + "\n")


def data_to_sqlite(export_path, database, table, all_quotes):
    engine = sqlalchemy.create_engine('sqlite:///{}'.format(os.path.join(export_path, database)))
    df = pd.DataFrame(all_quotes)
    df.columns = ['stock_code', 'name']
    df.to_sql(table, engine, if_exists='replace')


if __name__ == '__main__':
    all_quotes = load_all_quote_symbol()
    export_path = './database'
    data_export(export_path, all_quotes, file_name='symbols')
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database')
    # data_to_sqlite(database_path, 'StockKing.db', 'company_info', all_quotes)
