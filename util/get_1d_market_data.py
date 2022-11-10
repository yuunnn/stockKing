import tushare as ts
import os
import json
import pandas as pd
import datetime as dt


def code_util(x):
    _code, _exchange = x.split('.')
    return _exchange.lower() + str(_code)


def GetDataFromTushare(_symbol, today, fpath):
    pro = ts.pro_api("7e0b71553d2355108c8c429dfe48bd42f20fa82a80d08b92b1128426")
    df = pro.daily(trade_date=today)
    df['ts_code'] = df['ts_code'].apply(code_util)
    df = df[df['ts_code'].isin(_symbol['Symbol'])]
    for index, row in df.iterrows():
        ncode = row['ts_code']
        # sp = code.index(".")
        # ncode = code[sp + 1:].lower() + code[:sp]
        trade_date = str(row['trade_date'])
        date = trade_date[0:4] + '-' + trade_date[4:6] + '-' + trade_date[6:8]
        topen = row['open']
        thigh = row['high']
        tlow = row['low']
        tclose = row['close']
        tvolume = row['vol']
        tamount = row['amount']
        tchange = row['change']

        value = [ncode, date, topen, tlow, thigh, tclose, tvolume, tamount, tchange ]
        names = ['symbol', 'date', 'open', 'low', 'high', 'close', 'volume', 'amount', 'change']

        fname = fpath + '/day_csv/' + ncode + ".csv"
        if not os.path.exists(fname):
            fo = open(fname, "w+")
            fo.write(",".join(names) + "\n")
            valueline = ",".join([str(i) for i in value])
            fo.write(valueline + "\n")
            fo.close()
        else:
            fo = open(fname, "a")
            valueline = ",".join([str(i) for i in value])
            fo.write(valueline + "\n")
            fo.close()


if __name__ == "__main__":
    file = open('../database/symbols.json', encoding='utf-8').readlines()
    symbols = json.loads(file[0])
    df_symbols = pd.DataFrame(symbols)

    end = dt.date(2022,11,8)
    delta = dt.timedelta(days=1)
    length = 2
    for i in range(length):
        date = end.strftime('%Y%m%d')
        GetDataFromTushare(df_symbols, date, '..')
        end -= delta
    # python qlib-main/scripts/dump_bin.py dump_update --csv_path  data/day_csv --qlib_dir ~/.qlib/qlib_data/cn_data --include_fields  open,close,high,low,volume,factor