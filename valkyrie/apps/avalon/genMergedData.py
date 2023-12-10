import os
from multiprocessing import Pool
from functools import partial

import pandas as pd
import numpy as np
import pandas_market_calendars as mcal

from valkyrie.tools import ymd_dir, ymd, cpu_at_80, calc_diff_days
from valkyrie.securities import stocks_good_dvd, ROOT_PATH

def merge_bidask_and_trades(date, ticker, data_src, trade_offset = '1s'):
    ymd_path = ymd_dir(date)
    df_bidask = pd.read_hdf(f'{ROOT_PATH}/raw/{data_src}/bid_ask/{ymd_path}/{ticker}.h5')
    try:
        df_trades = pd.read_hdf(f'{ROOT_PATH}/raw/{data_src}/trade/{ymd_path}/{ticker}.h5')
        df_trades.index += pd.Timedelta(trade_offset)
        df_merged = pd.concat([df_bidask, df_trades]).sort_index()
    except Exception as e:
        print(e)
        df_merged = df_bidask
        df_merged['price'], df_merged['size'], df_merged['exchange'] = np.nan, np.nan, ''

    df_merged.rename({
        'priceBid':'bid',
        'priceAsk':'ask',
        'sizeBid' :'bz',
        'sizeAsk' :'az',
        'price'   :'trd_px',
        'size'    :'trd_sz',
        'exchange':'exch'}, axis=1, inplace=True)

    for c in ['bid','ask','bz','az']:
        df_merged[c] = df_merged[c].ffill()

    return df_merged

def gen_merged_data(date, stocks, name, data_src):
    print(f'merging data for {date}')
    df_merged = []
    for stock in stocks:
        exception_str = f"error for {stock} on {date}"
        try:
            df = merge_bidask_and_trades(date, stock, data_src=data_src, trade_offset = '1s')
            df['ymd'] = df.index.map(lambda t: ymd(t))
            date_ymd = ymd(date)
            df = df.query(f'ymd == "{date_ymd}"').copy()
            if df.shape == 0:
                raise Exception(" due to zero shape")

            df_merged.append(df)
        except Exception as e:
            print(exception_str +' ' + str(e))

    df_merged = pd.concat(df_merged).sort_index()

    for c in 'bid ask bz az trd_px trd_sz'.split():
        df_merged[c] = df_merged[c].astype(float)

    for c in 'ticker exch specialConditions'.split():
        df_merged[c] = df_merged[c].astype(str)

    outpath = f'{ROOT_PATH}/merged/{data_src}/{ymd_dir(date)}/'
    os.makedirs(outpath, exist_ok=True)
    df_merged.to_hdf(f'{outpath}/{name}.h5', key='data')
    return

if __name__ == "__main__":
    #name = 'test'
    #df_sec = pd.read_csv(f'{ROOT_PATH}/universe/{name}_sec_file.csv')
    #stocks = df_sec['ticker']

    name = 'stocks_good_dvd'
    stocks = stocks_good_dvd()
    stocks.append('PFF')

    sdate, edate = '20200601', '20211201'
    nyse = mcal.get_calendar('NYSE')
    data_src ='hist_ib'

    dates = list(nyse.schedule(start_date=sdate, end_date=edate).index)
    f_gen_merged_data = partial(gen_merged_data, stocks=stocks, name=name, data_src='hist_ib')

    with Pool(cpu_at_80()) as pool:
        pool.map(f_gen_merged_data, dates)
