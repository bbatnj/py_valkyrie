import os

import pandas as pd
import numpy as np
import pandas_market_calendars as mcal

from valkyrie.tools import ymd_dir, df_col2str, cpu_at_80
from valkyrie.securities import stocks_good_dvd
from valkyrie.data import resample_bid_ask,check_hist_ib_data_integrity, MONTHLY_DVD_STOCKS
from valkyrie.quants import calc_current_yield

from multiprocessing import Pool
from functools import partial


def resample_calc_cy(date, stocks, outpath):
    print(f'working for {date}')
    err_stocks = []
    for stock in stocks:
        try:
            df = resample_bid_ask(date, [stock], 'd:/valkyrie/data/raw/hist_ib/bid_ask')
            dvd_freq = 'M' if stock in MONTHLY_DVD_STOCKS else 'Q'
            df = calc_current_yield(df, dvd_freq)
            df = df_col2str(df)
            path = f'{outpath}/{ymd_dir(date)}'
            os.makedirs(path, exist_ok=True)
            df.to_hdf(f'{path}/{stock}.h5', key='data')
        except Exception as e:
            err_stocks.append(stock)

    if err_stocks:
        print(f'error resampling for {stock} on {date}')
    if len(err_stocks) == len(stocks):
        print(f'no data for resampling on {date}')

if __name__ == "__main__":
    nyse = mcal.get_calendar('NYSE')
    sdate, edate = '20200601', '20210630'
    dates = [date for date in nyse.schedule(start_date=sdate, end_date=edate).index]
    #stocks = stocks_good_dvd()
    stocks = ['PFF']
    name = 'good_dvd'

    # f_resample_calc_cy(dates[0])
    # raw_data_path = "d:/valkyrie/data/raw/hist_ib"
    # f_check_hist_ib_data_integrity = partial(check_hist_ib_data_integrity, stocks=stocks, path=raw_data_path)
    # with Pool(cpu_at_80()) as pool:
    #     pool.map(f_check_hist_ib_data_integrity(dates))

    outpath = f'd:/valkyrie/data/resampled/hist_ib_raw_md/{name}'
    f_resample_calc_cy = partial(resample_calc_cy, stocks=stocks, outpath=outpath)
    with Pool(cpu_at_80()) as pool:
        pool.map(f_resample_calc_cy, dates)



