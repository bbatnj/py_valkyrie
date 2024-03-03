# stk_stream_load_demo.py:
# The demo script show how to use stk_stream
from __future__ import print_function
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('/mnt/sda/NAS/Release/shennong/stk/0.0.3/20230905-230712-ea77dfb0/python')

import shennong.utils.symbol as sn_symbol
from shennong.stk import stream
from shennong.utils import symbol
import datetime

EXCHANGE='.BNF'

PRODUCT='bina_future'

REGION = 'gb'

exchange_type = 1
h5_load_dir="/mnt/sda/NAS/Crypto/AllData"

freq='tick'

out_dir_root = '/mnt/sda/NAS/ShareFolder/bb/data/bin'

from numba import jit
import numpy as np
import os

@jit
def parse_bin_df(df):
    df['server_recv_time'] = (df['server_recv_time'] * 1e6).astype(np.int64)

    ts_array = df['server_recv_time'].values
    min_dt = 1;
    for i in np.arange(1, df.shape[0]):
        if (ts_array[i] - ts_array[i-1] < min_dt):
            ts_array[i] = ts_array[i-1] + min_dt

    df['ts'] = df['server_recv_time'].apply(pd.Timestamp)
    df.set_index('ts', inplace=True)
    df['side'] = df.eval('-2 * side + 3') #2-> -1, 1->1

    return df

def run_for_a_day(date : str) :
    start_datetime = f'{date} 00:00:00'
    end_datetime = f'{date} 23:59:59'
    symbol = 'BDM.BTC.USDT.FP'
    symbol2code = {
        'BDM.BTC.USDT.FP' : 'BTCUSDT',
        'BDM.BTC.BUSD.FP' : 'BTCBUSD',
        'BDM.ETH.USDT.FP' : 'ETHUSDT',
        'BDM.XRP.USDT.FP' : 'XRPUSDT'
    }

    out_dir = f'{out_dir_root}/{symbol}/{date}'
    os.system(f'mkdir -p {out_dir}')

    code_list = [symbol2code[symbol]]
    symbol_list = [x+EXCHANGE for x in code_list]
    symbol = symbol_list[0]
    # load depth data into a dictionary
    print('load quote:')
    df_quote = stream.load(start_datetime = start_datetime, end_datetime = end_datetime,
                           key_group_name = 'raw_'+REGION+'_' +PRODUCT+'_' +'quote', 
                           symbol_list = symbol_list,
                           region=REGION, product=PRODUCT, freq=freq,
                           load_root = h5_load_dir, verbose=True)[symbol]


    #print('quote:')
    quote_columns = 'exchange_time ask_amount ask_price bid_price bid_amount'.split()
    df_quote = df_quote[quote_columns]
    df_quote.rename({
        'exchange_time':'exch_ts_ms',
        'ask_amount' : 'aq',
        'bid_amount' : 'bq',
        'bid_price'  : 'bpx',
        'ask_price'  : 'apx',
                    }, axis=1, inplace=True)
    #    print(df_quote)
    df_quote.to_parquet(path=f'{out_dir}/quote.parquet', engine='auto')#, compression='gzip')

#    print('load depth:')
#    key_group_name = 'raw_'+REGION+'_' +PRODUCT+'_' +'depth'
#    print('key group name:',key_group_name)
#     df_depth = stream.load(start_datetime = start_datetime, end_datetime = end_datetime,
#                            key_group_name = 'raw_'+REGION+'_' +PRODUCT+'_' +'depth',
#                            symbol_list = symbol_list,
#                            region=REGION, product=PRODUCT, freq=freq,
#                            load_root = h5_load_dir, verbose=True)[symbol]
#    print('depth:')
#    print(df_depth)
#    df_depth.to_parquet(path=f'{out_dir}/df_depth.parquet.gzip', engine='auto', compression='gzip')

    print('load transaction:')
    df_trx = stream.load(start_datetime = start_datetime, end_datetime = end_datetime,
                           key_group_name = 'raw_'+REGION+'_' + PRODUCT + '_' +'transaction', 
                           symbol_list = symbol_list,
                           region=REGION, product=PRODUCT, freq=freq,
                           load_root = h5_load_dir, verbose=True)[symbol]


    trx_columns = 'exchange_time side price volume'.split()
    df_trx = df_trx[trx_columns]
    df_trx['side'] = df_trx.eval('-2 * side + 3')
    df_trx.rename({'exchange_time':'exch_ts_ms',
                   'volume' : 'qty',
                   'price'  : 'px'
                   }, axis = 1, inplace=True)
    #print(df_trx)
    df_trx.to_parquet(path=f'{out_dir}/transcation.parquet', engine='auto', compression='gzip')
