import os
import copy
from itertools import product as cartesian_product

import pandas as pd
import numpy as np
import torch

from numba import jit

from valkyrie.quants.utils import add_fwd_ret_by_n_, toDf32_


data_root_dir = '/home/bb/data/BDM'

def resample_bdm_trade_tob(date, instr, freq):

    def df2float32(df):
        for c in df:
            if df.dtypes[c] == 'float64' or df.dtypes[c] == np.float64:
                df[c] = df[c].astype(np.float32)

    #trx
    df_trx = pd.read_parquet(f'{data_root_dir}/{instr}/tick/{date}/transcation.parquet')
    df2float32(df_trx)

    def resample_trade(df, freq):
        col2agg = {'px':['first','last', 'max', 'min'], 'qty':['sum']}
        cols = col2agg.keys()
        df_res = df[cols].resample(freq, label='right').agg(col2agg)
        df_res.columns = ['_'.join(c) for c in df_res.columns]

        return df_res

    def resample_quote(df, freq):
        agg_funcs = ['first','last', 'max', 'min']
        cols = 'aq	apx	bpx	bq'.split()
        df_res = df[cols].resample(freq, label='right').agg(agg_funcs)
        df_res.columns = ['_'.join(c) for c in df_res.columns]


        return df_res

    df_buy  = resample_trade(df_trx.query('side > 0.0'), freq)
    df_buy.columns = ['buy_' + c for c in df_buy.columns]

    df_sell = resample_trade(df_trx.query('side < 0.0'), freq)
    df_sell.columns = ['sell_' + c for c in df_sell.columns]

    #quote
    df_quote = pd.read_parquet(f'{data_root_dir}/{instr}/tick/{date}/quote.parquet')
    df2float32(df_quote)
    df_tob = resample_quote(df_quote, freq)

    #merged
    #df_res = pd.concat([df_tob, df_buy, df_sell], axis = 1, join='outer')
    df_res = df_tob.join(df_buy, how='outer').join(df_sell, how='outer')

    return df_res


def save_sampled_df(date, freq, data_root_dir, instr):
    df_res = resample_bdm_trade_tob(date, instr, freq)
    dst_dir = f'{data_root_dir}/{instr}/sampled/{freq}/{date}'
    os.system(f'mkdir -p {dst_dir}')
    dist_file = f'{dst_dir}/df_trd_tob_freq_{freq}.parquet'
    df_res.to_parquet(dist_file, compression='gzip')
    print(f'saved sampled df as {dist_file}')


def read_sampled_df(sdate, edate, freq, data_root_dir, instr):
    date_list = [str(e.date()) for e in pd.date_range(sdate, edate)]

    df_res = []
    for dt in date_list:
        src_dir = f'{data_root_dir}/{instr}/sampled/{freq}/{dt}'
        df_res.append(pd.read_parquet(f'{src_dir}/df_trd_tob_freq_{freq}.parquet'))
    df_res = pd.concat(df_res, axis = 0).sort_index()

    # cols = 'buy_px_first buy_px_last buy_px_max buy_px_min sell_px_first sell_px_last sell_px_max sell_px_min'
    # for c in cols.split():
    #     origin_c = c.replace('buy_px', 'apx').replace('sell_px', 'bpx')
    #     df_res[c].fillna(df_res[origin_c], inplace=True)

    df_res['bpx_last'].ffill(inplace=True)
    df_res['apx_last'].ffill(inplace=True)
    df_res['aq_last'].ffill(inplace=True)
    df_res['bq_last'].ffill(inplace=True)
    #
    for c in df_res:
        if c.startswith('bpx_'):
            df_res[c].fillna(df_res['bpx_last'], inplace=True)
        elif c.startswith('apx_'):
            df_res[c].fillna(df_res['apx_last'], inplace=True)
        elif c.startswith('aq_'):
            df_res[c].fillna(df_res['aq_last'], inplace=True)
        elif c.startswith('bq_'):
            df_res[c].fillna(df_res['bq_last'], inplace=True)

    df_res['buy_px_last'].ffill(inplace=True)
    df_res['sell_px_last'].ffill(inplace=True)

    for c in df_res:
        if c.startswith('buy_px_'):
            df_res[c].fillna(df_res['buy_px_last'], inplace=True)
        elif c.startswith('sell_px_'):
            df_res[c].fillna(df_res['sell_px_last'], inplace=True)
        elif 'qty_' in c:
            df_res[c].fillna(0.0, inplace=True)

    #fill trade px
    #df_res['buy_qty_sum'].fillna(0.0, inplace = True)
    #df_res['sell_qty_sum'].fillna(0.0, inplace = True)
    # cols = 'buy_px_first buy_px_last buy_px_max buy_px_min sell_px_first sell_px_last sell_px_max sell_px_min'
    # for c in cols.split():
    #     origin_c = c.replace('buy_px', 'apx').replace('sell_px', 'bpx')
    #     df_res[c].fillna(df_res[origin_c], inplace=True)
    return df_res


class DataMgr:
    def __init__(self, sdate, edate, freq, data_root_dir, instrs, ret_n_s):
        self.instr2df = {}

        for instr in instrs:
            df = read_sampled_df(sdate, edate, freq, data_root_dir, instr)
            df['mid_last']  = df.eval('0.5 * (bpx_last + apx_last)')
            df['sprd_last'] = df.eval('apx_last - bpx_last')
            df['sprd2_last']= df['sprd_last'] * df['sprd_last']
            add_fwd_ret_by_n_(df, 'log', 'mid_last', ret_n_s, 'sprd2_last')
            self.instr2df[instr] = df.dropna()

    def get(self, instr):
        return self.instr2df[instr]

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    sdate, edate = '2023-01-01', '2023-11-30'
    instrs = ['BDM.BTC.USDT.FP', 'BDM.BTC.BUSD.FP', 'BDM.ETH.USDT.FP', 'BDM.SOL.USDT.FP', 'BDM.XRP.USDT.FP']
    freqs = ['100ms']#['50ms', '1s']


    #########################################################
    #DataMgr
    #########################################################
    # ret_n_s = [30]
    # data_mgr = DataMgr(sdate, edate, freq, '/home/bb/data/BDM', instrs = instrs, ret_n_s = ret_n_s)
    # df_res = data_mgr.get(instr)
    # df2t2 = Df2T2(df_res, M = 3, xcols = ['buy_qty_sum', 'sell_qty_sum'], ycol = 'mid_last_ret_30_n',
    #               wcol = 'wgt_mid_last_ret_30_n', mul = 2, dtype = torch.float32, device='cpu')
    #
    # x = df2t2[0]

    #########################################################
    # loading
    #########################################################
    #df_res = read_sampled_df(sdate, edate, freq, '/home/bb/data/BDM', instr = instrs)


    #########################################################
    # saving
    #########################################################
    for freq, instr in cartesian_product(freqs, instrs):
        print(f'resampling for {instr} @ {freq}')
        date_list = [str(e.date()) for e in pd.date_range(sdate, edate)]

        f_save_sampled_df = partial(save_sampled_df,
                                  freq = freq, data_root_dir = '/home/bb/data/BDM', instr = instr)

        f_save_sampled_df(date_list[0])
        #with ProcessPoolExecutor(6) as executor:
        #    executor.map(f_save_sampled_df, date_list)

    exit(0)
