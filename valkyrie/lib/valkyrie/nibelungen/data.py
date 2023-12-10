data_root_dir = '/home/bb/data/BDM'
instr = 'BDM.BTC.USDT.FP'
date = '2023-11-01'

import os

import pandas as pd
import numpy as np
import torch

from numba import jit

from valkyrie.quants.utils import add_fwd_ret_by_n_, toDf32_

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
    df_res.to_parquet(f'{dst_dir}/df_trd_tob_freq_{freq}.parquet', compression='gzip')
    print(f'saved sampled df for {date}')


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


import torch
from torch.utils.data import Dataset, DataLoader


class Df2T2(Dataset):
    def __init__(self, df: pd.DataFrame, M, xcols, ycol, wcol=None, mul=2, dtype=torch.float32, device='cpu', yscaler = 1e4):
        super().__init__()

        self.M = M
        self.W = M
        self.n_features = len(xcols)

        final_x_cols = xcols.copy()  # we do not wanna change xcols

        dfs = [df[xcols]]
        for freq_mul_index in np.arange(1, self.M):
            freq_mul = mul ** freq_mul_index
            df_temp = df[xcols].rolling(freq_mul).agg('mean')
            df_temp.columns = [f'{c}_x{freq_mul:04d}' for c in xcols]
            dfs.append(df_temp)
            final_x_cols += list(df_temp.columns)

        # xcols must be sorted by feature then freq_mul
        final_x_cols = sorted(final_x_cols)

        dfs.append(df[[ycol]])  # y
        self.df = pd.concat(dfs, axis=1)  # x's

        prev_n = self.df.shape[0]
        self.df.dropna(inplace=True)

        n = self.df.shape[0]
        print(f'after dropping na {prev_n} -> {n}')

        weights = np.ones(n)
        if wcol:  # w
            weights = df[wcol].values

        weights = weights / np.sqrt(np.mean(weights * weights))

        X = torch.tensor(self.df[final_x_cols].values, dtype=dtype, device=device)
        # xcols must be sorted by feature then freq_mul
        X = X.T.view(self.n_features, self.M, self.df.shape[0]).contiguous()
        self.X = X
        # X.shape is now (n_features, M, n_samples)

        self.offset = self.W - 1
        self.yw = torch.tensor(np.c_[yscaler * self.df[ycol].values, weights], dtype=dtype, device=device)
        self.n = self.X.shape[-1] - self.W + 1

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[:,:,idx: idx+self.offset+1], self.yw[idx+self.offset]

    @jit
    def get_yw(self):
        n = self.n
        c = np.prod(self[0][0].shape)
        #X_sk = np.empty((n, c))
        y_sk = np.empty(n)
        w_sk = np.empty(n)
        for i in np.arange(n):
            z = self[i]
            #X_sk[i, :] = z[0].reshape(-1)
            y_sk[i] = z[1][0]
            w_sk[i] = z[1][1]
        return y_sk, w_sk

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    sdate, edate = '2023-01-01', '2023-01-31'
    instr = 'BDM.BTC.USDT.FP'
    freq = '1s'
    ret_n_s = [30]

    #########################################################
    #DataMgr
    #########################################################
    data_mgr = DataMgr(sdate, edate, freq, '/home/bb/data/BDM', instrs = ['BDM.BTC.USDT.FP'], ret_n_s = [30])
    df_res = data_mgr.get(instr)
    df2t2 = Df2T2(df_res, M = 3, xcols = ['buy_qty_sum', 'sell_qty_sum'], ycol = 'mid_last_ret_30_n',
                  wcol = 'wgt_mid_last_ret_30_n', mul = 2, dtype = torch.float32, device='cpu')

    x = df2t2[0]

    #########################################################
    # loading
    #########################################################
    #df_res = read_sampled_df(sdate, edate, freq, '/home/bb/data/BDM', instr = 'BDM.BTC.USDT.FP')

    #########################################################
    # saving
    #########################################################
    # f_save_sampled_df = partial(save_sampled_df,
    #                           freq = '1s', data_root_dir = '/home/bb/data/BDM', instr = 'BDM.BTC.USDT.FP')
    #
    # date_list = [str(e.date()) for e in pd.date_range(sdate, edate)]
    #
    # with ProcessPoolExecutor(16) as executor:
    #     executor.map(f_save_sampled_df, date_list)
