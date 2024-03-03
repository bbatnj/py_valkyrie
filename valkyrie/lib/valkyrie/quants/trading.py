import glob

import numpy as np
import pandas as pd
from typing import Tuple

from valkyrie.quants.linear_model import calc_auto_corr
from valkyrie.quants.utils import sharpe, win_ratio

from quants.time_average import calc_ts_ema, calc_ts_ems, calc_ems_vol
from tools import format_df_nums


def fifo_match(trades: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Takes a dataframe of trades and produces a dataframe of
    FIFO matches for entries/exits. Trades where we entered
    long have sign 1, trades where we entered short have sign -1.
    :param trades: a dataframe with the following columns:
        * side: 0 for long, 1 for short
        * quantity: quantity of the trade
        * price: price of the trade
        * fee: fee of the trade. payable.
        * receipt_time: time of receipt of the trade
    :return: a dataframe of FIFO-matched trades
    """
    trades = trades.copy()
    trades["sign"] = -(trades.side * 2 - 1)
    open_qty = 0
    open_trades = []
    res = []
    for i, t in trades.iterrows():
        if open_qty == 0:
            # No open trades, so we open one
            open_trades.append((i, t.receipt_time, t.price, t.quantity, t.sign))
            open_qty = t.quantity * t.sign
        else:
            # We have an open trade
            if t.sign * open_qty > 0:
                # This increases our exposure
                open_qty += t.quantity * t.sign
                open_trades.append((i, t.receipt_time, t.price, t.quantity, t.sign))
            else:
                # This decreases our exposure
                cur_qty = t.quantity
                while cur_qty > 0 and len(open_trades) > 0:
                    if open_trades[0][3] > cur_qty:
                        # The current trade reduces the exposure, but does not
                        # eliminate/reverse it
                        res.append(
                            (
                                open_trades[0][0],
                                open_trades[0][1],
                                open_trades[0][2],
                                cur_qty,
                                open_trades[0][4],
                                i,
                                t.receipt_time,
                                t.price,
                            )
                        )
                        open_trades[0] = (
                            open_trades[0][0],
                            open_trades[0][1],
                            open_trades[0][2],
                            open_trades[0][3] - cur_qty,
                            open_trades[0][4],
                        )
                        open_qty += cur_qty * t.sign
                        cur_qty = 0
                    else:
                        # The current trade at least eliminates the exposure of
                        # the first open trade
                        cur_qty -= open_trades[0][3]
                        res.append(
                            (
                                open_trades[0][0],
                                open_trades[0][1],
                                open_trades[0][2],
                                open_trades[0][3],
                                open_trades[0][4],
                                i,
                                t.receipt_time,
                                t.price,
                            )
                        )
                        open_qty -= open_trades[0][3] * open_trades[0][4]
                        open_trades.pop(0)
                if cur_qty > 0:
                    # The current trade reverses the exposure
                    open_trades.append((i, t.receipt_time, t.price, cur_qty, t.sign))
                    open_qty += cur_qty * t.sign
    res = pd.DataFrame(
        res,
        columns=[
            "entry_trade_id",
            "entry_time",
            "entry_price",
            "quantity",
            "sign",
            "exit_trade_id",
            "exit_time",
            "exit_price",
        ],
    )
    res["net_pnl"] = (res.exit_price - res.entry_price) * res.sign * res.quantity
    # account for fees
    res["fee_pnl"] = 0
    #buy_trade_frac = res.quantity / trades.loc[res.entry_trade_id].quantity.values
    #sell_trade_frac = res.quantity / trades.loc[res.exit_trade_id].quantity.values
    #res.fee_pnl -= trades.loc[res.entry_trade_id].fee.values * buy_trade_frac
    #res.fee_pnl -= trades.loc[res.exit_trade_id].fee.values * sell_trade_frac
    res["pnl"] = res.net_pnl + res.fee_pnl
    return res, open_trades

def calc_trade_markout(df_trades, df_mkt_px, Ts,
                       resampling_frequency='1s'):  # forward trade pnl marked at time of trade
    df_mkt_px = df_mkt_px.resample(rule=f'{resampling_frequency}').last()
    df_merged = pd.merge_asof(df_trades, df_mkt_px, left_index=True, right_index=True, direction='backward')
    # pnl_adj by formula
    df_merged['markout_adj'] = df_merged.eval("net_qty * (mid - price)")

    # Resample trades and sum up fee, signed quantity and pnl_adj
    agg_map = {k: "sum" for k in ["fee", "qty", "net_qty", "markout_adj"]}
    agg_map |= {'mid': 'last'}
    df_merged = df_merged.resample('1s').agg(agg_map)
    df_merged = df_merged.query('qty > 0').copy()

    # df_merged.ffill(inplace=True)  # otherwise fill NA with 0

    for T in Ts:
        if T >= 1e9:
            T = int(1e9)
            markout_pnl = 'total_markout'
        else:
            markout_pnl = f'markout_{T}s'
            offset = int(T / pd.Timedelta(resampling_frequency).total_seconds())

        df_merged['dpx'] = df_merged['mid'].shift(-offset) - df_merged['mid']
        df_merged[markout_pnl] = df_merged.eval('net_qty * dpx + markout_adj - fee')
    return df_merged.dropna(how='any', subset=[c for c in df_merged if 'markout' in c])

def calc_trading_pnl(df_trades, df_mkt_px, T=300, resampling_frequency='1s'): #pnl marketd at time of px change
    if T >= 1e9:
        T = int(1e9)
        pnl_col = 'total_pnl'
    else:
        pnl_col = int(T * pd.Timedelta(resampling_frequency).total_seconds())
        pnl_col = f'trading_pnl_{pnl_col}s'

    df_mkt_px = df_mkt_px.resample(rule=f'{resampling_frequency}').last()
    df_merged = pd.merge_asof(df_trades, df_mkt_px, left_index=True, right_index=True, direction='backward')

    # pnl_adj by formula
    df_merged['pnl_adj'] = df_merged.eval("net_qty * (mid - price)")

    # Resample trades and sum up fee, signed quantity and pnl_adj
    df_merged = df_merged.resample('1s').sum()[["fee", "qty", "net_qty", "pnl_adj"]]

    # only mid and change in mid should be precisely evaluated at the end of each second
    df_merged['mid'] = df_mkt_px[df_mkt_px.index.isin(df_merged.index)]['mid'].bfill()
    df_merged['dpx'] = df_merged['mid'].diff()
    df_merged.fillna(0, inplace=True)  # otherwise fill NA with 0

    # Calculate the first term in the trading PnL (the second term is pnl_adj)
    df_merged['net_qty_rolling_sum'] = df_merged["net_qty"].rolling(f'{T}s', closed='left').sum()
    df_merged['rolling_sample_pnl'] = df_merged.eval("dpx * net_qty_rolling_sum")

    # Calculate rolling trading_pnl
    df_merged['rolling_trading_pnl'] = df_merged.eval('rolling_sample_pnl + pnl_adj - fee')
    df_merged[pnl_col] = df_merged['rolling_trading_pnl'].cumsum()
    return df_merged.fillna(0.0)



class SimPostTradeAnalysis:
    def __init__(self, ag6_symbol, sim_name, ev_path, local_sim = False, sample = '*', tactic_pattern = '*THOR*'):
        # df_stats
        df_stats_list = []
        if not local_sim:
            pattern = f'{ev_path}/{sim_name}/*/{sample}/*/stats/*/name={tactic_pattern}/*.parquet'
        else:
            pattern = f'{ev_path}/{sim_name}/stats/startDate*/{tactic_pattern}/*.parquet'

        ## adj for previous trunk pnl
        prev_trunk_end_pnl = 0.0
        for fn in sorted(glob.glob(pattern)):
            df_temp = pd.read_parquet(fn)
            df_temp['PnL'] += prev_trunk_end_pnl
            df_temp['PnL'] = df_temp['PnL'].ffill()
            df_temp['ts'] = df_temp['time']
            df_temp['dt'] = df_temp['ts'].diff().fillna(0.0)
            df_temp['time'] = df_temp['time'].apply(pd.Timestamp)
            df_temp['date'] = df_temp['time'].apply(lambda t: t.date())
            prev_trunk_end_pnl = df_temp['PnL'].iloc[-1]
            df_stats_list.append(df_temp)

        self.df_stats_list = df_stats_list
        df_stats = pd.concat(self.df_stats_list).sort_index()


        df_stats = df_stats.set_index('time').sort_index()
        self.df_stats = df_stats
        self.sdate, self.edate = self.df_stats['date'].values[0], self.df_stats['date'].values[-1]
        self.pos_corr_lags = [60, 300, 900, 1800]

        # df_trades
        if not local_sim:
            pattern = f'{ev_path}/{sim_name}/*/{sample}/*/matchedTrades/*/name={tactic_pattern}/*.parquet'
        else:
            pattern = f'{ev_path}/{sim_name}/matchedTrades/date=*/{tactic_pattern}/*.parquet'
        # #print(f'glob pattern = {pattern}')
        # df_trades = []
        # for fn in sorted(glob.glob(pattern)):
        #     try:
        #         df_trade = pd.read_parquet(fn)
        #         df_trade['time'] = df_trade['time'].apply(pd.Timestamp)
        #         df_trade['date'] = df_trade['time'].apply(lambda t: t.date())
        #         df_trade['dt'] = df_trade['time'].diff().apply(lambda t: t.total_seconds()).fillna(0.0)
        #         df_trade = df_trade.set_index('time').sort_index()
        #         side2sgn = {b'SELL': -1, b'BUY': 1}
        #         df_trade['side'] = df_trade['side'].apply(lambda x: side2sgn[x])
        #         # import pdb
        #         # pdb.set_trace()
        #         df_trade['net_qty'] = df_trade.eval('qty * side')
        #         df_trade['cum_net_qty'] = df_trade['net_qty'].cumsum()
        #         df_trades.append(df_trade)
        #     except Exception as e:
        #         print(f'error processing {fn}')

        df_trades = pd.concat([pd.read_parquet(fn) for fn in sorted(glob.glob(pattern))], axis = 0).sort_index()
        df_trades['time'] = df_trades['time'].apply(pd.Timestamp)
        df_trades = df_trades.sort_values('time')
        df_trades['dt'] = df_trades['time'].diff().apply(lambda t: t.total_seconds()).fillna(0.0)
        df_trades['date'] = df_trades['time'].apply(lambda t: t.date())
        df_trades = df_trades.set_index('time')

        side2sgn = {b'SELL': -1, b'BUY': 1}
        df_trades['side'] = df_trades['side'].apply(lambda x: side2sgn[x])
        df_trades['net_qty'] = df_trades.eval('qty * side')
        df_trades['cum_net_qty'] = df_trades['net_qty'].cumsum()
        self.df_trades = df_trades

        # df_order
        df_order = []
        if not local_sim:
            pattern = f'{ev_path}/{sim_name}/*/{sample}/*/orders/*/name={tactic_pattern}/*.parquet'
        else:
            pattern = f'{ev_path}/{sim_name}/orders/date=*/{tactic_pattern}/*.parquet'
        for fn in sorted(glob.glob(pattern)):
            df_order.append(pd.read_parquet(fn))
        df_order = pd.concat(df_order).sort_index()
        df_order['time'] = df_order['time'].apply(pd.Timestamp)
        df_order['createTime'] = df_order['createTime'].apply(pd.Timestamp)

        order_cols = ['time', 'handle', 'limitPrice', 'avgExecutedPrice', 'remainingQty',
                      'executedQty', 'pendingExecutedQty', 'side', 'type', 'timeInForce',
                      'done', 'tradeOriginalQty',
                      'doneReason', 'createTime', 'responseSuccess',
                      'responseOrderRejectReason']
        df_order = df_order[order_cols]
        self.df_order = df_order.set_index('time').sort_index()

        self.grid_freq_ms = 1000
        #mid_fld = RH:TODO
        #mid
        #self.df_mid = RH:::TODO

        #vol
        def calc_ems_vol(df_mid, mid_fld = 'mid', vol_hls = [60, 300, 1800], freq_ms = 1000, max_vol = np.inf):
            df_vol = df_mid.copy()
            freq_sec = freq_ms / 1e3
            max_ret = max_vol / np.sqrt(365 * 24 * 3600 / freq_sec) * 5.0
            df_vol['var_raw'] = (df_vol[mid_fld].shift(-1) / df_vol[mid_fld]).apply(np.log).clip(-max_ret, max_ret)
            df_vol['var_raw'] = df_vol['var_raw'].apply(np.square())
            df_vol['dt'] = df_vol.index
            df_vol['dt'] = df_vol['dt'].diff().apply(lambda x : x.total_seconds())
            for hl in vol_hls:
                df_vol[f'vol_{hl}']=calc_ts_ems(df_vol['var_raw'].values, df_vol['dt'].values, hl)

    def calc_bucketed_trade_markout(self):
        df_net_qty = 0 # TODO: calc_ems_net_qty(self.df_trades.copy())
        df_net_qty.index = df_net_qty.index + pd.Timedelta('1s')
        df_vol = calc_ems_vol(self.df_mid)
        df_vol.index = df_vol.index + pd.Timedelta('1s')

        df_trade_markout = calc_trade_markout(self.df_trades, self.df_mid, Ts=[180, 600, 1800])

        df_merged = pd.merge_asof(df_trade_markout, df_vol, left_index=True, right_index=True)
        df_merged = pd.merge_asof(df_merged, df_net_qty, left_index=True, right_index=True)

        for c in df_merged:
            if 'markout_' in c:
                df_merged[c] = df_merged.eval(f'{c} / qty')

        df_merged['net_qty_dis'] = pd.qcut(df_merged['net_qty_ems_30'].abs(), q=5)
        df_merged['vol_dis'] = pd.qcut(df_merged['vol_300'], q=5)

        col_markouts = [c for c in df_merged if 'markout_' in c and '_adj' not in c]

        df_vol = df_merged.groupby(['vol_dis'])[col_markouts].agg(['mean', 'median', 'std'])
        df_net_qty = df_merged.groupby(['net_qty_dis'])[col_markouts].agg(['mean', 'median', 'std'])
        return df_vol, df_net_qty

    def calc_order_stats(self):
        # null_val = b'NULL_VAL'
        # powe = b'POST_ONLY_WOULD_EXECUTE'
        # ioc = b'IOC'
        # gtx = b'GTX'
        # cancelled = b'CANCELLED'
        # filled = b'FILLED'
        # rejected = b'REJECTED'

        f_exec_stats = lambda df: df.query('doneReason != b"NULL_VAL"').groupby('handle').last().groupby(['timeInForce', 'doneReason'])[
            ['remainingQty', 'executedQty']].agg(['sum', 'count'])
        df_exec_stats = self.df_order.groupby(lambda time: time.date()).apply(f_exec_stats)
        df_exec_stats.columns = ['_'.join(c) for c in df_exec_stats.columns]
        df_exec_stats = df_exec_stats.groupby(['timeInForce', 'doneReason']).agg(['mean', 'median', 'std'])
        return df_exec_stats

    def calc_pnl_stats(self, exclusive_dates = None):
        if exclusive_dates:
            exclusive_dates = {pd.Timestamp(e).date() for e in exclusive_dates }
        else:
            exclusive_dates = {}

        df_stats = pd.concat(self.df_stats_list, axis = 0)
        df_daily = df_stats.groupby('date')[['PnL']].apply(lambda df: df[['PnL']].iloc[-1] - df[['PnL']].iloc[0])
        df_daily = df_daily.query('date not in @exclusive_dates').copy()
        df_daily['CumPnL'] = df_daily['PnL'].cumsum()
        #df_daily['PnL'] = df_daily['CumPnL'].diff().fillna(df_daily['CumPnL'].iloc[0])
        return df_daily

    def calc_trading_pnls(self, holdings_sec: list):
        df_res = []
        holdings_sec.append(np.inf) #total pnl
        for t in holdings_sec:
            df_pnl = calc_trading_pnl(self.df_trades, self.df_mid, T=t)
            df_res.append(df_pnl)
        df_res = pd.concat(df_res, axis=1)
        df_res = df_res.loc[:, ~df_res.columns.duplicated()]
        df_res['cum_net_qty'] = df_res['net_qty'].cumsum()
        df_res['cum_gross_qty']= df_res['net_qty'].abs().cumsum()

        cols = [c for c in df_res if 'trading_pnl_' in c]
        cols += 'mid cum_net_qty qty net_qty total_pnl fee'.split()
        df_res = df_res[cols].fillna(0.0)
        df_res = pd.merge_asof(df_res,
                               self.df_stats[['PnL']].rename({'PnL' : 'PnL_from_engine'}, axis=1),
                               left_index= True, right_index=True)
        return df_res

    def calc_daily_trades(self):
        df_daily = self.df_trades.groupby('date')[['notional']].agg(['sum', 'count'])
        df_daily.columns = ['notional', 'trd_cnt']
        return df_daily

    def calc_pos_corr(self):
        df_corr = self.df_stats.groupby('date')[['position']].apply(
            lambda df: calc_auto_corr(df, 'position', self.pos_corr_lags))
        df_corr.index.rename('lag', level=1, inplace=True)
        max_func = lambda x: x.abs().max()
        max_func.__name__ = 'max_abs'
        df_corr = df_corr.groupby('lag').agg(['mean', 'median', 'std', max_func])
        df_corr = df_corr.rename({'position': 'pos_corr'}, axis=1)
        df_corr.columns = ['_'.join(c) for c in df_corr]
        return df_corr

class GroupSimPTA:
    def __init__(self, instr, sim_name: str, tactic_pattern='*'):
        import glob
        ev_path = '' #TODO
        self.sim_name = sim_name
        n_samples = len(glob.glob(f'{ev_path}/{sim_name}/*/sample_*'))
        df_res, daily_pnls = {}, []

        instr2ag6 = {
            'BDM.ETH.USDT.FP': 'BdmEthUsdtFp0',
            'BDM.BTC.USDT.FP': 'BdmBtcUsdtFp0',
            'BIN.BTC.TUSD': 'BinBtcTusd',
        }
        ag6_symbol = instr2ag6[instr]

        self.ptas = []
        print(f'n_sims = {n_samples}')

        for i in range(1, n_samples + 1):
            sample = f'sample_{i}'
            pta = SimPostTradeAnalysis(ag6_symbol, sim_name, ev_path, sample=sample, tactic_pattern=tactic_pattern)
            self.ptas.append(pta)

    def summarize(self, exclusive_dates={}):
        df_res, daily_pnls = [], []
        for pta in self.ptas:
            df_pnl = pta.calc_pnl_stats(exclusive_dates=exclusive_dates)
            df_volume = pta.calc_daily_trades()
            df_daily = pd.concat([df_pnl, df_volume], axis=1)
            # exclusive_dates=['20230629', '20230630'])# + list(pd.date_range('20230729', '20230814')))
            # daily_pnls.append(df_daily)
            pnl = df_daily['PnL']

            order_stats = pta.calc_order_stats()
            exe_qty = order_stats['executedQty_sum']['mean'].sum(
                axis=0)  # RH: double check this, see if IOC numbers are correct by done Reason
            gtx_exe_qty = order_stats.query('timeInForce == b"GTX"')['executedQty_sum']['mean'].sum(axis=0)
            unfilledqty = order_stats['remainingQty_sum']['mean'].sum(axis=0)
            qty_sent = unfilledqty + exe_qty

            df_pos_corr = pta.calc_pos_corr()
            res = {
                # pnl's
                # 'n_days' : np.nan, #todo
                'sharpe': sharpe(pnl),
                'pnl_mean': pnl.mean(),
                'pnl_median': pnl.median(),
                'pnl_std': pnl.std(),
                'pnl_min': pnl.min(),
                'win_ratio': win_ratio(pnl),
                # activities
                'avg_notional': df_daily['notional'].mean(),
                'avg_trade_cnt': df_daily['trd_cnt'].mean(),
                'exe_qty': exe_qty,
                'qty_sent': qty_sent,
                'qty_fill_ratio': exe_qty / qty_sent,
                'post_qty_exe_ratio': gtx_exe_qty / exe_qty,
                # 'orders_sent' : np.nan,  #todo
                # 'gtx_order_cancel_ratio' : np.nan, #todo
                # 'ioc_fill_ratio' : np.nan
            }

            # pos corr
            for lag in pta.pos_corr_lags:
                res[f'pos_corr_{lag}'] = df_pos_corr.loc[lag, 'pos_corr_median']
            df_res.append(res)

        df_res = pd.DataFrame(df_res)
        comma_sep_cols = [c for c in df_res if 'pnl' in c or 'notional' in c]
        return format_df_nums(df_res, comma_sep_cols)  # , daily_pnls

