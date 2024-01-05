import os
import sys
import argparse
import datetime
import collections
import logging
import time
from multiprocessing import Process
import copy
from abc import ABC, abstractmethod
from collections import defaultdict as DefaultDict

import pandas as pd
import numpy as np
import pandas_market_calendars as mcal

from Wrapper import TestWrapper, TestClient, SetupLogger, printWhenExecuting, printinstance
from ibapi.common import * # @UnusedWildImport
from ibapi.order_condition import * # @UnusedWildImport
from ibapi.contract import * # @UnusedWildImport
from ibapi.order import * # @UnusedWildImport
from ibapi.ticktype import * # @UnusedWildImport
from ibapi.tag_value import TagValue
from ContractSamples import ContractSamples
from ibapi.utils import iswrapper

from valkyrie.tools import *
from valkyrie.data import check_hist_ib_data_integrity
from valkyrie.securities import *

# this is here for documentation generation
"""
#! [ereader]
        # You don't need to run this in your code!
        self.reader = reader.EReader(self.conn, self.msg_queue)
        self.reader.start()   # start thread
#! [ereader]
"""



class DataMgr(ABC):
    def __init__(self, sec_list, date, start_time, outpath):
        date = str(date).split()[0]
        self.start_ts = pd.Timestamp(date + " " + start_time, tz='US/EASTERN').value/1e9
        self.current_ts = self.start_ts
        self.sec_list = copy.deepcopy(sec_list)
        self.lastdata = None
        self.date = date
        self.outpath = outpath
        self.tick2df = DefaultDict(lambda: [])

    def _onCurrentFinished(self):
        self.current_ts = self.start_ts
        if self.sec_list:
            ticker = self.sec_list.pop(0)
            self._df2h5(ticker)

    def _onMoreData(self, last_tick):
        print(f'onMoreData {pd.Timestamp(last_tick.time * 1e9)}')
        self.current_ts = last_tick.time + 1
        self.lastdata = copy.deepcopy(last_tick)

    @abstractmethod
    def _onTicks(self, ticker, ticks):
        pass


    def _df2h5(self, ticker):
        os.makedirs(self.outpath, exist_ok=True)
        try:
            df = pd.concat(self.tick2df[ticker])
            print(f'writing ' + self.outpath + f'\\{ticker}.h5')
            df.to_hdf(self.outpath + f'\\{ticker}.h5', key='data')
        except Exception as e:
            print(f'error generate h5 for {ticker} at {self.outpath} due to ' + e)

    def onBlockData(self, ticker, ticks):
        self._onTicks(ticker, ticks)

        if 0 <= len(ticks) < 1000:
            self._onCurrentFinished()
        else:
            last_tick = ticks[-1]
            if last_tick.time < self.current_ts:
                self._onCurrentFinished()
            else:
                self._onMoreData(last_tick)

    def currentSec(self):
        if self.sec_list:
            return self.sec_list[0]
        else:
            return None



class BidAskDataMgr(DataMgr):
    def __init__(self, sec_list, date, start_time, outpath):
        super().__init__(sec_list, date, start_time, outpath)

    def _onTicks(self, ticker, ticks):
        df = pd.DataFrame(ticks, columns=['obj'])
        df['time'] = df['obj'].apply(lambda x: pd.Timestamp(x.time * 1e9, tz='US/EASTERN'))
        df['priceBid'] = df['obj'].apply(lambda x: x.priceBid)
        df['priceAsk'] = df['obj'].apply(lambda x: x.priceAsk)
        df['sizeBid'] = df['obj'].apply(lambda x: x.sizeBid)
        df['sizeAsk'] = df['obj'].apply(lambda x: x.sizeAsk)
        df = df.drop('obj', axis=1).set_index('time').sort_index()
        df['ticker'] = ticker
        self.tick2df[ticker].append(df)

class TradeDataMgr(DataMgr):
    def __init__(self, sec_list, date, start_time, outpath):
        super().__init__(sec_list, date, start_time, outpath)

    def _onTicks(self, ticker, ticks):
        df = pd.DataFrame(ticks, columns=['obj'])
        df['time'] = df['obj'].apply(lambda x: pd.Timestamp(x.time * 1e9, tz='US/EASTERN'))
        df['price'] = df['obj'].apply(lambda x: x.price)
        df['size'] = df['obj'].apply(lambda x: x.size)
        df['exchange'] = df['obj'].apply(lambda x: x.exchange)
        df['specialConditions'] = df['obj'].apply(lambda x: x.specialConditions)
        df = df.drop('obj', axis=1).set_index('time').sort_index()
        df['ticker'] = ticker
        self.tick2df[ticker].append(df)

# ! [socket_init]
class HistDataDownloader(TestWrapper, TestClient):
    def __init__(self, date, stocks, outpath, nbbo_px_only = False):
        TestWrapper.__init__(self)
        TestClient.__init__(self, wrapper=self)
        # ! [socket_init]
        self.nKeybInt = 0
        self.started = False
        self.nextValidOrderId = None
        self.permId2ord = {}
        self.reqId2nErr = collections.defaultdict(int)
        self.globalCancelOnly = False

        self.reqid2ticker = {}
        self.reqid = 18000

        self.date = pd.Timestamp(date)
        self.outpath = outpath
        self.stock_list = stocks
        self.nbbo_px_only = nbbo_px_only

        stocks = set()
        for stk in self.stock_list:
            bid_ask_exist = os.path.isfile(f'{self.outpath}/bid_ask/{ymd_dir(self.date)}/{stk}.h5')
            trades_exist  = os.path.isfile(f'{self.outpath}/trade/{ymd_dir(self.date)}/{stk}.h5')
            if not (bid_ask_exist and trades_exist):
                stocks.add(stk)

        stock_list = list(stocks)
        self.bidask_mgr = BidAskDataMgr(stock_list, date, "03:30:00",
                                        f'{self.outpath}/bid_ask/{ymd_dir(self.date)}')
        if self.nbbo_px_only:
            stock_list = []
        self.trade_mgr  = TradeDataMgr(stock_list, date, "03:30:00",
                                       f'{self.outpath}/trade/{ymd_dir(self.date)}')

    def dumpReqAnsErrSituation(self):
        logging.debug("%s\t%s\t%s\t%s" % ("ReqId", "#Req", "#Ans", "#Err"))
        for reqId in sorted(self.reqId2nReq.keys()):
            nReq = self.reqId2nReq.get(reqId, 0)
            nAns = self.reqId2nAns.get(reqId, 0)
            nErr = self.reqId2nErr.get(reqId, 0)
            logging.debug("%d\t%d\t%s\t%d" % (reqId, nReq, nAns, nErr))

    @iswrapper
    # ! [connectack]
    def connectAck(self):
        if self.asynchronous:
            self.startApi()

    # ! [connectack]

    @iswrapper
    # ! [nextvalidid]
    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)

        logging.debug("setting nextValidOrderId: %d", orderId)
        self.nextValidOrderId = orderId
        print("NextValidId:", orderId)
    # ! [nextvalidid]

        # we can start now
        self.start()

    def start(self):
        if self.started:
            return

        self.started = True

        if self.globalCancelOnly:
            print("Executing GlobalCancel only")
            self.reqGlobalCancel()
        else:
            print("Request Account requests")
            self.historicalTicksOperations()
            print("Executing requests ... finished")

    def keyboardInterrupt(self):
        self.nKeybInt += 1
        if self.nKeybInt == 1:
            self.stop()
        else:
            print("Finishing test")
            self.done = True

    def stop(self):
        print("Executing cancels")
        print("Executing cancels ... finished")

    @iswrapper
    # ! [error]
    def error(self, reqId: TickerId, errorCode: int, errorString: str):
        super().error(reqId, errorCode, errorString)
        print("Error. Id:", reqId, "Code:", errorCode, "Msg:", errorString)

    # ! [error] self.reqId2nErr[reqId] += 1


    @iswrapper
    def winError(self, text: str, lastError: int):
        super().winError(text, lastError)


    @iswrapper
    # ! [tickprice]
    def tickPrice(self, reqId: TickerId, tickType: TickType, price: float,
                  attrib: TickAttrib):
        super().tickPrice(reqId, tickType, price, attrib)
        print("TickPrice. TickerId:", reqId, "tickType:", tickType,
              "Price:", price, "CanAutoExecute:", attrib.canAutoExecute,
              "PastLimit:", attrib.pastLimit, end=' ')
        if tickType == TickTypeEnum.BID or tickType == TickTypeEnum.ASK:
            print("PreOpen:", attrib.preOpen)
        else:
            print()
    # ! [tickprice]


    def reqHistTicks(self, data_mgr, fields):
        if data_mgr.currentSec():
            sec = data_mgr.currentSec()
            contract = ContractSamples.USStockAtSmart(sec)
            now = str(pd.Timestamp(data_mgr.current_ts * 1e9, tz='US/EASTERN'))
            now = now.replace('-','')[0:17]
            print(f'requesting ({self.reqid}) {fields} for {data_mgr.currentSec()} {now}')
            self.reqid2ticker[self.reqid] = sec
            time.sleep(1)
            self.reqHistoricalTicks(self.reqid, contract, now, "", 1000, fields,
                                useRth = 0, ignoreSize= self.nbbo_px_only, miscOptions=[])
            self.reqid += 1
        else:
            print(f'all done for {fields}')



    @printWhenExecuting
    def historicalTicksOperations(self):
        # ! [reqhistoricalticks]
        # self.reqHistoricalTicks(18003, ContractSamples.USStockAtSmart('AGNC'),
        #                        "20220113 03:39:33", "", 1000, "BID_ASK", 1, True, [])
        self._exit_if_done()
        self.reqHistTicks(self.bidask_mgr, 'BID_ASK')
        self.reqHistTicks(self.trade_mgr, 'TRADES')
        # ! [reqhistoricalticks]



    @iswrapper
    # ! [historicaldata]
    def historicalData(self, reqId:int, bar: BarData):
        print("HistoricalData. ReqId:", reqId, "BarData.", bar)
    # ! [historicaldata]

    @iswrapper
    # ! [historicaldataend]
    def historicalDataEnd(self, reqId: int, start: str, end: str):
        super().historicalDataEnd(reqId, start, end)
        print("HistoricalDataEnd. ReqId:", reqId, "from", start, "to", end)
    # ! [historicaldataend]

    @iswrapper
    # ! [historicalDataUpdate]
    def historicalDataUpdate(self, reqId: int, bar: BarData):
        print("HistoricalDataUpdate. ReqId:", reqId, "BarData.", bar)
    # ! [historicalDataUpdate]

    def _exit_if_done(self):
        if not (self.bidask_mgr.currentSec() or self.trade_mgr.currentSec()):
            print('all market data done')
            self.done = True
            #self.close()
            sys.exit()

    @iswrapper
    # ! [historicalticksbidask]
    def historicalTicksBidAsk(self, reqId: int, ticks: ListOfHistoricalTickBidAsk, done: bool):
        #for tick in ticks:
        #    print("HistoricalTickBidAsk. ReqId:", reqId, tick, done)
        ticker = self.reqid2ticker[reqId]
        self.bidask_mgr.onBlockData(ticker, ticks)
        self.reqHistTicks(self.bidask_mgr, 'BID_ASK')
        self._exit_if_done()

    # ! [historicalticksbidask]

    @iswrapper
    # ! [historicaltickslast]
    def historicalTicksLast(self, reqId: int, ticks: ListOfHistoricalTickLast, done: bool):
        ticker = self.reqid2ticker[reqId]
        self.trade_mgr.onBlockData(ticker, ticks)
        self.reqHistTicks(self.trade_mgr, 'TRADES')
        self._exit_if_done()
    # ! [historicaltickslast]

    @iswrapper
    # ! [tickReqParams]
    def tickReqParams(self, tickerId:int, minTick:float,
                      bboExchange:str, snapshotPermissions:int):
        super().tickReqParams(tickerId, minTick, bboExchange, snapshotPermissions)
        print("TickReqParams. TickerId:", tickerId, "MinTick:", minTick,
              "BboExchange:", bboExchange, "SnapshotPermissions:", snapshotPermissions)

    def dumpTestCoverageSituation(self):
        for clntMeth in sorted(self.clntMeth2callCount.keys()):
            logging.debug("ClntMeth: %-30s %6d" % (clntMeth,
                                                   self.clntMeth2callCount[clntMeth]))

        for wrapMeth in sorted(self.wrapMeth2callCount.keys()):
            logging.debug("WrapMeth: %-30s %6d" % (wrapMeth,
                                                   self.wrapMeth2callCount[wrapMeth]))
    # ! [tickReqParams]


def run_for_date(date, stocks, outpath):
    SetupLogger()
    logging.debug("now is %s", datetime.datetime.now())
    logging.getLogger().setLevel(logging.ERROR)

    cmdLineParser = argparse.ArgumentParser("api tests")
    # cmdLineParser.add_option("-c", action="store_True", dest="use_cache", default = False, help = "use the cache")
    # cmdLineParser.add_option("-f", action="store", type="string", dest="file", default="", help="the input file")
    cmdLineParser.add_argument("-p", "--port", action="store", type=int,
                               dest="port", default=7496, help="The TCP port to use")
    cmdLineParser.add_argument("-C", "--global-cancel", action="store_true",
                               dest="global_cancel", default=False,
                               help="whether to trigger a globalCancel req")
    args = cmdLineParser.parse_args()
    print("Using args", args)
    logging.debug("Using args %s", args)
    # print(args)


    # enable logging when member vars are assigned
    from ibapi import utils
    Order.__setattr__ = utils.setattr_log
    Contract.__setattr__ = utils.setattr_log
    DeltaNeutralContract.__setattr__ = utils.setattr_log
    TagValue.__setattr__ = utils.setattr_log
    TimeCondition.__setattr__ = utils.setattr_log
    ExecutionCondition.__setattr__ = utils.setattr_log
    MarginCondition.__setattr__ = utils.setattr_log
    PriceCondition.__setattr__ = utils.setattr_log
    PercentChangeCondition.__setattr__ = utils.setattr_log
    VolumeCondition.__setattr__ = utils.setattr_log

    try:
        stock_list = stocks
        nbbo_px_only = False
        app = HistDataDownloader(date, stock_list,
                                 outpath=outpath,
                                 nbbo_px_only=nbbo_px_only)

        if args.global_cancel:
            app.globalCancelOnly = True
        # ! [connect]
        app.connect("127.0.0.1", args.port, clientId=0)
        # ! [connect]
        print("serverVersion:%s connectionTime:%s" % (app.serverVersion(),
                                                      app.twsConnectionTime()))

        # ! [clientrun]
        app.run()
        # ! [clientrun]
    except Exception as e:
        print(f'error due to {e}')
    finally:
        app.dumpTestCoverageSituation()
        app.dumpReqAnsErrSituation()

#import processsss

if __name__ == "__main__":
    nyse = mcal.get_calendar('NYSE')
    outpath = f"{ROOT_PATH}/raw/hist_ib/"

    for date in nyse.schedule(start_date='2020-01-01', end_date='20211231').index:
        stocks = stocks_good_dvd()
        stocks = ['PFF']
        re1 = check_hist_ib_data_integrity(date, stocks, outpath, 'bid_ask')
        #re2 = check_hist_ib_data_integrity(date, stocks, outpath, 'trade')
        re2 = re1
        stocks = set(re1.keys()).intersection(set(re2))

        if stocks:
            print(f'working on {date} for {stocks}')
            run_for_date(date, stocks, outpath)
            #p = Process(target=run_for_date, args=(date, stocks, outpath))
            #p.start()
            #p.join()
            print('sleeping for 120')
            time.sleep(120)
