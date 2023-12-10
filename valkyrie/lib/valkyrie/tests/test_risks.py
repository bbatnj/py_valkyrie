from unittest import TestCase
from valkyrie.components.risk import *

class TestSecurityRiskGroup(TestCase):
    def test_on_fill(self):
        md = MessageDispatcher()
        message_logger = MessageLogger(verbose=False)
        md.add_risk_listeners([message_logger])

        trading_config_fn = f'{ROOT_PATH}/universe/trading_config_test.json'
        sec_fn = f'{ROOT_PATH}/universe/test_sec_file.csv'

        config = {
            'sec_fn': sec_fn,
            'trading_config_fn': trading_config_fn
        }

        srg = SecurityRiskGroup('srg', config, md)

        tv_update = TVUpdate()
        tv_update.stk = "PFF"
        tv_update.risk_tv = 30.00
        tv_update.tv = 30.00
        srg.onTV(tv_update)

        tv_update.risk_tv = 25.00
        tv_update.tv = 25.00
        for stk in ['AGNCM', 'AGNCN', 'AGNCO']:
            tv_update.stk = stk
            srg.onTV(tv_update)

        order_update = OrderUpdate()
        order_update.response = OrderResponse.Fill
        order_update.stk = 'PFF'
        order_update.sz = 5e5 / 30
        order_update.side = Side.Buy
        order_update.response = OrderResponse.Fill
        srg.onOrderUpdate(order_update)

        order_update.stk = 'AGNCM'
        order_update.sz = 5e5 / 25.0
        order_update.side = Side.Sell
        srg.onOrderUpdate(order_update)

        self.assertAlmostEqual(message_logger.stk2risk['PFF'].adj,   -0.005, 4)
        self.assertAlmostEqual(message_logger.stk2risk['PFF'].u,     5e-8,  9)
        self.assertAlmostEqual(message_logger.stk2risk['AGNCM'].adj, 0.3, 4)
        self.assertAlmostEqual(message_logger.stk2risk['AGNCM'].u,   6.4e-7, 8)

        self.assertAlmostEqual(message_logger.stk2risk['AGNCN'].adj, 0.1, 4)
        self.assertAlmostEqual(message_logger.stk2risk['AGNCN'].u,   6.4e-7, 8)
        self.assertAlmostEqual(message_logger.stk2risk['AGNCO'].adj, 0.1, 4)
        self.assertAlmostEqual(message_logger.stk2risk['AGNCO'].u,   6.4e-7, 8)

        print('Done risk test')

