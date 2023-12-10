import json
from copy import deepcopy
from valkyrie.securities import stocks_good_dvd, parent, ROOT_PATH

class ConfigGenerator:
  def __init__(self, config, params):
    self.config = deepcopy(config)
    self.params = deepcopy(params)


  def _expand_universe(self, stocks):
    #stocks = [stk for stk in stocks if stk not in self.config['universe']]
    universe = deepcopy(self.config['universe'])
    universe += stocks
    return universe

  def _expand_risk(self, stocks):
    risk_config = deepcopy(self.config['risk'])
    params = self.params

    factors, loadings = risk_config['factors'], risk_config['loadings']
    for stk in stocks:
      p = parent(stk)
      factors[f'RES_{p}']    = { 'tol' : params['res_tol'], 'cost' : params['cost'] }
      factors[f'IDIO_{stk}'] = { 'tol' : params['idio_tol'], 'cost' : params['cost'] }

      loadings[stk] = {"MKT_PFF": 1.0, f"RES_{p}": 1.0, f"IDIO_{stk}": 1.0}
    return risk_config

  def generate(self, stocks):
    universe = self._expand_universe(stocks)
    risk_config = self._expand_risk(stocks)

    config = deepcopy(self.config)
    config['universe'] = universe
    config['risk'] = risk_config
    return config

pff_config = {
  "env": {
    "sim_timer_interval": "30s",
    "sim_root_dir": "/home/bb/projects/valkyrie/sim_dir"
  },

  "log":
  {
    "file"   : "./trading_config_test.out",
    "level"  : "critical",
    "summary": "./trade_summary.h5"
  },

  "universe": ["PFF"],

  "risk": {
    "factors" :
    {
      "MKT_PFF"   : {"tol":  8e5, "cost": 0.02},
      "IDIO_PFF":   {"tol":  8e5, "cost": 0.02},
    },

    "loadings":
    {
      "PFF":   {"MKT_PFF": 1.0, "IDIO_PFF": 1.0}
    }
  },

  "stalker" : {
    "default": {
      "avg_sprd": 0.03,
      "fee": 0.005,
      "fill_prob_base": 0.1,
      "fill_prob_coef": 0.2,
      "max_offset": 0.03,
      "max_order_sz": 600,
      "min_delta_sz": 50,
      "min_edge": 0.01,
      "min_order_sz": 200,
      "order_interval_in_sec": 120,
      "exchs": [
        "ISLAND",
        "NYSE",
        "ARCA"
      ]
    },
    "PFF": {
      "avg_spread": 0.01,
      "max_offset": 0.01,
      "stalk": False
    },
    "pricer": {
      "default": {
        "dvd_dir": "/home/bb/projects/valkyrie/data/universe/dvd/research_20220205",
        "record_feature": True
      }
    },
    "sim_matcher": {
      "default": {}
    }
  }
}

if __name__ == "__main__":
  params = {'res_tol':2.5e4, 'idio_tol':2.5e4, 'cost':0.04}
  config = ConfigGenerator(pff_config, params).generate(stocks_good_dvd())

  with open(f'{ROOT_PATH}/universe/trading_config_test2.json', 'w') as f:
    json.dump(config, f, indent= 2, sort_keys= True)











