from data import ROOT_PATH
from valkyrie.components.simulator import Simulator

if __name__ == "__main__":
  data_path = '{ROOT_PATH}/merged/hist_ib_raw_md'

  config = {'sec_fn': f'{ROOT_PATH}/universe/test_sec_file.csv'}

  sim = Simulator(data_path, config)
  sdate, edate = '20200601', '20200630'
  sim.run(sdate, edate)
