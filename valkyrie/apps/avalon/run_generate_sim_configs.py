import numpy as np
import argparse
from valkyrie.tools import ConfigGenerator as CG
from valkyrie.tools import ROOT_DIR



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--tpl', help='template file')
  parser.add_argument('--group', help='sim group', default="")

  args = vars(parser.parse_args())
  tpl, group = args['tpl'], args['group']

  pattern2params = {
                   '@quoter_avg_spread' : np.arange(0.01, 0.15, 0.02),
                   '@quoter_max_offset' : [0.03, 0.05]
  }

  cg = CG(f'{ROOT_DIR}/configs/{tpl}', pattern2params)

  cg.generate(f'{ROOT_DIR}/configs/{group}')