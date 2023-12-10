import gc
import datetime
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression as LR
from sklearn.cross_decomposition import PLSRegression as PLSR
from statsmodels.stats.weightstats import DescrStatsW
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
#%matplotlib notebook
pd.set_option('display.max_rows', 500)

#############################
#valkyrie
#############################
from valkyrie.securities import stocks_good_dvd, parent
from valkyrie.quants.linear_model import lm_fit, wcorr, analyze_features
#from valkyrie.quants.feature_analyzer import FeatureMgr
from valkyrie.tools import *
from valkyrie.components.trade_summary import GroupSimViewer

#############################
#valkyrie Machine learning
#############################
import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import LeakyReLU as LReLU
import valkyrie.ml.data as ml_data
import valkyrie.ml.modules as ml_modules
import valkyrie.ml.utils as ml_utils
from valkyrie.ml.data import mrf_linear_fit
Filter1D = ml_modules.Filter1D
BN1D = ml_modules.BN1D





