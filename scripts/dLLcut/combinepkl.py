import pandas as pd
import glob
import sys
import os
cwd = os.getcwd()
utilspath = cwd + '/../../utils/'
sys.path.append(utilspath)
import utils
import constant
import argparse
import numpy as np

files = ['140_100_1_sel.pkl','140_100_2_sel.pkl','140_100_3_sel.pkl','140_30_1_sel.pkl','140_30_2_sel.pkl','140_30_3_sel.pkl','140_30_4_sel.pkl','135_30_5_sel.pkl','135_30_6_sel.pkl','135_30_8_sel.pkl','135_30_10_sel.pkl','135_30_11_sel.pkl']
datafolder = '/Users/gaior/DAMIC/data/simDC2/pkl/'
dffirst = pd.read_pickle(datafolder + files[0])
dfall = pd.DataFrame(columns=dffirst.columns)
for f in files:
    print f
    dftemp = pd.read_pickle(datafolder + f)
    dfall = dfall.append(dftemp)

DCfile = constant.datafolder + '/DC/DCfile.pkl'
dfDC = pd.read_pickle(DCfile)
datadcdf = dfDC
size = datadcdf.shape[0]
datadcdf = datadcdf.merge(dfall,left_on=['image','ext'], right_on=['RUNID','EXTID'], how='outer')
#datadcdf = datadcdf.dropna()
dfall = datadcdf

dfall.to_pickle(datafolder + 'all.pkl')
