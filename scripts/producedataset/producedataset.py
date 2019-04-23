#####################################################################
## starts from the pkl data and apply the multiple cuts defined    ##
## merges also with the DC file                                    ##
#####################################################################
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
############################
##  argument parser       ##
############################
parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, nargs='?',default='', help="pickle file with clusters dataframe")
parser.add_argument("outfolder", type=str, nargs='?',default='', help="output folder")
parser.add_argument("outname", type=str, nargs='?',default='', help="output name")
parser.add_argument("--cut", type=str, nargs='?', default="RUNID!=0",help="additional cut to be separated as in a dataframe query selection")
args = parser.parse_args()
file = args.file
outfolder = args.outfolder
cut = args.cut
outname = args.outname


DCfile = constant.datafolder + '/DC/DCfile.pkl'
dfDC = pd.read_pickle(DCfile)
df = pd.read_pickle(file)
df = df.query(constant.basecuts)
df = df.query(constant.radoncut) # radon cut removes some runid
datadcdf = dfDC
size = datadcdf.shape[0]
datadcdf = datadcdf.merge(df,left_on=['image','ext'], right_on=['RUNID','EXTID'], how='outer')
datadcdf = datadcdf.dropna()

df = datadcdf.query(cut)

df.to_pickle(outfolder +  outname + '.pkl')
