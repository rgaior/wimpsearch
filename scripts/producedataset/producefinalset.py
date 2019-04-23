#####################################################################
## starts from the PFS data and apply the multiple cuts defined    ##
## if previous analysis to produce a final data frame              ##
#####################################################################
import pandas as pd
import glob
import sys
import os
cwd = os.getcwd()
classpath = cwd + '/../../classes/'
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
parser.add_argument("outfolder", type=str, nargs='?',default='', help="output folder for the final dataset")
parser.add_argument("cutversion", type=int, nargs='?', default=1,help="1: image cut + same dLL for all. --- 2: dLL cut depends on the dark current")
parser.add_argument("outname", type=str, nargs='?',default='', help="output name")

args = parser.parse_args()
file = args.file
cutversion = args.cutversion
outfolder = args.outfolder
outname = args.outname


DCfile = constant.datafolder + '/DC/DCfile.pkl'
dfDC = pd.read_pickle(DCfile)
df = pd.read_pickle(file)

df = df.query(constant.basecuts) # position cut, mask,  ll cut,  qmax
df = df.query(constant.radoncut) # radon cut removes some runid
datadcdf = dfDC
size = datadcdf.shape[0]
datadcdf = datadcdf.merge(df,left_on=['image','ext'], right_on=['RUNID','EXTID'], how='outer')
datadcdf = datadcdf.dropna()
df =datadcdf
countofim = 0
df = df.query("RUNID < 3704")

if cutversion  == 1:
    df = df.query("DC < " + str(constant.dclimvalue))
    for id in np.unique(df.RUNID):
        dftemp = df.query("RUNID == " +str(id))
        countofim += len(np.unique(dftemp.EXTID))
    print 'countofim = ' ,countofim
    df = df.query("ll -llc  < " + str(constant.dllcutvalue))
if cutversion  == 2:
    #fit of the dc dll:
#    print z[0], z[1]    
    z = np.polyfit(constant.a_dc,constant.a_dllcut,1)
    dllcutfit = np.poly1d(z)
    newdf = pd.DataFrame(columns=df.columns)
    for ind,row in df.iterrows():
        dc = row['DC']
        dllcut = utils.getdllm2(dc)
        dll = row['ll'] - row['llc']
        if (dll < dllcut):
            newdf = newdf.append(row)

    df = newdf
df = utils.dfbasic(df) 
df.to_pickle(outfolder +  outname+'_' + str(cutversion) + '.pkl')
