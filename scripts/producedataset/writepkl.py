import numpy as np
import matplotlib.pyplot as plt
import sys
import os
cwd = os.getcwd()
utilspath = cwd + '/../../utils/'
sys.path.append(utilspath)
import utils
import constant
import glob 
import argparse
import pandas as pd
############################
##  argument parser       ##
############################
parser = argparse.ArgumentParser()
parser.add_argument("infolder", type=str, nargs='?',default='', help="root file folder")
parser.add_argument("outfolder", type=str, nargs='?',default='', help="pkl file folder")
parser.add_argument("outname", type=str, nargs='?',default='', help="pkl file name")
parser.add_argument("--onedfperfile",action='store_true',help="if specified will write a dataframe for each root file")
parser.add_argument("--infile",type=str,nargs='?',default='',help='if specified takes only one file in the infolder')
parser.add_argument("--N",type=int,nargs='?',default=None,help='if specified takes N files in the folder')
parser.add_argument("--sel",action='store_true',help='if specified will write also a dataframe with selection criteria defined by basecuts in constant.py')
parser.add_argument("--sim",action='store_true',help='if specified will write also a dataframe for the simclusters tree')

args = parser.parse_args()
infolder = args.infolder
outfolder = args.outfolder
outname = args.outname
infile = args.infile
Nfile = args.N
sel = args.sel
onedfperfile = args.onedfperfile
sim = args.sim
# define the folder with files depending on the options
files = glob.glob(infolder + '/*.root*')
print args
if Nfile:
    files = files[:Nfile]
if infile:
    files = [infolder + infile]
print files
# dataframe with the information of all the root tree
dfall = pd.DataFrame()
dfallsim = pd.DataFrame()

print 'number of files to be converted: ', len(files)
# loop over the defined files array

for f in files:
    fname = f[f.rfind('/')+1:f.rfind('.')]
    dft = utils.readtree(f,'clusters')    
    print f
    if sim:
        dftsim = utils.readtree(f,'simclusters')    
        dftsim = utils.dfbasic(dftsim)
    if onedfperfile:
        dftsim.to_pickle(outfolder + fname +'_sim.pkl')
        dft.to_pickle(outfolder + fname+'.pkl')
        if sel:
            if dft.shape[0] > 0:
                dftsel = dft.query(constant.basecuts)
            else:
                dftsel = dft
            dftsel = utils.dfbasic(dftsel)
            dftsel.to_pickle(outfolder + fname+'_sel.pkl')
    else:
        dfall = dfall.append(dft,ignore_index=True)
        if sim:
            dfallsim = dfallsim.append(dftsim,ignore_index=True)
import time

if not onedfperfile:
    if sel:
        dfallsel = dfall.query(constant.basecuts)
        dfallsel = utils.dfbasic(dfallsel) 
        dfallsel.to_pickle(outfolder + outname+'_sel.pkl')
    start = time.time()
    dfall = utils.dfbasic(dfall)
    end = time.time()
    print 'elapsed time = ', end - start
    dfall.to_pickle(outfolder + outname+'.pkl')
    if sim:
        dfallsim = utils.dfbasic(dfallsim)
        dfallsim.to_pickle(outfolder + outname+'_sim.pkl')
