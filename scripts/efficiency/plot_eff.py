#####################################################################
## plots the efficiency of one of the extension or all of them     ##
## Allows to enter a cut expresion                                 ##
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
import matplotlib.pyplot as plt
############################
##  argument parser       ##
############################
parser = argparse.ArgumentParser()
parser.add_argument("--ext", type=int, nargs='*',default=[0], help="extension number")
parser.add_argument("--cut", type=str, nargs='?', default="RUNID!=0",help="additional cut to be separated as in a dataframe query selection")
args = parser.parse_args()
exts = args.ext
cut = args.cut

# define here the default cut:
if cut == "RUNID!=0":
    reccut = constant.recpositioncut + ' & ' + constant.basecuts + ' & multirows == 0'
    simcut = constant.simpositioncut 
else:
    reccut = constant.recpositioncut + ' & ' + constant.basecuts + ' & multirows == 0' + ' & ' + cut
    simcut = constant.simpositioncut 

print "chosen extension: ", exts
print "cut chosen: ", cut

# get the data
#the data were simuated in two steps, a first batch of simulation 
# was done at low energy with an exponential energy distribution
# a second batch was done with a flat distribution in energy.
# for each batch we have the reconstructed and simulated trees.
dflowrec = pd.read_pickle(constant.datafolder + '/efficiency/lowErec.pkl')
dflowsim = pd.read_pickle(constant.datafolder + '/efficiency/lowEsim.pkl')

dfhighrec = pd.read_pickle(constant.datafolder + '/efficiency/lowErec.pkl')
dfhighsim = pd.read_pickle(constant.datafolder + '/efficiency/lowEsim.pkl')

fig = plt.figure(figsize=(8,6))
ax = plt.subplot(111)
col = {1:'black',2:'red',3:'blue',4:'green',6:'gray',11:'darkseagreen',12:'lightskyblue'}

for ext in exts:
    print 'ext = ', ext
    if ext == 0:    # ext == 0 means all extension together
        dflowrec_sel = dflowrec.query(reccut)
        dflowsim_sel = dflowsim.query(simcut)
        dfhighrec_sel = dfhighrec.query(reccut)
        dfhighsim_sel = dfhighsim.query(simcut)
    else:
        dflowrec_sel = dflowrec.query(reccut + ' & EXTID == ' +str(ext))
        dflowsim_sel = dflowsim.query(simcut + ' & EXTID == ' +str(ext))
        dfhighrec_sel = dfhighrec.query(reccut + ' & EXTID == ' +str(ext))
        dfhighsim_sel = dfhighsim.query(simcut + ' & EXTID == ' +str(ext))
    bins = np.linspace(0.03,0.8,45)
#    [cbins, eff_p, err_eff_p] = utils.geteffsim(dflowrec_sel,dflowsim_sel,bins,"RUNID!=0","RUNID!=0")
    [cbins, eff_p, err_eff_p] = utils.geteff(dflowrec_sel,dflowsim_sel,bins,"RUNID!=0","RUNID!=0")
#    ax.errorbar(cbins,eff_p,yerr=err_eff_p,label='ext: '+str(ext),fmt='.-')
    if ext == 0:
        ax.errorbar(cbins,eff_p,yerr=err_eff_p,label='all ext',lw=2,fmt='o-',color='darkorange',zorder=20)
    else:
        ax.errorbar(cbins,eff_p,yerr=err_eff_p,label='ext: '+str(ext),lw=2,fmt='.-',color=col[ext])

plt.xlabel('reconstructed energy [keV]')
#plt.xlabel('simulated energy [keV]')
plt.ylabel('efficiency')
plt.legend()
plt.show()        





