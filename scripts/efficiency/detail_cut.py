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
args = parser.parse_args()
exts = args.ext


cutrec1 = '(centery < 42 & centery > 1) & (centerx < 8250 & centerx > 4400) & (simdistx == 0 | (simdistx > 5) & (simdisty > 2) ) '
cutsim1 = '(simy < 42 & simy > 1) &  (simx < 8250 & simx > 4400) & ((simdistx > 5) & (simdisty > 2) )'

print "chosen extension: ", exts

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
reccut = cutrec1
simcut = cutsim1
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
    [cbins, eff_p, err_eff_p] = utils.geteffsim(dflowrec_sel,dflowsim_sel,bins,"RUNID!=0","RUNID!=0")
    ax.errorbar(cbins,eff_p,yerr=err_eff_p,label='positions cut',fmt='.-')
    print 'eff after cut1 = ', np.mean(eff_p[20:35])


cutrec2 = cutrec1 + ' & is_premasked == 0'
cutsim2 = cutsim1 
reccut = cutrec2
simcut = cutsim2
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
    [cbins, eff_p, err_eff_p] = utils.geteffsim(dflowrec_sel,dflowsim_sel,bins,"RUNID!=0","RUNID!=0")
    ax.errorbar(cbins,eff_p,yerr=err_eff_p,label='is_premasked'+str(ext),fmt='.-')
    print 'eff after cut2 = ', np.mean(eff_p[20:35])

cutrec3 = cutrec2 + ' & touchmask == 0'
cutsim3 = cutsim2 
reccut = cutrec3
simcut = cutsim3
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
    [cbins, eff_p, err_eff_p] = utils.geteffsim(dflowrec_sel,dflowsim_sel,bins,"RUNID!=0","RUNID!=0")
    ax.errorbar(cbins,eff_p,yerr=err_eff_p,label='touchmask',fmt='.-')
    print 'eff after cut3 = ', np.mean(eff_p[20:35])


cutrec4 = cutrec3 + ' & is_masked == 0'
cutsim4 = cutsim3 
reccut = cutrec4
simcut = cutsim4
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
    [cbins, eff_p, err_eff_p] = utils.geteffsim(dflowrec_sel,dflowsim_sel,bins,"RUNID!=0","RUNID!=0")
    ax.errorbar(cbins,eff_p,yerr=err_eff_p,label='is_masked',fmt='.-')
    print 'eff after cut4 = ', np.mean(eff_p[20:35])

cutrec5 = cutrec4 + ' & multirows == 0'
cutsim5 = cutsim4 
reccut = cutrec5
simcut = cutsim5
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
    [cbins, eff_p, err_eff_p] = utils.geteffsim(dflowrec_sel,dflowsim_sel,bins,"RUNID!=0","RUNID!=0")
    ax.errorbar(cbins,eff_p,yerr=err_eff_p,label='multirows',fmt='.-')
    print 'eff after cut5 = ', np.mean(eff_p[20:35])

cutrec6 = cutrec5 + ' & ' + constant.llcut + ' & ' +  constant.qmaxcut
cutsim6 = cutsim5 
reccut = cutrec6
simcut = cutsim6
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
    [cbins, eff_p, err_eff_p] = utils.geteffsim(dflowrec_sel,dflowsim_sel,bins,"RUNID!=0","RUNID!=0")
    ax.errorbar(cbins,eff_p,yerr=err_eff_p,label='llcut and qmaxcut',fmt='.-')
    print 'eff after cut6 = ', np.mean(eff_p[20:35])
plt.xlabel('simulated energy [keV]')
plt.ylabel('efficiency')
plt.legend()
plt.show()        





