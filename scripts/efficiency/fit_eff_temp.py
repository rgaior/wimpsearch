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


from scipy.optimize import curve_fit
def eff_function(x,a,b,c,d,e,f,g):
    eff = (1./(1+np.exp(-(x-a)*b))  - c ) *((d*x +e)+(f*np.exp(-g*x)))
    return eff

def eff_functionfit(x, eff, error=None):
    # parameters in Javier's code for signal eff 1x100
    a=0.159514
    b=20.2396
    c=0.0752355
    d=0.00467332
    e=0.503398
    f=0.157751
    g= 1.05736
    aa = 8.27531595e-02
    bb = 7.47224752e+01
    cc = 2.75889754e-02
    dd =  -1.44252052e-02
    ee = 4.60813172e-01
    ff = 3.24463575e-01
    gg = -1.23398725e-04
#    init_guess = [a,b,c,d,e,f,g]
    init_guess = [aa,bb,cc,dd,ee,ff,gg]    
#    if error!=None:
    popt, pcov = curve_fit(eff_function, x, eff,sigma=error,p0=init_guess)
    return [popt,pcov]


# get the data
#the data were simuated in two steps, a first batch of simulation 
# was done at low energy with an exponential energy distribution
# a second batch was done with a flat distribution in energy.
# for each batch we have the reconstructed and simulated trees.
dflowrec = pd.read_pickle(constant.datafolder + '/efficiency/lowErec.pkl')
dflowsim = pd.read_pickle(constant.datafolder + '/efficiency/lowEsim.pkl')

dfhighrec = pd.read_pickle(constant.datafolder + '/efficiency/highErec.pkl')
dfhighsim = pd.read_pickle(constant.datafolder + '/efficiency/highEsim.pkl')

#fig = plt.figure(figsize=(8,6))
#ax = plt.subplot(2,1)
#f, axarr = plt.subplots(2, sharex=True)
#f.suptitle('Sharing X axis')
f, (ax0, ax1) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3,1]},sharex=True)
col = {1:'black',2:'red',3:'blue',4:'green',6:'gray',11:'darkseagreen',12:'lightskyblue',0:'darkorange'}
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
    bins = np.linspace(0.03,0.8,55)
    bins_high = np.linspace(0.65,9,70)
#    [cbins, eff_p, err_eff_p] = utils.geteffsim(dflowrec_sel,dflowsim_sel,bins,"RUNID!=0","RUNID!=0")
    [cbins, eff_p, err_eff_p] = utils.geteff(dflowrec_sel,dflowsim_sel,bins,"RUNID!=0","RUNID!=0")
    [cbins_high, eff_p_high, err_eff_p_high] = utils.geteff(dfhighrec_sel,dfhighsim_sel,bins_high,"RUNID!=0","RUNID!=0")
    cbins_all = cbins
    cbins_all = np.append(cbins_all,cbins_high)
    eff_all = eff_p
    eff_all = np.append(eff_all,eff_p_high)
    err_eff_all = err_eff_p
    err_eff_all = np.append(err_eff_all,err_eff_p_high)



#popt = eff_functionfit(a_e, a_eff, a_eff_err)
#    [popt, pcov] = eff_functionfit(cbins_all, eff_all,err_eff_all)
#    print popt
#    print np.sqrt(np.diag(pcov))
#fig = plt.figure()
    if ext == 0:
        extstring = 'all ext'
    else:
        extstring = str(ext)
    ax0.errorbar(cbins_all, eff_all,err_eff_all,fmt='.',color=col[ext])
#    ax0.plot(cbins_all,eff_function(cbins_all,*popt),lw=2,label='ext= ' + extstring,color=col[ext])
#    ax1.plot(cbins_all, (eff_all - eff_function(cbins_all,*popt) )/err_eff_all,'.',color=col[ext])

#    ax.errorbar(cbins,eff_p,yerr=err_eff_p,label='ext: '+str(ext),fmt='.-')
#    ax.errorbar(cbins_high,eff_p_high,yerr=err_eff_p_high,label='ext: '+str(ext),fmt='.-')
#    ax.set_xlim(0,9) 




#print a_eff



#covmat = np.cov(cbins_all,)
#err = np.dot(pcov,cbins_all.T)
#err = np.dot(cbins_all,err)
#print err
plt.xlim(0,5)
ax1.set_xlabel('reconstructed energy [keV]')
ax0.set_ylabel('efficiency')
ax1.set_ylabel('residuals [sigma]')
ax0.legend()
plt.show()



# fig = plt.figure()
# msim = 4116*1.5e-3*42*1.5e-1*0.0669*2.33 
# plt.errorbar(a_e, msim*a_eff, msim*a_eff_err,fmt='.')
# plt.plot(a_e,msim*eff_function(a_e,*popt),lw=2,label='fit')
# plt.xlim(0,5)
# plt.xlabel('reconstructed energy [keV]')
# plt.ylabel('mass acceptance')


# fig = plt.figure()

# plt.errorbar(a_e, a_eff, a_eff_err,fmt='.')
# plt.plot(a_e,eff_function(a_e,*popt),lw=2,label='fit')
# plt.xlabel('reconstructed energy [keV]')
# plt.ylabel('efficiency')
# plt.xlim(0,0.4)

# print popt


# plt.legend()
# plt.show()        





