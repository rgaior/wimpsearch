import os
#os.chdir('/Users/gaior/DAMIC/code/data_analysis/cluster/analyse/dcstudy')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import sys
import os
cwd = os.getcwd()
utilspath = cwd + '/../../utils/'
sys.path.append(utilspath)
import utils
import constant
import glob
import argparse
from scipy.optimize import curve_fit


def expo(x,a,b,c):
    return a*(np.exp((x - b)/c))
def prim_expo(x,a,b,c):
    return a*c*(np.exp((x - b)/c))
def inv_expo(y,a,b,c):
    return  c*np.log(y/a) + b
# returs the x such that the integral of the exponential between x and -inf is y
def inv_intexpo(y,a,b,c):
    return  c*np.log(y/(c*a)) + b
def inv_intexpo2(y,a,b):
    # expo(ax+b)
    return  (1/a)*np.log( (y*a)/b)

def logtofit(x,a,b,c):
    return np.log(a) + (x - b)/c

############################
##  argument parser       ##
############################
parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, nargs='?',default='idm', help="idm or all")
parser.add_argument("blow", type=int, nargs='?',default='', help="low edge of fit range")
parser.add_argument("bup", type=int, nargs='?',default='', help="upper edge of fit range")
parser.add_argument("bsize", type=float, nargs='?',default='', help="binsize")

args = parser.parse_args()
dataset = args.dataset
blow = args.blow
bup = args.bup
bsize = args.bsize

limidbasse = 2000
if dataset == 'all':
    limid = 5000
else:
    limid = 3704

DClim = constant.dclimvalue
extcut = 0
ext = 4
DCfile = constant.datafolder + '/DC/DCfile.pkl'
cuts = "RUNID >" +str(limidbasse) + " & " + "RUNID < " +str(limid)  + " & " + constant.basecuts + " & " +  constant.badimage + " & " + constant.radoncut + " & " + " DC < " +str(DClim )
datafolder = constant.basefolderpostidm

### simulations
simDCfolder = '/Users/gaior/DAMIC/data/simDC2/pkl/'
df = pd.read_pickle(simDCfolder + 'all.pkl')
df = df.query("sime == 0")
df = df.query(cuts)
df = utils.mergewithDC(DCfile,df)
if extcut == 1:
    df = df.query("EXTID == " +str(ext))


#bsize = 0.3
#blow = -18
#bup = -8
allbins = np.arange(-30,0,bsize)

dllbins = allbins[ (allbins> blow) & (allbins < bup) ]

cbins = (dllbins[:-1] + dllbins[1:])/2
[bins,logn,errlogn] = utils.getdatatofit(df,dllbins)

#
# df = df.reset_index()
# df['binned'] = pd.cut(df['dll'],dllbins)
# dllbinned = df.groupby('binned')
#
# n = dllbinned.count().dll
#  dllbinned.dll.mean()

# err = np.sqrt(n)
# logn = np.log(n)
# lnn =  logn.values
# errlnn = 1/np.sqrt(n)
z,V = np.polyfit(bins, logn, 1,w=1/errlogn, full = False ,cov=True )
p = np.poly1d(z)
#
#
# # data:
dfdata = pd.read_pickle(datafolder + 'datapostidm2018.pkl')
dfdata = utils.mergewithDC(DCfile,dfdata)
dfdata = utils.dfbasic(dfdata)
dfdata = dfdata.query(cuts)

[binsdata,logndata,errlogndata] = utils.getdatatofit(dfdata,dllbins)
#
# dfdata = dfdata.reset_index()
# dfdata['binned'] = pd.cut(dfdata['dll'],dllbins)
# dlldatabinned = dfdata.groupby('binned')
# ndata = dlldatabinned.count().dll
# errdata = np.sqrt(ndata)
# logndata = np.log(ndata)
# lnndata =  logndata.values
# errlnndata = 1/np.sqrt(ndata)
zdata,Vdata = np.polyfit(binsdata, logndata, 1,w=1/errlogndata, full = False ,cov=True )
pdata = np.poly1d(zdata)
#
# # integral = 1
a = z[0]
b = z[1]
adata = zdata[0]
bdata = zdata[1]

evnr = 0.1
lnevnr = np.log(evnr)
alphasim = (np.log(a) - b + lnevnr )/(a)
alphadata = (np.log(adata) - bdata + lnevnr )/(adata)
print ('-------------alphasim = ' , alphasim)
print ('-------------alphadata = ' , alphadata)
#
#
#
[binsall,lognall,errlognall] = utils.getdatatofit(df,allbins)
[binsalldata,lognalldata,errlognalldata] = utils.getdatatofit(dfdata,allbins)

plt.errorbar(binsall,lognall,yerr=errlognall,fmt='o',color='k',label='simulation',alpha=0.5)
plt.errorbar(binsalldata,lognalldata,yerr=errlognalldata,fmt='+',color='r',label='data',alpha=0.5)
#plt.errorbar(cbins,logndata,yerr=errlnn,fmt='+',color='r',label='data')
plt.plot(bins,p(bins),'-',lw=2,color='k')
plt.plot(binsdata,pdata(binsdata),'-',lw=2,color='r')
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
# place a text box in upper left in axes coords
textstr = '\n'.join((
    'range: ['+str(blow) +';'+str(bup)+']' + ' binsize = '+str(bsize),
    'dll cut:',
    r'$\mathrm{data}=%.2f$' % (alphadata, ),
    r'$\mathrm{simulation}=%.2f$' % (alphasim, )))
ax = plt.gca()
ax.text(0.05, 0.75, textstr, transform= ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
plt.xlabel('dLL')
plt.ylabel('ln(entries)')
plt.legend()
figfolder = '/Users/gaior/DAMIC/code/plots/'
#figname = figfolder + '20190624/fit_' +dataset+'_' +str(blow)+'_'+str(bup) + '_'+ str(bsize)+'.png'
figname = figfolder + '20190624/fit_' +dataset+'_' +str(blow)+'_'+str(bup) + '_'+ str(bsize)+'.pdf'
#plt.savefig(figname)
plt.show()

#print b, n
