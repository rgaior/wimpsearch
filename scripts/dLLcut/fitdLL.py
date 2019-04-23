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


# get the data from the pkl file:
datafile = constant.datafolder + '/exposed/data1idm.pkl'
data = pd.read_pickle(datafile)
blankfile = constant.datafolder + '/blank/blank1idm.pkl'
blank = pd.read_pickle(blankfile)
#data = data.query("DC < " + str(constant.dclimvalue))
data = data.query("DC < 3")

print "#######################"
print "cutting image with DC larger than ",  constant.dclimvalue, " ADU/1x100 pixel"
print "#######################"



binsize = 0.3
firstbin = -50
lastbin = 0
bins = np.arange(firstbin,lastbin,binsize)
print bins
plt.hist(data.ll - data.llc, bins=bins,log=True)
plt.hist(blank.ll - blank.llc ,bins=bins,log=True)
plt.show()
# rangehigh = [-10]
# rangelow = [-40,-30,-20]
# cols = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
# fixedlow = -18


# #for l,h,col in zip(fixedlow,rangehigh,cols):
# for h,col in zip(rangehigh,cols):
#     bins = np.arange(fixedlow,h,binsize)

#     data = datadcsel.reset_index()
#     datadcsel1['binned'] = pd.cut(datadcsel1['dll'],bins)
#     dllbinned = datadcsel1.groupby('binned')
#     n = dllbinned.count().dll
#     cbins = (bins[:-1] + bins[1:])/2
#     err = np.sqrt(n)
#     integral = 1

# #     # try to fit the log:
#     logn = np.log(n)
#   #  print logn
#     #    print err
#     for i in range(len(n)):
#         if n[i] == 0:
#             err[i] = 1
# #    popt, pcov = curve_fit(logtofit, cbins, logn/integral, sigma=err/integral)
#     logerr = err/n
#     z,V = np.polyfit(cbins, logn,  1,w=1/logerr, full = False ,cov=True )
#     p = np.poly1d(z)
#     a_errfit = np.array([])
#     for x in cbins:
#         a = np.array([x,1])
#         errfit = np.dot(V,a)
#         errfit = np.dot(np.transpose(a),errfit)
#         a_errfit = np.append(a_errfit,errfit)
#     resu = np.exp(p(cbins))
#     errresu = np.sqrt(a_errfit)*resu
#     print errresu
#     plt.errorbar(cbins,resu,yerr=errresu)
#     plt.errorbar(cbins,n,err,fmt='o')
#     plt.plot(cbins,resu + errresu,'--')
#     plt.plot(cbins,resu - errresu,'--')    
#     print inv_intexpo2(0.01,z[0],np.exp(z[1]))
# #    print inv_intexpo(0.01,z)    
# print z
# plt.yscale('log')
# plt.xlabel('dLL')
# plt.ylabel('counts')
