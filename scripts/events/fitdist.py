

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
from scipy.special import factorial
def poisson(x,lam,k):
    return k*(np.power(lam,x)*np.exp(-lam)/factorial(x))
#x = np.arange(0,100,1)
#plt.plot(x,poisson(x,10))
from scipy.optimize import curve_fit
#popt30, pcov30 = curve_fit(poisson, cbins30, n30,sigma = errn30,absolute_sigma=True)

ev30ks = {1:[382,  49,   4,],2:[264,  76,   9,   1,   1,],3: [369,  66,   7,],4:[361,  74,   7,],6:[352,  84,   6,], 11:[363,  72,   5,   2,],12:[249,  71,   6,   1,   1,]}
ev100ks = {1: [25,  9,  1,],2:[13, 11,  4,  2,  0,  1,], 3:[21,  9,  5,],4:[14, 16,  4,  1,], 6:[10, 16,  7,  2,], 11:[16, 16,  2,  1,]}
exts = [1,2,3,4,6,11,12]

f, ((ax1, ax2, ax3, ax4), (ax6, ax11 ,ax12, axtot)) = plt.subplots(2, 4,  sharey='row',figsize=(12,8))
axdict = {1:ax1,2:ax2,3:ax3,4:ax4,6:ax6,11:ax11,12:ax12,13:axtot}

for ext in exts:
    print 'exxt = ' , ext
#    f = plt.figure()
    b30 = np.arange(0,len(ev30ks[ext]),1)
    n30 = ev30ks[ext]
    errn30 = np.sqrt(n30)
    errn30[errn30==0] = 1
    popt30, pcov30 = curve_fit(poisson, b30, n30,sigma = errn30,absolute_sigma=True)

    
    axdict[ext].errorbar(b30,n30,yerr=errn30,c='b',fmt='o')
    axdict[ext].plot(np.arange(0,6,1),poisson(np.arange(0,6,1),*popt30),'-',c='b',label='fit 30ks')
    chi2 = np.sum((n30 - poisson(b30,*popt30))**2/errn30)
    print 'chi2 = ', chi2/(len(n30) + 1)
    if (ext!=12):
        b100 = np.arange(0,len(ev100ks[ext]),1)
        n100 = ev100ks[ext]
        errn100 = np.sqrt(n100)
        errn100[errn100==0] = 1
        popt100, pcov100 = curve_fit(poisson, b100, n100,sigma = errn100,absolute_sigma=True)
        
        axdict[ext].errorbar(b100,n100,yerr=errn100,c='r',fmt='o')
        axdict[ext].plot(np.arange(0,6,1),poisson(np.arange(0,6,1),*popt100),'--',c='r',label='fit 100ks')

        chi2 = np.sum((n100 - poisson(b100,*popt100))**2/errn100)
        print 'chi2 = ', chi2/(len(n100) + 1)
    
    axdict[ext].legend()
    axdict[ext].set_yscale('log',nonposy='clip')
    axdict[ext].set_ylim(1e-1,500)
    axdict[ext].set_xlim(0,5)
    axdict[ext].set_title('ext: ' +str(ext))

plt.show()
# n30 =  
# mean30 =  0.1310344827586207
# n100 =  [25.  9.  1.]
# ext =  2
# n30 =  [264.  76.   9.   1.   1.]
# mean30 =  0.28774928774928776
# n100 =  [13. 11.  4.  2.  0.  1.]
# ext =  3
# n30 =  [369.  66.   7.]
# mean30 =  0.18099547511312217
# n100 =  [21.  9.  5.]
# ext =  4
# n30 =  [361.  74.   7.]
# mean30 =  0.19909502262443438
# n100 =  [14. 16.  4.  1.]
# ext =  6
# n30 =  [352.  84.   6.]
# mean30 =  0.2171945701357466
# n100 =  [10. 16.  7.  2.]
# ext =  11
# n30 =  [363.  72.   5.   2.]
# mean30 =  0.19909502262443438
# n100 =  [16. 16.  2.  1.]
# ext =  12
# n30 =  [249.  71.   6.   1.   1.]
