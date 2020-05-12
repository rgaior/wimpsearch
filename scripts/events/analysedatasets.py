

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

##########################
## import data set
## argument parser
############################
parser = argparse.ArgumentParser()
parser.add_argument("infile", type=str, nargs='?',default='', help="pickle file with the dataframe")
parser.add_argument("infile2", type=str, nargs='?',default='', help="pickle file with the dataframe")
parser.add_argument("--cut", type=str, nargs='?', default="RUNID!=0",help="additional cut to be separated as in a dataframe query selection")

args = parser.parse_args()
filename = args.infile
filename2 = args.infile2
cut = args.cut

limidbasse = 2000
limid = 4150
DClim = constant.dclimvalue
DCfile = constant.datafolder + '/DC/DCfile.pkl'
cuts = "RUNID >" +str(limidbasse) + " & " + "RUNID < " +str(limid)  + " & " + constant.basecuts + " & " +  constant.badimage +  " & " +  constant.negativepixel + " & " + constant.radoncut + " & " + " DC < " +str(DClim) +" & sime==0"

df = pd.read_pickle(filename)
#df = utils.mergewithDC(DCfile,df)
print df.columns
df = df.query(cuts)


df2 = pd.read_pickle(filename2)

df2 = utils.mergewithDC(DCfile,df2)
print df2.columns
df2 = df2.query(cuts)
print 'additional cut : ' , cut
df = df.query(cut)
df2 = df2.query(cut)
############################
## plot various distributions
#############################

#df = df.dropna()
#df2 = df2.dropna()

######## distribution  in Energy
f = plt.figure()
bins_ene = np.arange(0.03,0.2,0.01)

plt.hist(df.ene1,bins=bins_ene,histtype = 'step', lw=2)
plt.hist(df2.ene1,bins=bins_ene,histtype = 'step', lw=2)
plt.xlabel("energy")

######## distribution  in sigma
f = plt.figure()
bins_sigma = np.arange(0.,2,0.01)
plt.hist(df.sigma,bins=bins_sigma,histtype = 'step', lw=2)
plt.hist(df2.sigma,bins=bins_sigma,histtype = 'step', lw=2)
plt.xlabel("sigma")
print df.shape[0]

######## distribution  in dLL
f = plt.figure()
bins_dll = np.arange(-50,0,0.3)
plt.hist(df.dll,bins=bins_dll,histtype = 'step', lw=2,log=True)
plt.hist(df2.dll,bins=bins_dll,histtype = 'step', lw=2,log=True)
plt.xlabel("dLL")


######## group by dll
bins = pd.cut(df['dll'],np.arange(-50,0,3))
b_ene = df.groupby(bins)['ene1'].agg(['count', 'mean','std'])
b_x = df.groupby(bins)['centerx'].agg(['count', 'mean','std'])
b_y = df.groupby(bins)['centery'].agg(['count', 'mean','std'])
b_sigma = df.groupby(bins)['sigma'].agg(['count', 'mean','std'])
b_dll = df.groupby(bins)['dll'].agg(['count', 'mean','std'])
b_ene = b_ene.dropna(axis=0,thresh=2)
b_dll = b_dll.dropna(axis=0,thresh=2)
b_x = b_x.dropna(axis=0,thresh=2)
b_y = b_y.dropna(axis=0,thresh=2)
b_sigma = b_sigma.dropna(axis=0,thresh=2)
std_ene = np.nan_to_num(b_ene['std'])
std_x = np.nan_to_num(b_x['std'])
std_y = np.nan_to_num(b_y['std'])
std_sigma = np.nan_to_num(b_sigma['std'])

######## group by dll
bins2 = pd.cut(df2['dll'],np.arange(-50,0,3))
b_ene2 = df2.groupby(bins2)['ene1'].agg(['count', 'mean','std'])
b_x2 = df2.groupby(bins2)['centerx'].agg(['count', 'mean','std'])
b_y2 = df2.groupby(bins2)['centery'].agg(['count', 'mean','std'])
b_sigma2 = df2.groupby(bins2)['sigma'].agg(['count', 'mean','std'])
b_dll2 = df2.groupby(bins2)['dll'].agg(['count', 'mean','std'])
b_ene2 = b_ene2.dropna(axis=0,thresh=2)
b_dll2 = b_dll2.dropna(axis=0,thresh=2)
b_x2 = b_x2.dropna(axis=0,thresh=2)
b_y2 = b_y2.dropna(axis=0,thresh=2)
b_sigma2 = b_sigma2.dropna(axis=0,thresh=2)
std_ene2 = np.nan_to_num(b_ene2['std'])
std_x2 = np.nan_to_num(b_x2['std'])
std_y2 = np.nan_to_num(b_y2['std'])
std_sigma2 = np.nan_to_num(b_sigma2['std'])



f = plt.figure()
plt.errorbar(b_dll['mean'],b_ene['mean'],yerr=std_ene,fmt='o')
plt.errorbar(b_dll2['mean'],b_ene2['mean'],yerr=std_ene2,fmt='o')
plt.xlabel('dLL')
plt.ylabel('mean energy')

f = plt.figure()
plt.errorbar(b_dll['mean'],b_x['mean'],yerr=std_x,fmt='o')
plt.errorbar(b_dll2['mean'],b_x2['mean'],yerr=std_x2,fmt='o')
plt.xlabel('dLL')
plt.ylabel('mean centerx')


f = plt.figure()
plt.errorbar(b_dll['mean'],b_y['mean'],yerr=std_y,fmt='o')
plt.errorbar(b_dll2['mean'],b_y2['mean'],yerr=std_y2,fmt='o')
plt.xlabel('dLL')
plt.ylabel('mean centery')

f = plt.figure()
plt.errorbar(b_dll['mean'],b_sigma['mean'],yerr=std_sigma,fmt='o')
plt.errorbar(b_dll2['mean'],b_sigma2['mean'],yerr=std_sigma2,fmt='o')
plt.xlabel('dLL')
plt.ylabel('mean sigma')

f = plt.figure()
bins_dll = np.arange(-50,0,0.3)
plt.hist(df.dll,bins=bins_dll,histtype = 'step', lw=2,log=True)
plt.hist(df2.dll,bins=bins_dll,histtype = 'step', lw=2,log=True)
plt.xlabel("dLL")

print df[['RUNID','EXTID','centerx','centery','ene1','sigma','DC']]
for i, row in df.iterrows():
    filename = glob.glob(constant.basefolderpostidm + "/rootc/" + "*" + str(int(row["RUNID"])) +"*_"+str(int(row["EXTID"]))+".root")
    print filename
    print 'EventBrowser ' , "-c " + str(row["cid"]) + " " + filename[0]

plt.show()












#bins_x = np.arange(0,5000)
#plt.plot(df.)
#
#
# from scipy.special import factorial
# def poisson(x,lam,k):
#     return k*(np.power(lam,x)*np.exp(-lam)/factorial(x))
# #x = np.arange(0,100,1)
# #plt.plot(x,poisson(x,10))
# from scipy.optimize import curve_fit
# #popt30, pcov30 = curve_fit(poisson, cbins30, n30,sigma = errn30,absolute_sigma=True)
#
# ev30ks = {1:[382,  49,   4,],2:[264,  76,   9,   1,   1,],3: [369,  66,   7,],4:[361,  74,   7,],6:[352,  84,   6,], 11:[363,  72,   5,   2,],12:[249,  71,   6,   1,   1,]}
# ev100ks = {1: [25,  9,  1,],2:[13, 11,  4,  2,  0,  1,], 3:[21,  9,  5,],4:[14, 16,  4,  1,], 6:[10, 16,  7,  2,], 11:[16, 16,  2,  1,]}
# exts = [1,2,3,4,6,11,12]
#
# f, ((ax1, ax2, ax3, ax4), (ax6, ax11 ,ax12, axtot)) = plt.subplots(2, 4,  sharey='row',figsize=(12,8))
# axdict = {1:ax1,2:ax2,3:ax3,4:ax4,6:ax6,11:ax11,12:ax12,13:axtot}
#
# for ext in exts:
#     print 'exxt = ' , ext
# #    f = plt.figure()
#     b30 = np.arange(0,len(ev30ks[ext]),1)
#     n30 = ev30ks[ext]
#     errn30 = np.sqrt(n30)
#     errn30[errn30==0] = 1
#     popt30, pcov30 = curve_fit(poisson, b30, n30,sigma = errn30,absolute_sigma=True)
#
#
#     axdict[ext].errorbar(b30,n30,yerr=errn30,c='b',fmt='o')
#     axdict[ext].plot(np.arange(0,6,1),poisson(np.arange(0,6,1),*popt30),'-',c='b',label='fit 30ks')
#     chi2 = np.sum((n30 - poisson(b30,*popt30))**2/errn30)
#     print 'chi2 = ', chi2/(len(n30) + 1)
#     if (ext!=12):
#         b100 = np.arange(0,len(ev100ks[ext]),1)
#         n100 = ev100ks[ext]
#         errn100 = np.sqrt(n100)
#         errn100[errn100==0] = 1
#         popt100, pcov100 = curve_fit(poisson, b100, n100,sigma = errn100,absolute_sigma=True)
#
#         axdict[ext].errorbar(b100,n100,yerr=errn100,c='r',fmt='o')
#         axdict[ext].plot(np.arange(0,6,1),poisson(np.arange(0,6,1),*popt100),'--',c='r',label='fit 100ks')
#
#         chi2 = np.sum((n100 - poisson(b100,*popt100))**2/errn100)
#         print 'chi2 = ', chi2/(len(n100) + 1)
#
#     axdict[ext].legend()
#     axdict[ext].set_yscale('log',nonposy='clip')
#     axdict[ext].set_ylim(1e-1,500)
#     axdict[ext].set_xlim(0,5)
#     axdict[ext].set_title('ext: ' +str(ext))
#
# plt.show()
# # n30 =
# # mean30 =  0.1310344827586207
# # n100 =  [25.  9.  1.]
# # ext =  2
# # n30 =  [264.  76.   9.   1.   1.]
# # mean30 =  0.28774928774928776
# # n100 =  [13. 11.  4.  2.  0.  1.]
# # ext =  3
# # n30 =  [369.  66.   7.]
# # mean30 =  0.18099547511312217
# # n100 =  [21.  9.  5.]
# # ext =  4
# # n30 =  [361.  74.   7.]
# # mean30 =  0.19909502262443438
# # n100 =  [14. 16.  4.  1.]
# # ext =  6
# # n30 =  [352.  84.   6.]
# # mean30 =  0.2171945701357466
# # n100 =  [10. 16.  7.  2.]
# # ext =  11
# # n30 =  [363.  72.   5.   2.]
# # mean30 =  0.19909502262443438
# # n100 =  [16. 16.  2.  1.]
# # ext =  12
# # n30 =  [249.  71.   6.   1.   1.]
