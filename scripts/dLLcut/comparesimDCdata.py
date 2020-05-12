#################################################################
# this script shows the comparison of the number of clusters ####
# in the data and simulation as a function of the DC         ####
#################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import sys
import os
cwd = os.getcwd()
utilspath = cwd + '/../../utils/'
sys.path.append(utilspath)
import utils
import constant
import argparse
limidbasse = 2000
limid = 3704
DClim = constant.dclimvalue

#DClim =
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

print '########## simDC ################ '
print 'size = ' , df[df.ll - df.llc > -50].shape[0]
print 'size of large dLL = ' , df[df.ll - df.llc < -25].shape[0]
dflarge = df.query("dll > -26 & dll < -18")
print dflarge[['centerx','centery','dll','RUNID','EXTID','touchmask','linlength','ene1']]
#print df[df.dll > -50][["RUNID","EXTID","centerx","centery","ene1","touchmask","DC"]]
#print dflowdll

#print df.groupby(['RUNID','EXTID']).size()
print np.unique(df.RUNID)


# data:
dfdata = pd.read_pickle(datafolder + 'datapostidm2018.pkl')
dfdata = utils.mergewithDC(DCfile,dfdata)
dfdata = dfdata.query(cuts)
if extcut == 1 :
    dfdata = dfdata.query("EXTID == " +str(ext))
print '########## data ################ '
print 'size = ' , dfdata[dfdata.ll -dfdata.llc > -50].shape[0]
print np.unique(dfdata.RUNID)
dflargedata = dfdata.query("ll-llc > -26 & ll-llc < -18")
print dflargedata[['centerx','centery','ll','llc','RUNID','EXTID','touchmask','linlength','ene1']]

# blank:
dfblank = pd.read_pickle(datafolder + 'blankpostidm2018.pkl')
dfblank = utils.mergewithDC(DCfile,dfblank)
dfblank = dfblank.query(cuts)
if extcut == 1:
    dfblank = dfblank.query("EXTID == " +str(ext))

print '########## blank ################ '
print 'size = ' , dfblank[dfblank.ll -dfblank.llc > -50].shape[0]
dflarge = dfblank[dfblank.ll - dfblank.llc < -25]
print dflarge[['centerx','centery','llc','RUNID','EXTID','touchmask','linlength']]
print len(np.unique(dfblank.RUNID))

dllbins = np.arange(-50,1,0.5)
f = plt.figure()
ndata, bdata,p = plt.hist(dfdata.ll -dfdata.llc,bins = dllbins,color='r',alpha=0.2)
nsim, bsim,p  = plt.hist(df.dll,bins = dllbins,color='k',alpha=0.2)
nblank, bblank,p  = plt.hist(dfblank.ll -dfblank.llc,bins = dllbins,color='b',alpha=0.1)
f = plt.figure()
#ndata, bdata,p = plt.hist(dfdata.ll -dfdata.llc,bins = dllbins,color='r',alpha=0.5)
#nsim, bsim,p  = plt.hist(df.dll,bins = dllbins,color='k',alpha=0.2)

# Plot figure with subplots of different sizes
fig = plt.figure(1)
#fig, ax = plt.subplots(2,1 , sharex='col')
# set up subplot grid
gridspec.GridSpec(4,1)

# large subplot
plt.subplot2grid((4,1), (0,0), colspan=1, rowspan=3)
#plt.locator_params(axis='x', nbins=5)
#plt.locator_params(axis='y', nbins=5)
plt.xlabel('')
plt.ylabel('Entries')
plt.errorbar((bblank[0:-1]+bblank[1:])/2,nblank,yerr=np.sqrt(nblank),fmt='o',color='b',alpha=0.3,label="blank")
plt.errorbar((bsim[0:-1]+bsim[1:])/2,nsim,yerr=np.sqrt(nsim),fmt='o',color='k',alpha=0.5,label="blank + DC")
plt.errorbar((bdata[0:-1]+bdata[1:])/2,ndata,yerr=np.sqrt(ndata),fmt='o',color='r',alpha=0.5,label="data")
plt.yscale('log')
plt.subplot2grid((4,1), (3,0))
plt.plot((dllbins[:-1]+dllbins[1:]), (ndata - nsim)/ndata,'o',color='k',alpha=0.2)
plt.plot((dllbins[:-1]+dllbins[1:]), (ndata - nblank)/ndata,'o',color='b',alpha=0.2)
#nsim, bsim,p  = plt.hist(df.dll,bins = dllbins,color='k',alpha=0.2)
#nblank, bblank,p  = plt.hist(dfblank.ll -dfblank.llc,bins = dllbins,color='b',alpha=0.1)

#plt.locator_params(axis='x', nbins=5)
#plt.locator_params(axis='y', nbins=5)
plt.ylabel('difference w.r.t. data')

fig = plt.figure()
#plt.errorbar((bblank[0:-1]+bblank[1:])/2,nblank,yerr=np.sqrt(nblank),fmt='o',color='b',alpha=0.3,label="blank")
#plt.errorbar((bsim[0:-1]+bsim[1:])/2,nsim,yerr=np.sqrt(nsim),fmt='o',color='k',alpha=0.5,label="blank + DC")
plt.errorbar((bdata[0:-1]+bdata[1:])/2,ndata,yerr=np.sqrt(ndata),fmt='o',color='r',alpha=0.5,label="data")


plt.yscale('log',nonposy='clip')
plt.legend(loc=2)

#plt.hist(df.dll,histtype ='step',log=True)

#plt.plot(dfblank.centerx,dfblank.centery,'.')
#plt.plot(df.centerx,df.centery,'+')

#plt.hist(dfblank.centerx,bins = np.arange(4000,9000,100) )
#plt.hist(df.centerx,bins = np.arange(4000,9000,100) )
#plt.hist(dfblank.centery,bins = np.arange(0,43,1) )
#plt.hist(df.centery,bins = np.arange(0,43,1) )
plt.show()
