#################################################################
# this script shows the comparison of the number of clusters ####
# in the data and simulation as a function of the DC         ####
#################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
cwd = os.getcwd()
utilspath = cwd + '/../../utils/'
sys.path.append(utilspath)
import utils
import constant
import argparse
limidbasse = 3559
limid = 3570
DClim = 3
extcut = 0
ext = 4
simDCfolder = '/Users/gaior/DAMIC/data/verifsimDC2/pkl/'
df = pd.read_pickle(simDCfolder + 'test4_sel.pkl')
DCfile = constant.datafolder + '/DC/DCfile.pkl'
dfDC = pd.read_pickle(DCfile)
datadcdf = dfDC
size = datadcdf.shape[0]
datadcdf = datadcdf.merge(df,left_on=['image','ext'], right_on=['RUNID','EXTID'], how='outer')
datadcdf = datadcdf.dropna()
df = datadcdf

df = df.query("sime ==0")
df = df.query("RUNID >" +str(limidbasse))
df = df.query("RUNID < " +str(limid))
df = df.query(constant.basecuts)
df = df.query(constant.badimage)
df = df.query("DC < " +str(DClim ))
if extcut == 1:
    df = df.query("EXTID == " +str(ext))

print '########## simDC ################ '
print 'size = ' , df[df.dll > -50].shape[0]
#print df[df.dll > -50][["RUNID","EXTID","centerx","centery","ene1","touchmask","DC"]]
#print dflowdll
print len(np.unique(df.RUNID))

#simDCfolder = '/Users/gaior/DAMIC/data/verifsimDC/pkl/'
df2 = pd.read_pickle(simDCfolder + 'test3_sel.pkl')
DCfile = constant.datafolder + '/DC/DCfile.pkl'
dfDC = pd.read_pickle(DCfile)
datadcdf = dfDC
size = datadcdf.shape[0]
datadcdf = datadcdf.merge(df2,left_on=['image','ext'], right_on=['RUNID','EXTID'], how='outer')
datadcdf = datadcdf.dropna()
df2 = datadcdf
df2 = df2.query("sime ==0")
df2 = df2.query("RUNID >" +str(limidbasse))
df2 = df2.query("RUNID < " +str(limid))
df2 = df2.query(constant.basecuts)
df2 = df2.query(constant.badimage)
df2 = df2.query("DC < " +str(DClim ))
if extcut == 1:
    df2 = df2.query("EXTID == " +str(ext))

print '########## simDC ################ '
print 'size = ' , df2[df2.dll > -50].shape[0]
#print df2[df2.dll > -50][["RUNID","EXTID","centerx","centery","ene1","touchmask","DC"]]
#print df2lowdll
print len(np.unique(df2.RUNID))


# data:
datafolder = '/Users/gaior/DAMIC/data/postidm2018/'
dfdata = pd.read_pickle(datafolder + 'datapostidm2018.pkl')
DCfile = constant.datafolder + '/DC/DCfile.pkl'
dfDC = pd.read_pickle(DCfile)
datadcdf = dfDC
size = datadcdf.shape[0]
datadcdf = datadcdf.merge(dfdata,left_on=['image','ext'], right_on=['RUNID','EXTID'], how='outer')
datadcdf = datadcdf.dropna()
dfdata = datadcdf
dfdata = dfdata.query("RUNID >" +str(limidbasse))
dfdata = dfdata.query("RUNID < " +str(limid))
dfdata = dfdata.query(constant.basecuts)
dfdata = dfdata.query(constant.badimage)
dfdata = dfdata.query("DC <  " +str(DClim))
if extcut == 1 :
    dfdata = dfdata.query("EXTID == " +str(ext))
print '########## data ################ '
print 'size = ' , dfdata[dfdata.ll -dfdata.llc > -50].shape[0]
print len(np.unique(dfdata.RUNID))


# blank:
datafolder = '/Users/gaior/DAMIC/data/postidm2018/'
dfblank = pd.read_pickle(datafolder + 'blankpostidm2018.pkl')
DCfile = constant.datafolder + '/DC/DCfile.pkl'
dfDC = pd.read_pickle(DCfile)
datadcdf = dfDC
size = datadcdf.shape[0]
datadcdf = datadcdf.merge(dfblank,left_on=['image','ext'], right_on=['RUNID','EXTID'], how='outer')
datadcdf = datadcdf.dropna()
dfblank = datadcdf
dfblank = dfblank.query("RUNID > " +str(limidbasse))
dfblank = dfblank.query("RUNID < " +str(limid))
dfblank = dfblank.query(constant.basecuts)
dfblank = dfblank.query(constant.badimage)
dfblank = dfblank.query("DC <  " +str(DClim))
if extcut == 1:
    dfblank = dfblank.query("EXTID == " +str(ext))
print '########## blank ################ '
print 'size = ' , dfblank[dfblank.ll -dfblank.llc > -50].shape[0]
print len(np.unique(dfblank.RUNID))

dllbins = np.arange(-50,1,0.3)

#plt.hist(dfblank.ll -dfblank.llc,histtype ='step',bins = dllbins,log=True)
plt.hist(dfdata.ll -dfdata.llc,histtype ='step',bins = dllbins,log=True)
#plt.hist(df.dll,bins = dllbins,histtype ='step',log=True)
plt.hist(df2.dll,bins = dllbins,histtype ='step',log=True)

#plt.plot(dfblank.centerx,dfblank.centery,'.')
#plt.plot(df.centerx,df.centery,'+')

#plt.hist(dfblank.centerx,bins = np.arange(4000,9000,100) )
#plt.hist(df.centerx,bins = np.arange(4000,9000,100) )
#plt.hist(dfblank.centery,bins = np.arange(0,43,1) )
#plt.hist(df.centery,bins = np.arange(0,43,1) )
plt.show()
