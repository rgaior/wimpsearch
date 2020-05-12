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
parser.add_argument("--cut", type=str, nargs='?', default="RUNID!=0",help="additional cut to be separated as in a dataframe query selection")

args = parser.parse_args()
filename = args.infile
cut = args.cut

df = pd.read_pickle(filename)
DClim = constant.dclimvalue
DCfile = constant.datafolder + '/DC/DCfile.pkl'

#df = utils.mergewithDC(DCfile,df)
#print df.columns
print ('additional cut : ' , cut)
limidbasse = 2000
#limid = 5000
limid = 3704
DClim = constant.dclimvalue
DCfile = constant.datafolder + '/DC/DCfile.pkl'

cuts = "RUNID >" +str(limidbasse) + " & " + "RUNID < " +str(limid)  + " & " + constant.basecuts + " & " + constant.positioncut  + " & " +  constant.badimage + " & " + constant.radoncut + " & " + " DC < " +str(DClim ) +" & sime==0 " +  " & " +  constant.negativepixelimage 

#df = utils.mergewithDC(DCfile,df)
############################
## plot various distributions
#############################

######## distribution  in X Y
df = df.dropna()
#df = utils.dfbasic(df)
df = df.query(cuts)
df = df.query(cut)

listofnpix = np.unique(df.npix)
data_to_plot = []
for npix in listofnpix:
    data = df[df.npix== npix].sigma.values
    data_to_plot.append(data) 


listofnpix4 = np.unique(df.npix4)
data_to_plot4 = []
for npix4 in listofnpix4:
    data = df[df.npix4== npix4].sigma.values
    data_to_plot4.append(data) 


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6),sharey=True)
bp = axes[0].violinplot(data_to_plot,listofnpix)
axes[0].plot(df.npix,df.sigma,'.',c="blue",alpha=0.3)
axes[0].set_xlabel("npix")
axes[0].set_ylabel("sigma")
bp = axes[1].violinplot(data_to_plot4,listofnpix4)
axes[1].plot(df.npix4,df.sigma,'.',c="blue",alpha=0.3)
axes[1].set_xlabel("npix4")

plt.show()
