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

# parser = argparse.ArgumentParser()
# parser.add_argument("totor", type=str, nargs='?',default='', help="pickle file with the dataframe")
# parser.add_argument("--cut", type=str, nargs='?', default="RUNID!=0",help="additional cut to be separated as in a dataframe query selection")

# args = parser.parse_args()
# filename = args.infile
# cut = args.cut

limidbasse = 2000
limid = 5000
#limid = 3704

DClim = constant.dclimvalue
extcut = 0
ext = 4
DCfile = constant.datafolder + '/DC/DCfile.pkl'
#cuts = "RUNID >" +str(limidbasse) + " & " + "RUNID < " +str(limid)  + " & " + constant.basecuts + " & " +  constant.badimage + " & " + constant.radoncut + " & " +  " DC < " +str(DClim )
cuts = constant.badimage + " & " + constant.radoncut + " & " +  " DC < " +str(DClim ) + " & " + constant.negativepixelimage
#cuts = " DC < " +str(DClim )
#cuts = "RUNID > 0"
datafolder = constant.basefolderpostidm


dfdata = pd.read_pickle(datafolder + 'datapostidm2018.pkl')
dfdata = utils.mergewithDC(DCfile,dfdata)
dfdata = dfdata.query(cuts)
images = dfdata.groupby(["RUNID","EXTID"])
totexposure = 0
a_runid = np.array([])
a_exposure = np.array([])
prev_runid = 2474
int_expo = 0
df =dfdata
nrofid = np.unique(df.RUNID)
lastid = np.max(df['RUNID'])
size = len(images)
counter = 0
for name, group in images:
    counter +=1
    runid = name[0]
    exposure = float(np.unique(group.EXPTIME)[0])
#    print exposure
    totexposure += exposure    
    print name[0]
    if runid != prev_runid:        
        a_exposure = np.append(a_exposure,int_expo)
        a_runid = np.append(a_runid, prev_runid)
        print 'int_expo =  ', int_expo
        prev_runid = runid 
    if counter == size:
        int_expo += exposure
        a_exposure = np.append(a_exposure,int_expo)
        print 'runid = ' , runid
        print 'prev_runid = ' , prev_runid
        a_runid = np.append(a_runid, prev_runid)
        print counter
        print 'last image expo =' , int_expo

    int_expo += exposure

print totexposure
print len(images)
outfile = '/Users/gaior/DAMIC/code/wimpsearch/data/Alex/' + 'idvsintexpo.npz'
np.savez(outfile, runid=a_runid, int_expo=a_exposure)
plt.plot(a_runid,a_exposure)
plt.show()
