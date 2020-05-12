

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
print 'additional cut : ' , cut
limidbasse = 2000
limid = 5000
DClim = constant.dclimvalue
DCfile = constant.datafolder + '/DC/DCfile.pkl'
#cuts = "RUNID >" +str(limidbasse) + " & " + "RUNID < " +str(limid)  + " & " + constant.basecuts + " & " +  constant.badimage + " & " + constant.radoncut + " & " + constant.negativepixelimage + " & " + " DC < " +str(DClim ) +" & sime==0"
cuts = "RUNID >" +str(limidbasse) + " & " + "RUNID < " +str(limid)  + " & " + constant.basecuts + " & " +  constant.badimage + " & " + constant.radoncut + " & " + " DC < " +str(DClim ) +" & sime==0"


df = df.query(cuts)
df = df.query(cut)
############################
## plot various distributions
#############################

######## distribution  in X Y
df = df.dropna()
#df = utils.dfbasic(df)
#f = utils.getimageplot(df.centerx,df.centery,100)
print 'size of df = ' , df.shape[0]

alexfile = '/Users/gaior/Downloads/forRomain_2.txt'
alexf = open(alexfile,'r')
alexevents = alexf.readlines()
for l in alexevents:
    l = l.split()
    a = df.query('RUNID == ' +str(l[0]) + ' & EXTID == ' +str(l[1]) + ' & cid == ' +str(l[2]) )
    if a.shape[0] == 0:
        print ' akjsn ', l

for i, row in df.iterrows():
    found = False
    for l in alexevents:
        l = l.split()
        if ( (int(l[0]) == row["RUNID"] ) & (int(l[1]) == row["EXTID"]) & ( int(l[2])== row["cid"] ) ):
#            print l
#            print row
            found = True
#    if found == False:
#        print 'counldnt find ', row['RUNID'] , ' ', row['EXTID'], ' ', row['cid'], ' ' , row['ene1']        
        
#    print row["RUNID"]

#print alexevents
#print np.sort(df.RUNID)

df.to_pickle("/Users/gaior/DAMIC/code/wimpsearch/data/Alex/selections.pkl")
