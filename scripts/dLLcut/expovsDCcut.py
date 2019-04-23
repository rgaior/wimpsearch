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


############################
##  argument parser       ##
############################
parser = argparse.ArgumentParser()
parser.add_argument("tag", type=str, nargs='?',default='IDM', help="define IDM or not")
args = parser.parse_args()
tag = args.tag

if tag == 'IDM':
    datafile = constant.datafolder + '/exposed/data1idm.pkl'
    data = pd.read_pickle(datafile)
    blankfile = constant.datafolder + '/blank/blank1idm.pkl'
    blank = pd.read_pickle(blankfile)
else:
    datafile = constant.datafolder + '/exposed/data1.pkl'
    data = pd.read_pickle(datafile)
    blankfile = constant.datafolder + '/blank/blank1.pkl'
    blank = pd.read_pickle(blankfile)

######################################################
#define the range in dLL where we count the cluster ##
######################################################
dll_low = -17
dll_high = 0
data = data.query("dll > " +str(dll_low) + " & dll < " +str(dll_high))
blank = blank.query("dll > " +str(dll_low) + " & dll < " +str(dll_high))
print data.shape[0]
print blank.shape[0]
#a_dc = np.array([0.5,1,2,3,4,5,6,7,8,9,10,20,30])
a_dc = np.array([0.5,2,5])
dllbins = np.arange(dll_low,dll_high,0.5)
for dc in a_dc:
    datatemp = data.query("DC< " +str(dc))
    blanktemp = blank.query("DC< " +str(dc))    
    print 'DC= ', dc, ' data : ' , datatemp.shape[0], ' blank: ' , blanktemp.shape[0]
    plt.figure()
    nd,bd,pd = plt.hist(datatemp.dll,bins=dllbins,log=True)
    nb,bb,pb = plt.hist(blanktemp.dll,bins=dllbins,log=True)

    plt.figure()
    plt.plot( (bd[:-1]+bd[1:])/2,(nd-nb)/nb )

plt.show()

