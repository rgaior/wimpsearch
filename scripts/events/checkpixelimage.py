import ROOT as R
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob as glob
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("infolder", type=str, nargs='?',default='', help="root file folder")

args = parser.parse_args()
infolder = args.infolder

counter = 0
files = glob.glob(infolder+ "*.root*")
columns=['RUNID','EXTID','centerx','centery','val']
df = pd.DataFrame(columns=columns)
for file in files:
    RUNID = int(file[file.find('000_')+4:file.rfind('_')] )
    EXTID = int(file[file.rfind('_')+1:file.rfind('.root')] )
    if counter %100 == 0:
        print 'counter = ',  counter
    counter+=1
    # if RUNID != 3537:
    #     continue

    filename = file
    f = R.TFile(filename)
    im = f.Get("image")
    mask = f.Get("sigma")

    a_bin = np.array([])
    a_entries = np.array([])
    for i in range(4000,9000,1):
        for j in range(0,45,1):
            imval = im.GetBinContent(i,j)
            sig = mask.GetBinContent(i,j)
        #print imval
            if (imval < -150) & (sig > 0):
                print 'i = ' ,i ,' j = ' , j
                dftemp = pd.DataFrame([[RUNID,EXTID,i,j,imval]],columns=columns)
                df = df.append(dftemp)
                #print imval

    f.Close()
df.to_pickle('negativepixel.pkl')
