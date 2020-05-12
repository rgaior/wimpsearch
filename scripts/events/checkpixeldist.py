import ROOT as R
import numpy as np
import matplotlib.pyplot as plt
import glob as glob
#folder = '/Users/gaior/DAMIC/data/simDC2/after_pfs/140_30_4/'
folder = '/Users/gaior/DAMIC/data/postidm2018/rootc/'
files = glob.glob(folder + "*.root")
counter = 0
for file in files:
    if counter %100 == 0:
        print counter
    counter+=1
    filename = file
    f = R.TFile(filename)
    pixhist = f.Get("pixel_distribution_masked")
    a_bin = np.array([])
    a_entries = np.array([])
    for i in range(10000):
        bin = pixhist.GetBinCenter(i)
        a_bin = np.append(a_bin,bin)
        entries = pixhist.GetBinContent(i)
        a_entries = np.append(a_entries,entries)
        if (bin < -150) & (entries > 0):
            print filename
            print 'bin = ' , bin , ' entries = ',entries
    f.Close()
