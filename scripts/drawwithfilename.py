import ROOT as R
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
cwd = os.getcwd()
classpath = cwd + '/../classes/'
utilspath = cwd + '/../utils/'
sys.path.append(utilspath)
import utils
import constant
import argparse

############################
##  argument parser       ##
############################
parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, nargs='?',default="", help="file name")
parser.add_argument("centerx", type=int, nargs='?',default=6620, help="center x")
parser.add_argument("centery", type=int, nargs='?',default=22, help="center y")
args = parser.parse_args()
centerx = args.centerx
centery = args.centery
filename = args.filename

f = R.TFile(filename)
#im = f.Get("image_raw")
im = f.Get("image")
centralbin_x = centerx
centralbin_y = centery
delta_x = 20
delta_y = 3
image = utils.getimagepart(im,centralbin_x,centralbin_y,delta_x,delta_y)
#print image[2]
f, axarr = plt.subplots()

fname = filename[filename.rfind('/'):]
f.suptitle(fname)
#plt.imshow(image[2],origin='lower',extent=[np.min(image[0]),np.max(image[0])+1,np.min(image[1]),np.max(image[1])+1],vmin=28,vmax=32,aspect='auto')
plt.imshow(image[2],origin='lower',extent=[np.min(image[0]),np.max(image[0])+1,np.min(image[1]),np.max(image[1])+1],vmin=np.min(image[2]),vmax=np.max(image[2]),aspect='auto')
#plt.imshow(image[2],origin='lower',extent=[np.min(image[0]),np.max(image[0])+1,np.min(image[1]),np.max(image[1])+1],aspect='auto')

#for (j,i),label in np.ndenumerate(image[2]):
#    print label
#    axarr.text(centralbin_x + i - delta_x + 0.5, centralbin_y + j - delta_y + 0.5,int(label),ha='center',va='center')

axarr.set_xlabel('X [pixel]')
axarr.set_ylabel('Y [pixel]')

plt.colorbar()

plt.show()
