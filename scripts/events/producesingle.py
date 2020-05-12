####################################################################################
#### produce the png files for the clusters in the file given in the input  ########
#### adds a columns with the path to that file in the dataframe             ########
####################################################################################
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
## argument parser
############################
parser = argparse.ArgumentParser()
parser.add_argument("infile", type=str, nargs='?',default='', help="pickle file with the dataframe")

args = parser.parse_args()
filename = args.infile
df = pd.read_pickle(filename)

rootfolder = constant.basefolderpostidm + '/rootc/'

for i, row in df.iterrows():
    run = int(row["RUNID"])
    ext = int(row["EXTID"])
    cid = int(row["cid"])
    filename = glob.glob(constant.basefolderpostidm + "/rootc/" + "*" + str(int(row["RUNID"])) +"*_"+str(int(row["EXTID"]))+".root")
    f = R.TFile(filename)
#im = f.Get("image_raw")                                                                                im = f.Get("image")
    centralbin_x = centerx
    centralbin_y = centery
    delta_x = 20
    delta_y = 3
    image = utils.getimagepart(im,centralbin_x,centralbin_y,delta_x,delta_y)

    print 
    
