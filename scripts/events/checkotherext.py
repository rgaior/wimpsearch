import ROOT as R
import numpy as np
import matplotlib.pyplot as plt
import glob as glob
import argparse
import pandas as pd
import sys
import os
cwd = os.getcwd()
classpath = cwd + '/../../classes/'
utilspath = cwd + '/../../utils/'
sys.path.append(utilspath)
import utils
import constant

parser = argparse.ArgumentParser()
parser.add_argument("infile", type=str, nargs='?',default='', help="pickle file with the dataframe")
parser.add_argument("--cut", type=str, nargs='?', default="RUNID!=0",help="additional cut to be separated as in a dataframe query selection")

args = parser.parse_args()
filename = args.infile
cut = args.cut

df = pd.read_pickle(filename)
print 'additional cut : ' , cut
df = df.query(cut)

delta_x = 20
delta_y = 3
counter =0
for i, row in df.iterrows():
#    if counter > 3:
#        continue
    counter+=1
    filenames = glob.glob(constant.basefolderpostidm + "/rootc/" + "*" + str(int(row["RUNID"])) +"*" )
    
#    print filenames
#    print filenames.sort
    centerx = row['centerx']
    centery = row['centery']
    nrow = 2
    ncol = 4
    fig, axs = plt.subplots(nrow, ncol,figsize=(12,8), sharex=True)
    fig.suptitle("RUNID : "+ str(int(row["RUNID"])))
    for ax,filename in zip(fig.axes, filenames) :
        f = R.TFile(filename)
        im = f.Get("image")
        fext = filename[filename.rfind('_')+1:filename.find('.root')]
#        print 'fext = ' , fext
#        print row["EXTID"]
        centralbin_x = int(centerx)
        centralbin_y = int(centery)
        image = utils.getimagepart(im,centralbin_x,centralbin_y,delta_x,delta_y)
        fname = filename[filename.rfind('/'):]
        displayedimage = ax.imshow(image[2],origin='lower',extent=[np.min(image[0]),np.max(image[0])+1,np.min(image[1]),np.max(image[1])+1],vmin=np.min(image[2]),vmax=np.max(image[2]),aspect='auto')

        f.Close()
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=12)
        ax.tick_params(axis="z", labelsize=12)
        #ax.set_xticklabels([centralbin_x - delta_x/2, centralbin_x,centralbin_x +delta_x/2], fontsize=10)
        #ax.set_yticklabels([centralbin_y - delta_y, centralbin_y,centralbin_y +delta_y], fontsize=10)
#        ax.set_xticks(fontsize='5')
        #ax.set_ylabel('',fontsize='10')
        ax.text(centralbin_x,centralbin_y + 2, "EXT: " + str(fext) , style='italic',bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
        if (int(row["EXTID"]) == int(fext)):
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(3)
                ax.spines[axis].set_color("r")
#            print 'herrrrrrrrrrrrreee '
#            ax.set_ylabel(fontweight='bold')
#            ax.set_xlabel(fontweight='bold')
        fig.tight_layout()
        fig.subplots_adjust(top=0.936)
        fig.colorbar(displayedimage, ax=ax)

    folder = '/Users/gaior/DAMIC/data/wimp/eventanalysis/allext/'
    fig.savefig(folder + "allext_"+str(row["RUNID"]) + '_'+str(str(row["EXTID"])) +'.png' )
#plt.show()
#print image[2]


#plt.imshow(image[2],origin='lower',extent=[np.min(image[0]),np.max(image[0])+1,np.min(image[1]),np.max(image[1])+1],vmin=28,vmax=32,aspect='auto')
#plt.imshow(image[2],origin='lower',extent=[np.min(image[0]),np.max(image[0])+1,np.min(image[1]),np.max(image[1])+1],vmin=np.min(image[2]),vmax=np.max(image[2]),aspect='auto')
#plt.imshow(image[2],origin='lower',extent=[np.min(image[0]),np.max(image[0])+1,np.min(image[1]),np.max(image[1])+1],aspect='auto')

#for (j,i),label in np.ndenumerate(image[2]):
#    print label
#    axarr.text(centralbin_x + i - delta_x + 0.5, centralbin_y + j - delta_y + 0.5,int(label),ha='center',va='center')

#axarr.set_xlabel('X [pixel]')
#axarr.set_ylabel('Y [pixel]')
