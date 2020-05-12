import numpy as np
import ROOT as R
import pandas as pd
import glob
import constant
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
def readtree(filename, treename):
    """function readtree to convert a root tree into a pandas dataframe.
    there are small tweak in order to get rid of branches addes several times (that shouldn't be there anyway)
    the function is maybe not optimal in term of time, it takes around 4 minutes to convert a tree of 250000 entries. """
    f = R.TFile(filename)
    t = f.Get(treename)
    if treename not in constant.listofknowntrees:
        raise Exception('Unknown name of tree')
    listofbranch = t.GetListOfBranches()
    columns = []

    # to prevent a leaf that would be twice in the tree to appear twice in the columns
    for branch in listofbranch:
        if branch.GetName() in columns:
            continue
        columns.append(branch.GetName())
    entries = t.GetEntries()
    alldata = np.zeros(shape=(entries,len(columns) ) )
    df = pd.DataFrame(columns=columns)
    counter = 0
    for event in t:
        if (counter % 1000 == 0):
            print('counter = ',  counter)
        eventdata = event.GetListOfLeaves()
        a_data = np.array([])
        filleddata = []
        for data in eventdata:
            if data.GetName() in filleddata:
                continue
            filleddata.append(data.GetName())
            a_data = np.append(a_data,data.GetValue())
        alldata[counter] = a_data
        counter+=1
    df = pd.DataFrame(alldata,columns=columns)
    return df



def getimageplot(x,y,xbinning=100):
    xlow = 4400
    xhigh = 8250
    ylow = 1
    yhigh = 42
#    xbinning = 100
    ybinning = 1
    xedges = np.arange(xlow,xhigh,xbinning)
    yedges = np.arange(ylow,yhigh,ybinning)

    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    H = H.T
    fig, ax2dhist = plt.subplots(figsize=(10, 8))
    max = H.max()
    the2dhist= ax2dhist.imshow(H,  cmap='Blues',vmin=0, vmax=max,interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto')
    # create new axes on the right and on the top of the current axes
    # The first argument of the new_vertical(new_horizontal) method is
    # the height (width) of the axes to be created in inches.
    divider = make_axes_locatable(ax2dhist)
    axHistx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax2dhist)
    axHisty = divider.append_axes("right", 1.2, pad=0.1, sharey=ax2dhist)

    # make some labels invisible
    axHistx.xaxis.set_tick_params(labelbottom=False)
    axHisty.yaxis.set_tick_params(labelleft=False)

    # now determine nice limits by hand:
    binwidth = 1

    xbins = np.arange(xlow,xhigh,xbinning)
    ybins = np.arange(ylow,yhigh,ybinning)
    axHistx.hist(x, bins=xbins,histtype='step',lw=1,color='k')
    axHisty.hist(y, bins=ybins, histtype='step',lw=1,color='k', orientation='horizontal')


    ticks=np.linspace(0,max,max+1)
    fig.colorbar(the2dhist, ax=ax2dhist,ticks=ticks)
    ax2dhist.set_xlabel('X [re-binned per ' +str(xbinning) + ' pixels]' )
    ax2dhist.set_ylabel('Y [pixel]')

    return fig

def getimagepart(image,centralbin_x,centralbin_y,delta_x,delta_y):
    array_x = range(centralbin_x - delta_x, centralbin_x + delta_x + 1)
    array_y = range(centralbin_y - delta_y, centralbin_y + delta_y + 1)
#    print array_y
    imtoshow = np.ndarray(shape=(len(array_x),len(array_y)))
    count = 0
    for x in array_x:
        imarray = np.array([])
        for y in array_y:
            imarray = np.append(imarray,image.GetBinContent(x + 1 ,y + 1))
        imtoshow[count] = imarray
        count+=1
    return [array_x,array_y,imtoshow.T]

def ADUtoelec(adu):
    efact = 2.6e-4
    return adu*efact*1000/3.77
def getimagepart1d(image,centralbin_x,centralbin_y,delta_x):
    array_x = range(centralbin_x - delta_x, centralbin_x + delta_x + 1)
    count = 0
    imarray = np.array([])
    for x in array_x:
        imarray = np.append(imarray, ADUtoelec(image.GetBinContent(int(x)  ,int(centralbin_y +1) )) )

    return [array_x,imarray]

def getimage(image):
    array_x = range(0,4200)
    array_y = range(1,43)
    imtoshow = np.ndarray(shape=(len(array_x),len(array_y)))
    count = 0
    for x in array_x:
        imarray = np.array([])
        for y in array_y:
            imarray = np.append(imarray,image.GetBinContent(x + 1 ,y + 1))
        imtoshow[count] = imarray
        count+=1
    return [array_x,array_y,imtoshow.T]




def getlistofimage(path):
    listoffile = glob.glob(path+ '*_11.root')
    listofim = []
    for f in listoffile:
        im = f[ f[:f.rfind('_')].rfind('_')+1:f.rfind('_')]
        print (listofim.append(int(im)))
    return listofim


def mergeruns(listofrun,folder,fname,dfall):
    for r in listofrun:
        folder2 = folder + constant.runname[r] + '/pkl/'
        print (folder2)
        df = pd.read_pickle(folder2 + fname + '.pkl')
        dfall = dfall.append(df)
    return dfall


def initialize_dataframe(path):
    dfex = pd.read_pickle(path)
    return pd.DataFrame(columns=dfex.columns)





def writebrowsercommand(df, outname, run, iteration):
    fout = open(outname,'w')
#    exptime = '30000' if ('30' in run) else '100000'
    for index, row in df.iterrows():
        exptime = str(int(getexpofromrunid(row['RUNID'])[1]))
        if iteration <=4:
            path = constant.basefolders[iteration] + constant.runname[run] + 'rootc/d44_snolab_Int-800_Exp-' +  exptime + '_' + str(int(row['RUNID'])) + '_' + str(int(row['EXTID'])) + '.root'
        if iteration ==5:
            path = constant.basefolders[iteration] + '/rootc/d44_snolab_Int-800_Exp-' +  exptime + '_' + str(int(row['RUNID'])) + '_' + str(int(row['EXTID'])) + '.root'

#EventBrowser /Users/gaior/DAMIC/data/official4/cryoOFF_100000s-IntW800_OS_1x100_run1/rootc/d44_snolab_Int-800_Exp-100000_2475_6.root
        fout.write('EventBrowser' + ' -c ' + str(int(row['cid'])) + ' ' + path + '\n')


def produceimagecut(dfdc,dclimvalue):
    imcuts = []
    count = 0
    totcount = 0
    for index, row in dfdc.iterrows():
        totcount+=1
        if row.ADUDC > dclimvalue:
            if count ==0:
                imcut = ''
                imcut += ' ( RUNID != '+ str(row.image) + ' | EXTID != ' +str(row.ext) + ') '
            else:
                imcut += ' & ( RUNID != '+ str(row.image) + ' | EXTID != ' +str(row.ext) + ') '
            count +=1
            if count==20:
                imcuts.append(imcut)
                count=0
            if totcount == dfdc.shape[0]:
                imcuts.append(imcut)

    return imcuts

def isrejectedbyDCcut(runid,extid,dfdc,dclimvalue):
    imcuts = []
    count = 0
    totcount = 0
    DC = float(dfdc[(dfdc.image == runid) & (dfdc.ext == extid)].DC)
    if DC > dclimvalue:
        return True
    else:
        return False

def DCimagecut(dfdc,dclimvalue):
    imcuts = np.unique(dfdc[dfdc.DC > dclimvalue])
    return imcuts

def produceimagecut_ryan(dfdc,dclimvalue):
    imcuts = []
    count = 0
    totcount = 0
    for index, row in dfdc.iterrows():
        totcount+=1
        if row.DC > dclimvalue:
            if count ==0:
                imcut = ''
                imcut += ' ( RUNID != '+ str(row.image) + ' | EXTID != ' +str(row.ext) + ') '
            else:
                imcut += ' & ( RUNID != '+ str(row.image) + ' | EXTID != ' +str(row.ext) + ') '
            count +=1
            if count==20:
                imcuts.append(imcut)
                count=0
            if totcount == dfdc.shape[0]:
                imcuts.append(imcut)

    return imcuts

def writeimagecut_ryan(dfdc,dclimvalue):
    strdcvalue = str(dclimvalue).replace(',','_')
    filename = '/Users/gaior/DAMIC/code/data_analysis/out/cuts/imcutsDC'+strdcvalue+ '.txt'
    f = open(filename,'w+')
    imcuts = []
    count = 0
    totcount = 0
    for index, row in dfdc.iterrows():
        totcount+=1
        if row.DC > dclimvalue:
            if count ==0:
                imcut = ''
                imcut += ' ( RUNID != '+ str(row.image) + ' | EXTID != ' +str(row.ext) + ') '
                f.write('RUNID EXTID \n')
                f.write(str(int(row.image)) + ' ' + str(int(row.ext)) + '\n' )
            else:
                imcut += ' & ( RUNID != '+ str(row.image) + ' | EXTID != ' +str(row.ext) + ') '
                f.write(str(int(row.image)) + ' ' + str(int(row.ext)) + '\n')
            count +=1
            if count==20:
                imcuts.append(imcut)
            if totcount == dfdc.shape[0]:
                imcuts.append(imcut)

#    return imcuts

def getremovedexpo_ryan(dfdc,dclimvalue):
    imcuts = []
    count = 0
    totcount = 0
    removedexpo = 0
    totexpo = 0
    imageremoved = 0
    for index, row in dfdc.iterrows():
        if row.DC > dclimvalue:
            runexpo = getexpofromrunid(row.image)
            removedexpo+=runexpo[1]
            imageremoved+=1
#    print 'number of removed images from DC cut: ', imageremoved
    return removedexpo

def getremovedimage_ryan(dfdc,dclimvalue):
    imcuts = []
    count = 0
    totcount = 0
    removedexpo = 0
    totexpo = 0
    imageremoved = 0
    for index, row in dfdc.iterrows():
        if row.DC > dclimvalue:
            runexpo = getexpofromrunid(row.image)
            removedexpo+=runexpo[1]
            imageremoved+=1
    return imageremoved

def getremovedimageperext_ryan(dfdc,dclimvalue,ext):
    imcuts = []
    count = 0
    totcount = 0
    removedexpo = 0
    totexpo = 0
    imageremoved = 0
    for index, row in dfdc.iterrows():
        if row.ext!=ext:
            continue
        if row.DC > dclimvalue:
            runexpo = getexpofromrunid(row.image)
            removedexpo+=runexpo[1]
            imageremoved+=1
    return imageremoved

def gettotexpo(dfdc):
    imcuts = []
    count = 0
    totcount = 0
    totexpo = 0
    for index, row in dfdc.iterrows():
        runexpo = getexpofromrunid(row.image)
        totexpo+=runexpo[1]
    return totexpo


def getexpofromfilename(f):
    f = f[f.rfind('/'):]
    f = f[f.rfind('Exp-')+4:]
    exp = f[:f.find('_')]
    return int(exp)

def getexpofromrunid(runid):
    run = ''
    if ( (runid >= 2473) & (runid <= 2484) ):
        run = 'run100ks1'
    if ( (runid >= 2559)  & (runid <= 2619)):
        run = 'run100ks2'
    if ( (runid >= 2829) & (runid <= 2839)):
        run = 'run100ks3'
    if ( (runid >= 2623) & (runid <= 2637)):
        run = 'run30ks1'
    if ( (runid >= 2843) & (runid <= 2951)):
        run = 'run30ks2'
    if ( (runid >= 3003) & (runid <= 3162)):
        run = 'run30ks3'
    if ( (runid >= 3203) & (runid <= 3487)):
        run = 'run30ks4'
    if ( (runid >= 3536) & (runid <= 3703)):
        run = 'run30ks5'
#    print 'runid = ' , runid, 'run = ', run
    expo = constant.runinfo[run][0]
#    print 'exop = ',  expo
    return [run,expo]


def getrunexposure(runname,extnr):
    expo = constant.runinfo[runname][0]
    totnrofruns = constant.runinfo[runname][1]
    nrofbadruns = len(constant.badruns[runname])

    runexpo = (totnrofruns - nrofbadruns)*extnr*expo
    return runexpo

def getexposurefromrunidlist(runidlist):
    totexpo = 0
    for r in runidlist:
        [runname, expo] = getexpofromrunid(r)
        totexpo += expo
    return totexpo


def removedexpofromDC(runname,dfdc, dclimvalue):
    removedexpo = 0
    for index, row in dfdc.iterrows():
        if row.runname != runname:
            continue
        if row.ADUDC > dclimvalue:
            removedexpo += row.exposuretime
    print (' runname = ',  runname)
    print(' exporempove ', removedexpo)
    return removedexpo

def removedexpoallfromDCryan(dfdc, dclimvalue,runidlist,ext):
    removedexpo = 0
    for index, row in dfdc.iterrows():
        if row.ext != ext:
            continue
        if row.DC > dclimvalue:
            print ("rowDC = ", row.DC , ' ' , row.image , ' ext = ' , row.ext)
            if row.image in runidlist:
                removedexpo += getexpofromrunid(int(row.image))[1]
    print (' exporempove ', removedexpo)
    return removedexpo

def removedexpofromDCryan(runname,dfdc, dclimvalue):
    removedexpo = 0
    for index, row in dfdc.iterrows():
        if row.runname != runname:
            continue
        if row.DC > dclimvalue:
            removedexpo += row.exposuretime
    print (' runname = ' , runname , ' exporempove ', removedexpo)
    return removedexpo

def removedexpofromDCperext(runname,dfdc, dclimvalue):
    removedexpo = 0
    removedexpoperext = {1:0, 2:0, 3:0, 4:0, 6:0, 11:0, 12:0}
    for index, row in dfdc.iterrows():
        if row.runname != runname:
            continue
        if row.ADUDC > dclimvalue:
            removedexpoperext[row.ext] += row.exposuretime
    print (' runname = ' , runname , ' exporempove ', removedexpoperext)
    return removedexpoperext




#return the error of the
def gethisterror(values, weights, bins):
    a_error = np.array([])
    binsindex = np.digitize(values,bins)

    # access elements for first bin
    for index in range(1,len(bins)):
        bin_ws = weights[np.where(binsindex==index)[0]]
    # error of fist bin
        error = np.sqrt(np.sum(bin_ws**2.))
#        print error
        a_error =np.append(a_error,error)

    return a_error


# return the position of the nrcluster entries from the produced image for display (i.e. an array).
def findpositionfromimage(H, clusternr, xlow, ylow,xbinning, ybinning):
    pos = np.where(H == clusternr)
    newpos = np.ndarray(shape=(len(pos[0]),2))
    for x,y,i in zip(pos[1], pos[0],range(len(pos[0]))):
        newpos[i] = np.array([x*xbinning + xlow, y*ybinning + ylow])
    return newpos



def findclusteratthatposition(df,x,y,delx,dely,runid,extid):
    dfsel = df.query('RUNID == ' + str(int(runid)))
    dfsel = dfsel.query('EXTID != ' + str(int(extid)))
    dfsel = dfsel.query('(centerx - ' +str(x) +' <= '+ str(float(delx)/2) + ') &  ' +  ' ( ' + str(x) + ' -centerx ' +' <= '+ str(float(delx)/2) + ')' )
    dfsel = dfsel.query('(centery - ' +str(y) +' <= '+ str(float(dely)/2) + ') &  ' +  ' ( ' + str(y) + ' -centery ' +' <= '+ str(float(dely)/2) + ')' )


#    print 'np.abs((df.centery - y) = ' , np.abs((df.centery - y))
#    print ' (np.abs(df.meanx - x) = ' ,  np.abs(df.meanx - x)

#    dfnew = df[ (np.abs(df.meanx - x) < delx ) & (np.abs(df.centery - y) < dely )]

    return dfsel

def findclusteratthatposition2(df,x,y,delx,dely,runid,extid,cid):
    dfsel = df.query('RUNID == ' + str(int(runid)))
    dfsel = dfsel.query('EXTID != ' + str(int(extid)))
    dfsel = dfsel.query('(centerx - ' +str(x) +' <= '+ str(float(delx)/2) + ') &  ' +  ' ( ' + str(x) + ' -centerx ' +' <= '+ str(float(delx)/2) + ')' )
    dfsel = dfsel.query('(centery - ' +str(y) +' <= '+ str(float(dely)/2) + ') &  ' +  ' ( ' + str(y) + ' -centery ' +' <= '+ str(float(dely)/2) + ')' )
    size = dfsel.shape[0]
    commondf = pd.DataFrame(columns=constant.commondfcol)
    for ext, c in zip(dfsel.EXTID.values, dfsel.cid.values):
        commondf = commondf.append(pd.DataFrame([[runid, extid, ext, cid, c]], columns=constant.commondfcol))
    return commondf

def findclusteratthatposition3(dfHEC,dfLEC,offsetx,offsety,windowx,windowy):
    runid = dfHEC.RUNID
    extid = dfHEC.EXTID
    cid = dfHEC.cluster_id
    HEC_energy = dfHEC['charge_total']/constant.energyconv
    dfLECsel = dfLEC.query('RUNID == ' + str(int(runid)))
    dfLECsel = dfLECsel.query('EXTID != ' + str(int(extid)))
    x = dfHEC.center_x
    y = dfHEC.center_y
    offsetx = x + offsetx + np.sign(offsetx)*float(windowx)/2
    offsety = y + offsety + np.sign(offsety)*float(windowy)/2
#    print 'x= ', x, 'offsetx = ' , offsetx, 'offsety = ' , offsety
    dfLECsel = dfLECsel.query('(centerx - ' +str(offsetx) +' <= '+ str(float(windowx)/2) + ') &  ' +  ' ( centerx - ' + str(offsetx) +' >= -'+ str(float(windowx)/2) + ')' )
    dfLECsel = dfLECsel.query('(centery - ' +str(offsety) +' <= '+ str(float(windowy)/2) + ') &  ' +  ' ( centery - ' + str(offsety) +' >= -'+ str(float(windowy)/2) + ')' )
    size = dfLECsel.shape[0]
    commondf = pd.DataFrame(columns=constant.commondfcol)
    if size == 0:
       commondf = commondf.append(pd.DataFrame([[runid, extid, cid, HEC_energy,0,0, 0,0,0,0,0,0,0 ]], columns=constant.commondfcol))
    else:
#        print ' x + np.sign(delx)*float(windowx)/2 = ', offsetx ,' offsety = ' , offsety ,
        for index, row in dfLECsel.iterrows():
            print (' centerx = ' , row['centerx'], ' offsetx = ' , offsetx)
            print (' centery = ' , row['centery'], ' offsety = ' , offsety)
            commondf = commondf.append(pd.DataFrame([[runid, extid, cid, HEC_energy, x,y,row['EXTID'], row['cid'],row['ene1'],row['centerx'],row['centery'],row['ll'],row['llc']]], columns=constant.commondfcol))
    return commondf



def th2_to_array(im,xlow,xup,ylow,yup):
    out = np.ndarray( shape=(xup,yup-ylow) )
    for i in range(xlow,xup):
        for j in range(ylow,yup):
            out[i,j] = im.GetBinContent(i,j)
    return out



def readpds(name):
    f = R.TFile(name)
    fname = name[name.rfind('/')+1:name.rfind('.pds')]
    print (' !!! fname == ' , fname)
    t = f.Get("deposits")
    columns = ['file','dep_x','dep_y','dep_z','dep_e']
    df = pd.DataFrame(columns=columns)
    for event in t:
        file =  fname
        dep_x =  event.dep_x
        dep_y =  event.dep_y
        dep_z =  event.dep_z
        dep_e =  event.dep_e

        dftemp = pd.DataFrame([[file,dep_x,dep_y,dep_z,dep_e]], columns=columns)
        df = df.append(dftemp,ignore_index=True)
    return df


def eff_errors(k, N, method):
    k = k.astype(float)
    N = N.astype(float)
    if method == 'binomial':
#        return (1/N)*np.sqrt(k*(1-(k/N) ))
        return (1/N)*np.sqrt(k*(1-(k/N) ))
    if method == 'poisson':
        return np.sqrt( (k*(N+k))/N**3 )
    if method == 'bayes':
        return np.sqrt( ((k+1)*(k+2)) / ((N+2)*(N+3)) - (k+1)**2/(N+2)**2 )




def getdllm2(dc):
    a_dc =constant.a_dc
    a_dllcut = constant.a_dllcut
    if dc < 100:
        dll = np.interp(dc,a_dc,a_dllcut)
    else:
        z = np.polyfit(constant.a_dc,constant.a_dllcut,1)
        p = np.poly1d(z)
        dll = p(dc)
    return dll

def putdll(df):
#    df['dll'] = df.ll - df.llcd
    df = df.assign(dll=pd.Series( df.ll - df.llc, index=df.index) )
    return df
# def ecal (row):
#     if row['EXTID'] == 1 :
#         return 1.009*row['ene1']
#     if row['EXTID'] == 2 :
#         return 0.956*row['ene1']
#     if row['EXTID'] == 3 :
#         return 0.979*row['ene1']
#     if row['EXTID'] == 4 :
#         return 0.984*row['ene1']
#     if row['EXTID'] == 6 :
#         return 1.010*row['ene1']
#     if row['EXTID'] == 11 :
#         return 0.969*row['ene1']
#     if row['EXTID'] == 12 :
#         return 0.999*row['ene1']
def ecal (row):
    if row['EXTID'] == 1 :
        return constant.ecalconstant[1]*row['ene1']
    if row['EXTID'] == 2 :
        return constant.ecalconstant[2]*row['ene1']
    if row['EXTID'] == 3 :
        return constant.ecalconstant[3]*row['ene1']
    if row['EXTID'] == 4 :
        return constant.ecalconstant[4]*row['ene1']
    if row['EXTID'] == 6 :
        return constant.ecalconstant[6]*row['ene1']
    if row['EXTID'] == 11 :
        return constant.ecalconstant[11]*row['ene1']
    if row['EXTID'] == 12 :
        return constant.ecalconstant[12]*row['ene1']


def dfbasic(df):
    df = putdll(df)
    df = df.assign(ene1=df.apply(lambda row: ecal(row), axis=1))
    return df

def dfbasicene(df):
    df['ene'] = df.apply(lambda row: ecal(row), axis=1)
    return df


def geteff(dfrec, dfsim, bins, cutrec, cutsim):
    dfsel = dfrec.query(cutrec)
    dfsel = dfsel.reset_index()

    dfsel['ebinned'] = pd.cut(dfsel['ene1'],bins)
    ebinned = dfsel.groupby('ebinned')

    dfselsim = dfsim.query(cutsim)
    dfselsim = dfselsim.reset_index()

    dfselsim['ebinned'] = pd.cut(dfselsim['sime'],bins)
    ebinnedsim = dfselsim.groupby('ebinned')

    recnr = ebinned.count().ene1
    simnr = ebinnedsim.count().sime

    eff = recnr/simnr
#    err_eff = utils_noroot.eff_errors(recnr, simnr, 'bayes')
    err_eff = eff_errors(recnr, simnr, 'poisson')
    cbins = (bins[1:] + bins[:-1]) /2
    return [cbins,eff,err_eff]

def geteffsim(dfrec, dfsim, bins, cutrec, cutsim):
    dfsel = dfrec.query(cutrec + "& sime >0")
    dfsel = dfsel.reset_index()

    dfsel['ebinned'] = pd.cut(dfsel['sime'],bins)
    ebinned = dfsel.groupby('ebinned')

    dfselsim = dfsim.query(cutsim)
    dfselsim = dfselsim.reset_index()

    dfselsim['ebinned'] = pd.cut(dfselsim['sime'],bins)
    ebinnedsim = dfselsim.groupby('ebinned')

    recnr = ebinned.count().ene1
    simnr = ebinnedsim.count().sime

    eff = recnr/simnr
    err_eff = eff_errors(recnr, simnr, 'poisson')
    cbins = (bins[1:] + bins[:-1]) /2
    return [cbins,eff,err_eff]


def mergewithDC(dcfilepath, df):
    dfDC = pd.read_pickle(dcfilepath)
    dftemp = dfDC
    dftemp = dftemp.merge(df,left_on=['image','ext'], right_on=['RUNID','EXTID'], how='outer')
    dftemp = dftemp.dropna()
    return dftemp

def getdatatofit(df,bins):
    n,b = np.histogram(df.dll,bins)
    logn = np.array([])
    errlogn = np.array([])
    bins = np.array([])
    for nn,bb in zip(n,b):
        if nn != 0:
            logn = np.append(logn,np.log(nn))
            errlogn = np.append(errlogn,1/np.sqrt(nn))
            bins = np.append(bins,bb)
    return [bins,logn,errlogn]
