from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
cwd = os.getcwd()
utilspath = cwd + '/../../utils/'
sys.path.append(utilspath)
from lmfit import minimize, Parameters
import constant
import utils
import pandas as pd
import seaborn as sns

def func(x,a,b):
    return a*x + b

def residual(params, x,y, erry):
    a = params['a']
    b = params['b']
    c = params['c']
    d = params['d']
    k = (d-b)/(a-c)
    model1 = a*x[x<k]+b
    model2 = c*x[x>=k]+d
    model = model1
    model =np.append(model, model2)
    return (y-model) / erry



# get the data from the pkl file:
limidbasse = 2000
#if dataset == 'all':
limid = 5000
#else:
#limid = 3704

#DClim = constant.dclimvalue
extcut = 0
ext = 4
DCfile = constant.datafolder + '/DC/DCfile.pkl'
cuts = "RUNID >" +str(limidbasse) + " & " + "RUNID < " +str(limid) + " & " +  constant.badimage+ " & " + constant.basecuts 
# +   + " & " + constant.radoncut
datafolder = constant.basefolderpostidm


#datafile = constant.datafolder + '/exposed/data1idm.pkl'

dfdata = pd.read_pickle(datafolder + 'datapostidm2018.pkl')
dfdata = utils.mergewithDC(DCfile,dfdata)
#dfdata = utils.dfbasic(dfdata)
dfdata = dfdata.query(cuts)

#datafile = constant.datafolder + '/exposed/data1.pkl'
data = dfdata
#pd.read_pickle(datafile)
print data.shape[0]
re = data.groupby(["RUNID","EXTID"])
a_DC = re.mean().DC
a_entries = re.DC.value_counts()


# obtain the profile
#bins = np.logspace(-2,2.5,30)
bins = np.logspace(-2,2.5,29)
ax = sns.regplot(x=a_DC, y=a_entries,x_bins=bins,x_estimator=np.mean,fit_reg=False, marker="o",scatter_kws={'alpha':1})
a_profx = np.array([])
a_proferry = np.array([])
a_profy = np.array([])
d = ax.collections[0]
d.set_offset_position('data')
data = d.get_offsets()

for l,d in zip(ax.lines,data):
    erry = (l.get_ydata()[1] - l.get_ydata()[0])/2
    a_profx = np.append(a_profx,d[0])
    a_profy = np.append(a_profy,d[1])
    a_proferry = np.append(a_proferry,erry)

x= a_profx
y= a_profy
erry= a_proferry
logx =np.log10(x[x>0])
logy =np.log10(y[x>0])
#error bar in log: dz = 0.434dy/y
errlogy = 0.434* (erry[x>0]/y[x>0])

# methodology
# we split in two parts: one with large DC and one at low DC
# we first fit each part to obtain initial values for a combined fit
# we fit the combined data with kinked lines

# large DC part
startpoint = 5
endpoint = 25
logxsel = logx[startpoint:endpoint]
logysel = logy[startpoint:endpoint]
errlogysel = errlogy[startpoint:endpoint]

# fit of the large DC part
xlin = np.linspace(0.5,2,10)
popt, pcov = curve_fit(func, logxsel, logysel,sigma=errlogysel)

## fit of the low DC part (~flat part)
startpoint2 = 0
endpoint2 = 4
logxsel2 = logx[startpoint2:endpoint2]
logysel2 = logy[startpoint2:endpoint2]
errlogysel2 = errlogy[startpoint2:endpoint2]

xflat = np.linspace(0.1,1.5,10)
popt2, pcov2 = curve_fit(func, logxsel2, logysel2,sigma=errlogysel2,p0=[0,1])

# fit of the combined data
params = Parameters()
params.add('a', value=0,min=-0.1, max=0.2,brute_step=0.001)
params.add('b', value=popt2[1],min=0, max=1)
params.add('c', value=1,min=0, max=3)
params.add('k', value=0.6,min=0.,max=1.5,vary=True,brute_step=0.001)
params.add('d', expr='k*(a-c)+b')

logxsel3 = logx[startpoint2:endpoint]
logysel3 = logy[startpoint2:endpoint]
errlogysel3 = errlogy[startpoint2:endpoint]

out = minimize(residual, params, args=(logxsel3, logysel3, errlogysel3),maxfev=1000)

print out.params.pretty_print()
v= out.params.valuesdict()
k = (v['d']- v['b'])/(v['a']-v['c'])
xl = np.linspace(logxsel3[0],k,10)
xr = np.linspace(k,logxsel3[-1],10)
fitteddatal = v['a']*xl + v['b']
fitteddatar = v['c']*xr + v['d']

# plotting part
plt.figure()
plt.plot(np.log10(a_DC),np.log10(a_entries),'.',c='orange',alpha=0.2,label='per image data')
plt.errorbar(logx,logy,yerr=errlogy,fmt='o',label='profile') # some point of leakage current are negative... this cause an warning only for the plotting
plt.plot(xl,fitteddatal,'-g',lw=2,label='fit')
plt.plot(xr,fitteddatar,'-r',lw=2,label='fit')
plt.xlabel('log10 (leakage current)')
plt.ylabel('log10 (cluster number)')
plt.xlim(-2.1,3)
plt.legend()
plt.show()
