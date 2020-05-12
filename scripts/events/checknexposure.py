import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
alexdatafolder  = "/Users/gaior/DAMIC/code/wimpsearch/data/Alex/"
free = pd.read_pickle(alexdatafolder + 'free_fit_llvals.pkl')
null = pd.read_pickle(alexdatafolder + 'null_fit_llvals.pkl')
free = free.sort_values(by=['time'])
null = null.sort_values(by=['time'])
all = free
all['delta'] = free.logp - null.logp
#print all


runidvsinexpo = np.load(alexdatafolder + "idvsintexpo.npz")

eventfile = pd.read_pickle(alexdatafolder + "selections.pkl")

#null = alexdata.sort_values(by=['time'])

deltalog = [1, -0.3]
cols = ['b', 'r']
 

expomax= 0
#for delta, col in zip(deltalog,cols):
cuts = [" delta < 1 ", " delta < -0.3 ","ene < 0.5","ene < 0.2","runid >0"]
for cut in cuts:
    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0.1,'height_ratios': [3, 1]})
    a_nrofev = np.array([])
    a_nrofevperid = np.array([])
    a_nrofev_perperiod = np.array([])
    a_expo = np.array([])
    a_runid = np.array([])
    nrofev = 0
#    allsel = all.query('delta < ' +str(delta) ) 
    allsel = all.query(cut) 
    print (allsel)
    runids = np.unique(allsel.runid)
    for runid in runids:
        nrofev = allsel.query('runid <= '+str(runid)).shape[0]
        nrofevperid = allsel.query('runid == '+str(runid)).shape[0]
        expo = runidvsinexpo['int_expo'][runidvsinexpo['runid']==runid]
        a_nrofev = np.append(a_nrofev,nrofev)
        a_nrofevperid = np.append(a_nrofevperid,nrofevperid)
        a_expo   = np.append(a_expo,expo)
        a_runid  = np.append(a_runid,runid)
        a_nrofev_perperiod = np.append(a_nrofev_perperiod, nrofev)
        if expo > expomax:
            expomax = expo

    z = np.polyfit(a_expo, a_nrofev, 1)
# #plt.plot(a_runid,a_nrofev,'o')
    p = np.poly1d(z)
    axs[0].plot(a_expo,a_nrofev,'.',label=cut)
    axs[0].plot(a_expo,p(a_expo),'-')
    axs[0].legend()
    axs[1].plot(a_expo,(a_nrofev-p(a_expo))/np.max(a_nrofev),'.')
    axs[1].set_xlabel('exposure [s]')
    axs[0].set_ylabel('nr of clusters')
    axs[1].set_ylabel('residuals')
#    axs[1].set_ylim(-0.2,0.2)
    print (a_nrofev)
    
    expo_axis = np.linspace(0,expomax,30)
    a_evs = np.array([])
    a_evs_err = np.array([])
    a_exps = np.array([])
    a_runids = np.array([])
    for exp1, exp2 in zip(expo_axis[:-1] ,expo_axis[1:]):
        evtosum = a_nrofev_perperiod[(a_expo>=exp1) & (a_expo<exp2)]
        events = a_nrofevperid[(a_expo>=exp1) & (a_expo<exp2)]
        print ('events = ', events)
        print ('evtosum = ' , evtosum)
        if len(events) == 0:
            a_evs = np.append(a_evs,0)
            a_evs_err = np.append(a_evs_err,0)
        else:
            a_evs = np.append(a_evs,np.sum(events))
            a_evs_err = np.append(a_evs_err,np.sqrt(np.sum(events)))
        print ('a_evs = ', a_evs)
        a_exps = np.append(a_exps,(exp1 + exp2) / 2)
#        exp = a_expo[(a_expo>=exp1) & (a_expo<exp2)]
#        a_nrofev_perperiod = np.append(a_nrofev_perperiod,np.sum()
#        print 'evtosum = ', evtosum
#        print  ' exp ' , exp
        
    tickarray = a_expo/expomax
    new_tick_locations = tickarray
#    plt.figure()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.set_xlim(0, expomax) 
    ax1.errorbar(a_exps, a_evs, yerr=a_evs_err,fmt='o',label=cut)
    ax1.set_ylabel('nr of events')
    ax1.set_xlabel('exposure [s]')
    ax2 = ax1.twiny()  
    divisionfactor = int(len(a_expo)/10)
#    a_exposhowed = a_expo[::divisionfactor]  
    a_runs = runids.astype(int)
    a_runidshowed = a_runs[::divisionfactor]  
#    print (len(a_runidshowed), ' -kasdjfn ' , len(a_exposhowed))
#    ax2.set_xticks(a_exposhowed)
    ax1.legend()
    

plt.show()
