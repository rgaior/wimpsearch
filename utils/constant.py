####################
## root tree data ##
####################
listofknowntrees = ['clusters','simclusters','clusters_tree']
datacolumns = ['RUNID','EXTID','EXPTIME','EXPSTART','efact','nvalidpix','cid','centerx','centery','linlength','is_masked','qguess','sguess','oguessg','oguessc','llg','llc','success','ll','meanx','meanx_err','sigma','sigma_err','efit','qbase','npix','npix4','ene1','ene_integ','chi2g','chi2c','touchmask','sime','simz','simx','simy','simn','simdistx','simdisty','multirows','type','status','ll_enlarg','llc_enlarg','ll_14','llc_14','gchi2','cchi2','qmax','qdelta_dx','qdelta_sx','is_premasked','exposuretime']

#######################################
## constant for the crosstalk codes ###
#######################################
commondfcol = ['RUNID','EXTID_HE','CID_HE','E_HE','X_HE','Y_HE','EXTID_LE','CID_LE','E_LE','X_LE','Y_LE','ll','llc']
energyconv = 3.7 #ev per electron
windowxsearch = 20
windowysearch = 1


#####################
### CCD constant ####
#####################

extensionlist = [1,2,3,4,6,11,12]
ccd_xmin = 4272
ccd_xmax = 8388
ccd_ymin = 1
ccd_ymax = 43
ecalconstant = {1:1.009,2:0.956,3:0.979,4:0.984,6:1.010,11:0.969,12:0.999}

####################
## folders    ######
####################
basefolder = '/Users/gaior/DAMIC/code/wimpsearch/'
datafolder = basefolder + '/data/'

basefolderidm = '/Users/gaior/DAMIC/data/idm2018/'
basefolderpostidm = '/Users/gaior/DAMIC/data/postidm2018/'
outfolder = '/Users/gaior/DAMIC/code/data_analysis/out/'
DCfolder = '/Users/gaior/DAMIC/data/DC/'


#############################
## cuts definition ##########
#############################
#positioncut = 'centery < 42 & centery > 1 & multirows == 0'
#recpositioncut = 'centery < 42 & centery > 1 & (simdistx == 0 | (simdistx > 5) & (simdisty > 2) )'
#simpositioncut = 'simy < 42 & simy > 1 & ((simdistx > 5) & (simdisty > 2) )'
positioncut = 'centery < 42 & centery > 1 & (centerx < 8250 & centerx > 4400) & multirows == 0'
recpositioncut = 'centery < 42 & centery > 1 & (centerx < 8250 & centerx > 4400) & (simdistx == 0 | (simdistx > 5) & (simdisty > 2) )'
simpositioncut = 'simy < 42 & simy > 1 & (simx < 8250 & simx > 4400) & ((simdistx > 5) & (simdisty > 2) )'
#positioncut = 'centery < 42 & centery > 2 & multirows == 0'
maskcut = 'is_masked == 0 & touchmask == 0 & success ==1'
llcut = 'll_14 < 90'
qmaxcut = 'qmax/(ene1*1000./3.77) > 0.2'
basecuts = maskcut + ' & ' +  llcut+ ' & ' +  qmaxcut

radoncut = " (RUNID<2564 | RUNID>2566) &  (RUNID< 2902 | RUNID> 2903) & (RUNID<3267 | RUNID>3336) & (RUNID<3353 | RUNID>3419) & (RUNID<3654 | RUNID> 3657) & (RUNID<3764 | RUNID>3767) & (RUNID<3826 | RUNID>3853) & (RUNID<3868 | RUNID>3874) & (RUNID<3913 | RUNID>3921) & (RUNID<4003 | RUNID > 4007) & (RUNID<4207 | RUNID > 4212)"

#dclimvalue = 7.16
dclimvalue = 6.76
#dclimvalue = 6.51
dllcutvalue = -26

a_dc = [0.86, 3.16227766, 10., 19.95262315, 39.81071706, 100.]
a_dllcut = [-24.21690438, -24.80158061, -32.28029745, -38.67258441, -55.36774714, -87.56080688]

listofexpo = '/Users/gaior/DAMIC/data/postidm2018/listofexpo.npy'
listofrun = '/Users/gaior/DAMIC/data/postidm2018/listofrun.npy'


badimage = 'RUNID!=2473 & RUNID!=2479 & RUNID!=2482 & RUNID!=2559 & RUNID!=2577 & RUNID!=2611 & RUNID!=2623  & RUNID!=2829 & RUNID!=2843 & RUNID!=2849 & RUNID!=2853 &  RUNID!=2902 & RUNID!=2927 & RUNID !=3003 & RUNID!=3011 & RUNID!=3018 & RUNID!=3020 & RUNID!=3059 & RUNID!=3112 & RUNID!=3203 & RUNID!=3250 & RUNID!=3332 & RUNID!=3345 & RUNID!=3417 & RUNID!=3453 & RUNID!=3473 & RUNID!=3483 & RUNID!=3536 & RUNID!=3545 & RUNID!=3584 & RUNID!=3634 & RUNID!=3636 & RUNID!=3637 & RUNID!=3638 & RUNID!=3639 & RUNID!=3654 & RUNID!=3655 & RUNID!=3656 & RUNID!=3685  & RUNID!=3698 & RUNID!=3707 & RUNID!=3751 & RUNID!=3807 & RUNID!=3852 & RUNID!=3905 & RUNID!=3931 & RUNID!=3958 & RUNID!=4011 & RUNID!=4044 & RUNID!=4074 & RUNID!=4083 & RUNID!=4127'


negativepixelimage = '(RUNID!=3024 | EXTID!=11) &(RUNID!=3029 | EXTID!=11) & (RUNID!=3029 | EXTID!=12)  & (RUNID!=3125 | EXTID!=11) & (RUNID!=3125 | EXTID!=12) &(RUNID!=3126 | EXTID!=11) & (RUNID!=3126 | EXTID!=12) & (RUNID!=3537 | EXTID!=2) & (RUNID!=3537 | EXTID!=12)  & (RUNID!=3538 | EXTID!=2) & (RUNID!=3538 | EXTID!=12)  & (RUNID!=3539 | EXTID!=2) & (RUNID!=3539 | EXTID!=12)  & (RUNID!=3540 | EXTID!=2) &(RUNID!=3540 | EXTID!=12) &(RUNID!=3541 | EXTID!=2) & (RUNID!=3541 | EXTID!=12) &  (RUNID!=3542 | EXTID!=2) & (RUNID!=3542 | EXTID!=12) & (RUNID!=3543 | EXTID!=2) & (RUNID!=3543 | EXTID!=12) & (RUNID!=3544 | EXTID!=2) & (RUNID!=3546 | EXTID!=12) & (RUNID!=3548 | EXTID!=12) & (RUNID!=3598 | EXTID!=12) &(RUNID!=3657 | EXTID!=4) & (RUNID!=3657| EXTID!=6) & (RUNID!=3657 | EXTID!=11) & (RUNID!=3657 | EXTID!=12)'



# runid = {'run100ks1':[2473,2474,2475,2479,2480,2481,2482,2483,2484],'run100ks2':[2559,2560,2561,2562,2563,2564,2565,2566,2567,2568,2569,2570,2571,2572,2573,2577,2611,2612,2613,2614,2615,2616,2617,2618,2619],'run30ks1':[2623,2624,2625,2626,2627,2628,2629,2630,2631,2632,2633,2634,2635,2636,2637]}
# runidind = {2473:1,2474:2,2475:3,2479:4,2480:5,2481:6,2482:7,2483:8,2484:9,2559:10,2560:11,2561:12,2562:13,2563:14,2564:15,2565:16,2566:17,2567:18,2568:19,2569:20,2570:21,2571:22,2572:23,2573:24,2577:25,2611:26,2612:27,2613:28,2614:29,2615:30,2616:31,2617:32,2618:33,2619:34}
