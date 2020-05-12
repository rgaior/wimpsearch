Data analysis with the python code:
1. Produce pkl files:
   we need first to convert root file to pandas dataframe. This is done with the script "writepkl.py"
   this code allows the user to convert the files from PFS to pandas dataframe, it can convert data or simulation clusters trees.
   During this conversion, an updated energy calibration is applied and the variable dll in inserted with the function "dfbasic"
   
2. apply basecut and insert the DC information with the script producedataset.py

3. Produce the Global DLL cut in /script/dLLcut/
   - first define the DC cut with findDCcut.py
   - find the DC cut required to find the condition of the previous paper (i.e. dLL distribution overlap)
   - the check of the exposure as a function of the DC cut is found in the script: expovsDCcut.py
   - The fit the data dLL distribution with fitdLL.py




Efficiency:
dataset: For the computation of the efficiency we simulate clusters with PointDepSim.C on blanks.
The dataset of blank is composed of the RUNIDs ending with "1". 
We produced two types of datasets: 
   - one at low energy 0.03keV to 1keV with an exponential energy distribution to have a more events a low energy 
   - one at larger energies from 0.7 to 9 keV

Data are located at:
root files:
/sps/hep/damic/gaior/efficiency/UW/after_pfs
/sps/hep/damic/gaior/efficiency/kdata/expo
/sps/hep/damic/gaior/efficiency/kdata/expo2
/sps/hep/damic/gaior/efficiency/kdata/high


corresponding pkl files:
/sps/hep/damic/gaior/efficiency/pkl/




