import glob
#import os

import numpy as np
import pandas as pd


#%% Function to generate dataframe out of simulation data files written by ExtendedProfileSampling plugin
def eps2df(folderSims, fileName='ExtendedProfileSampling_TS', quietMode=False):
    '''
    Function to generate dataframe out of simulation data files written by ExtendedProfileSampling plugin

    :param str folderSims: Path to the simulation folder
    :param str quietMode: Do not print imported files (default: 0)
    :return: dataframe including the simulation data
    '''
    
    ## Look in path 'folderSims' for desired files (profile data)
    # flist_simple      = sorted(glob.glob(folderSims+'/ExtendedProfileSampling_TS*.dat'))
    flist_simple      = sorted(glob.glob(folderSims+'/'+fileName+'*.dat'))
    #flist_higher_moms = sorted(glob.glob(folderSims+'/ExtendedProfileSampling_HigherMoments_TS*.dat'))
    
    dfTemp1 = pd.read_csv(flist_simple[0], delim_whitespace=True)
    
    timesteps = list()
    
    for fp in flist_simple:
        timestep = int(fp.rpartition('TS')[-1].split('.')[0])
        if timestep not in timesteps:
            timesteps.append(timestep)
        
    components = np.arange(int((len(dfTemp1.columns)-1)/21))
    pos = dfTemp1.loc[:,'pos']
    
    li1= []
    for datei in flist_simple:
        li2= []
        if not quietMode: print(datei)
        timestep = int(datei.rpartition('TS')[-1].split('.')[0])
        df = pd.read_csv(datei, delim_whitespace=True)
        for comp in components:
            dfTemp = df.filter(regex="\["+str(comp)+"\]")
            dfTemp.columns = dfTemp.columns.str.replace('\['+str(comp)+'\]','', regex=True)
            dfTemp.loc[:,'cid'] = comp
            dfTemp.loc[:,'timestep'] = timestep
            li2.append(dfTemp)
    
        li1.append(pd.concat(li2,axis=0,ignore_index=True))
    
    dfProfScalVect = pd.concat(li1,axis=0,ignore_index=True)
    dfProfScalVect = dfProfScalVect.assign(pos=np.tile(pos,int(dfProfScalVect.shape[0]/len(pos))))
    
    return dfProfScalVect


#%% Main program which can be called from the command line; Will convert profile data to dataframe including export
if __name__ == '__main__':
    
    flgMtdMain = 2 ## Decide on input: 1 for command line argument; 2 for hard coded values
    
    if flgMtdMain == 1:
        import argparse
        argparser = argparse.ArgumentParser(description='Simulation profile data to dataframe')
        argparser.add_argument('-s','--simPath', required=True, type=str, help='Path of simulation data')
        
        args = vars(argparser.parse_args())
        simPath = args['simPath']
        
    elif flgMtdMain == 2:
        print('Input set manually in script')
        simPath = '/home/pcfsuser/Simon/projects/005_Interface_resistivity/simulations/1clj/statEvap/T074_v01/run02'
        
    else:
        print('Requested input method not available')
    
    print('Path to simulation: '+simPath)
    
    dfDataProf = eps2df(simPath)
    
    print('Done')
    
