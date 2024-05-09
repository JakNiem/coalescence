import glob
import numpy as np
import pandas as pd


#%% Function to generate dataframe out of simulation data files written by ExtendedProfileSampling plugin
def gamma2df(folderSims, quietMode=False):
    '''
    Function to generate dataframe out of simulation data files written by GammaWriter plugin

    :param str folderSims: Path to the simulation folder
    :param str quietMode: Do not print imported files (default: 0)
    :return: dataframe including gamma
    '''
    
    ## Look in path 'folderSims' for desired gamma files
    flist_gamma = sorted(glob.glob(folderSims+'/gamma.dat'))


    li = []

    for datei in flist_gamma:
        if not quietMode: print(datei)
        ds = pd.read_csv(datei, skiprows=2, delim_whitespace=True)
        li.append(ds)
    
    dfGamma = pd.concat(li, axis=0, ignore_index=True)
    dfGamma.rename(columns={'simstep': 'timestep'}, inplace=True)

    
    return dfGamma


#%% Main program which can be called from the command line; Will convert profile data to dataframe including export
if __name__ == '__main__':
    
    print('Do not call directly! Use function instead.')
