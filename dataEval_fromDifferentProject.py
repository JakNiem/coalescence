#%%
import os
import shutil
import random
from site import execsitecustomize
import numpy as np
import pandas as pd
import sys

from datetime import datetime

import math

import matplotlib.pyplot as plt
import csv
import re



def main(args):

    dirList = os.listdir('.')
    folders = [f for f in dirList if os.path.isdir(f)]
    folders.sort()

    # outFilePath = os.path.normpath('regionSamplingEval.csv')
    for f in folders:
        print(f'Folder: {f}')
        if os.path.exists(f): 
            evalRegionSampling(os.path.normpath(f))
            print('----------------------------')
        else:
            print('folder does not seem to exist')
    # writeOutput(output, outFilePath)
    return 0




def evalRegionSampling(folder):

    # find relevant files
    dirList = os.listdir(folder)
    scalarSamples = [f for f in dirList if 'scalquant_all_reg1' in f]
    # print(scalarSamples)

    if len(scalarSamples) == 0:
        print(f'no relevant files found in folder {folder}.')
        return None

    print(f'{len(scalarSamples)} samples found for dir {folder}')
    #combine all dataframes into one dataframe 
    allData = pd.DataFrame()
    for sample in scalarSamples:
        samplePath = os.path.join(folder, sample)
        simstep = sample[21:-4]
        # print(f"sample: {sample}, simstep read: {simstep}, parsed to int: {int(simstep)}")
        a = pd.read_csv(samplePath, delimiter='\s+')
        a.insert(0, "simstep", [int(simstep)]*len(a), True) 
        allData = pd.concat([allData, a])

    # print(allData.head(20))
    # print(allData["simstep"])
    
    positions = allData['pos'].unique()
    listOfAverages = [allData.loc[allData['pos'] == pos].mean(axis = 0) for pos in positions]
    averages = pd.DataFrame(listOfAverages)
    
    print(averages)


    pIn = averages['p[1]'].iloc[int(len(averages)/2)]
    pOut = (averages['p[1]'].iloc[-1] + averages['p[1]'].iloc[0])/2

    avgTotal = averages.mean()

    print(f'pIn: {pIn}')
    print(f'pOut: {pOut}')
    print(f'abs(DeltaP): {(abs(pOut-pIn))}')

    pAvg = avgTotal['p[1]']
    rhoAvg = avgTotal['rho[0]']
    print(f'pAvgTotal: {pAvg}')
    print(f'rhoAvgTotal: {rhoAvg}')
    

    return averages




# Call main() at the bottom, after all the functions have been defined.
if __name__ == '__main__':
    main(sys.argv)
    # print(df3)

# %%
