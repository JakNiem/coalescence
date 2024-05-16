#%%
import os
from site import execsitecustomize
import numpy as np
import pandas as pd

import math

import re


dirList = os.listdir('.')
folders = [f for f in dirList if os.path.isdir(f)]
sims = [f for f in folders if f.startswith('T')]

# sims = ["T0.7_r6_d2", "T0.7_r6_d4"]




            


postprocList = [] #list of dicts
for simdir in sims:
    integerList = list(map(float, re.findall(r'[0-9,-.]+', simdir)))
    T = integerList[0]
    r = integerList[1]
    d_initial = integerList[2]

    print(f'directory: {simdir}, T={T}, r={r}, d={d_initial}')

    dirList = os.listdir(simdir)
    scalarSamples = [f for f in dirList if 'CylindricSampling' in f]
    # print(dirList)
    if len(scalarSamples) == 0:
        print(f'no relevant files found in directory {simdir}.')
    else:
        scalarSamples = sorted(scalarSamples)
        maxTimeStep =  int(re.findall(r'\d+', scalarSamples[-1])[0])
        print(f'max timestep = {maxTimeStep}')

        print(f"folder: {simdir}")


        # scalarSamples = [scalarSamples[1]]
        sampledf = pd.read_csv(os.path.join(simdir, scalarSamples[0]), delimiter='\s+')
        rhoDF_fistSample = sampledf.pivot(index='radius', columns='height', values='rho').iloc[[0, 1, 2, 3, 4]].sum()
        rhoLiq = rhoDF_fistSample.rolling(window = 8, center = True).mean().max()
        print(f'calculated liquid density: {rhoLiq}')
        rhoGibbs = rhoLiq/2    
        domainHeight = rhoDF_fistSample.index.max()

        # dataDict = {'T':[], 'r':[],'d_init':[], 't':[], 'd':  []}
        touch = False
        for sample in scalarSamples:
            # print(f'extracting data from sample {sample}')
            timeStep = int(re.findall(r'\d+', sample)[0]) 

            samplePath = os.path.join(simdir, sample)
            sampledf = pd.read_csv(samplePath, delimiter='\s+')
            #  height,  radius, numParts, rho, T,   ekin, p,  T_r,   T_y, T_t,   v_r,   v_y, v_t, p_r, p_y,     p_t, numSamples

            rhoDF = sampledf.pivot(index='radius', columns='height', values='rho')
            rho = rhoDF.iloc[[0, 1, 2, 3, 4]].sum() #series of rho with index y
            # rhoRollingNarrow = rho.rolling(window = 3, center = True).mean()
            # rhoRollingWide = rho.rolling(window = 8, center = True).mean()

            # print(f'rho: \n {rho}')
            # print('------------')
            # print(f'idmax: {rho.idxmax()}')
            # print('------------')

            rightMaxId = rho[rho.index >= domainHeight/2].idxmax()
            leftMaxId = rho[rho.index < domainHeight/2].idxmax()
            # # print(f'left: {leftMaxId}, right: {rightMaxId}')
            # rhoCentral = rho[((rho.index >= leftMaxId)  & (rho.index <= rightMaxId))]

            gapY = rho[((rho.index >= leftMaxId)  & (rho < rhoGibbs) & (rho.index <= rightMaxId))].index
            gapLimitsY = [gapY.min(), gapY.max()]
            d = gapY.max() - gapY.min()  
            if math.isnan(d):
                d = 0.

            firstTouch = False
            if (d == 0) and (touch == False):
                touch = True
                firstTouch = True

            sampleDict = {'dir': simdir, 'T':T, 'r':r,'d_init':d_initial, 't':timeStep, 'd': d, 'firstTouch': firstTouch}

            print(f'data for sample {sample}: {sampleDict}')
            postprocList.append(sampleDict)


postprocDF = pd.DataFrame(postprocList)
# print(postprocDF)
postprocDF.to_csv("postprocessing_dataScrap.csv", sep = ',')

print('postprocessing_dataScrap.csv done.')



print('generating tContactDF ...')
for dir in postprocDF['simdir'].unique():
    print(f'dir: {dir}')
    simDF = postprocDF[postprocDF['dir'] == dir]
    print(simDF)

    tContact = simDF[simDF['d']==0].t

    print(f'tContact for {dir}: {tContact}')

    # if sim['firstTouch'].values.sum() == 0:
    #     tContact = 0
    # else
    #     tContact = 









print('done.')
print('all done.')


