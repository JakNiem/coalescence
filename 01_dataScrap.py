#%%
import os
from site import execsitecustomize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math

import re


dirList = os.listdir('.')
folders = [f for f in dirList if os.path.isdir(f)]
# sims = [f for f in folders if f.startswith('T')]
sims = [f for f in folders if f.startswith('T0.7_r20_d4')]

# sims = ["T0.7_r6_d2", "T0.7_r6_d4"]


nLayers = 3 # number of layers to average over 
width = .5

            
def rhoSeries(df):
    rhoDF = df.pivot(index='radius', columns='height', values='rho')
    # print('-----------------------------')
    # print(rhoDF)

    ### averaging over several layers (r-direction):
    numParticlesDF = rhoDF.mul(rhoDF.index*(2*math.pi*width), axis = 0)
    VTotal = (nLayers*width)**2 * math.pi * width
    rho = numParticlesDF.iloc[0:nLayers].sum()/VTotal
    ### 
    return rho


postprocList = [] #list of dicts
for simdir in sims:
    integerList = list(map(float, re.findall(r'[0-9,-.]+', simdir)))
    T = integerList[0]
    r = integerList[1]
    d_initial = integerList[2]
    if len(integerList) > 3: # legacy compatability
        n = integerList[3]
    else: 
        n = 0

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
        firstSampleDF = pd.read_csv(os.path.join(simdir, scalarSamples[0]), delimiter='\s+')
        rho_firstSample = rhoSeries(firstSampleDF)
        rhoLiq = rho_firstSample.max()
        print(f'calculated liquid density: {rhoLiq}')
        ######### HARDCODED ::::::::::::::::
        rhoLiq = 6.
        
        rhoGibbs = rhoLiq/2    
        domainHeight = rho_firstSample.index.max()

        # dataDict = {'T':[], 'r':[],'d_init':[], 't':[], 'd':  []}
        touch = False
        for sample in scalarSamples:
            timeStep = int(re.findall(r'\d+', sample)[0]) 

            samplePath = os.path.join(simdir, sample)
            sampledf = pd.read_csv(samplePath, delimiter='\s+')

            rhoDF = sampledf.pivot(index='radius', columns='height', values='rho')
            # print('-----------------------------')
            # print(rhoDF)

            # ### averaging over several layers (r-direction):
            # numParticlesDF = rhoDF.mul(rhoDF.index*(2*math.pi*width)*width, axis = 0)
            # VTotal = (nLayers*width)**2 * math.pi * width
            # rho = numParticlesDF.iloc[0:nLayers].sum()/VTotal
            # ### 
            print(rhoDF.iloc[0])
            rho = rhoDF.iloc[[0, 1, 2, 3, 4,5,6,7]].sum()/8

            
            rightMaxId = rho[rho.index >= domainHeight/2].idxmax()
            leftMaxId = rho[rho.index < domainHeight/2].idxmax()

            gapY = rho[((rho.index >= leftMaxId)  & (rho < rhoGibbs) & (rho.index <= rightMaxId))].index
            gapLimitsY = [gapY.min(), gapY.max()]
            d = gapY.max() - gapY.min()  
            if math.isnan(d):
                d = 0.

            firstTouch = False
            if (d == 0) and (touch == False):
                touch = True
                firstTouch = True

            sampleDict = {'dir': simdir, 'T':T, 'r':r,'d_init':d_initial, 'n': n, 'rhoLiq':rhoLiq, 
                          't':timeStep, 'd': d, 'firstTouch': firstTouch}

            # print(f'data for sample {sample}: {sampleDict}')
            postprocList.append(sampleDict)


postprocDF = pd.DataFrame(postprocList)
postprocDF.to_csv("postprocessing_dataScrap.csv", sep = ',')

print('postprocessing_dataScrap.csv done.')


print('all done.')


