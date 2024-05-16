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
sims = [f for f in folders if f.startswith('T')]

# sims = ["T0.7_r6_d2", "T0.7_r6_d4"]


nLayers = 3 # number of layers to average over 
width = .5

            


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
        sampledf = pd.read_csv(os.path.join(simdir, scalarSamples[0]), delimiter='\s+')
        rhoDF_fistSample = sampledf.pivot(index='radius', columns='height', values='rho').iloc[[0, 1, 2, 3, 4]].sum()
        rhoLiq = rhoDF_fistSample.rolling(window = 8, center = True).mean().max()
        print(f'calculated liquid density: {rhoLiq}')
        rhoGibbs = rhoLiq/2    
        domainHeight = rhoDF_fistSample.index.max()

        # dataDict = {'T':[], 'r':[],'d_init':[], 't':[], 'd':  []}
        touch = False
        for sample in scalarSamples:
            timeStep = int(re.findall(r'\d+', sample)[0]) 

            samplePath = os.path.join(simdir, sample)
            sampledf = pd.read_csv(samplePath, delimiter='\s+')

            rhoDF = sampledf.pivot(index='radius', columns='height', values='rho')
            print('-----------------------------')
            print(rhoDF)
            

            ### averaging over several layers (r-direction):
            numParticlesDF = rhoDF.mul(rhoDF.index*(2*math.pi*width), axis = 0)
            VTotal = (nLayers*width)**2 * math.pi * width
            rho = numParticlesDF.iloc[0:nLayers].sum()*VTotal
            ### 


            
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

            sampleDict = {'dir': simdir, 'T':T, 'r':r,'d_init':d_initial, 'i': n,
                          't':timeStep, 'd': d, 'firstTouch': firstTouch}

            print(f'data for sample {sample}: {sampleDict}')
            postprocList.append(sampleDict)


postprocDF = pd.DataFrame(postprocList)
postprocDF.to_csv("postprocessing_dataScrap.csv", sep = ',')

print('postprocessing_dataScrap.csv done.')



print('generating tContactDF ...')
contactList = []
for dir in postprocDF['dir'].unique():
    print(f'dir: {dir}')
    simDF = postprocDF[postprocDF['dir'] == dir]
    
    T = simDF.iloc[0].T
    r = simDF.iloc[0].r
    d_init = simDF.iloc[0].d_init
    n = simDF.iloc[0].i

    print(simDF)
    d0 = simDF.iloc[0].d

    tContact = simDF[simDF['d']==0].t

    print(f'tContact for {dir}: \n {tContact}')

    if tContact.size == 0:
        tFirstContact = float('nan')
    else:
        tFirstContact = tContact.iloc[0]
    print(f'tFirstContact for {dir}: \n {tFirstContact}')

    contactList.append({'dir': simdir, 'T':T, 'r':r, 'd_init': d_init, 'i': n, 'd0':d0, 'tTouch':tFirstContact})

contactDF = pd.DataFrame(contactList)
print('contactDF:')
print(contactDF)
contactDF.to_csv("postprocessing_contactDF.csv", sep = ',')


    # if sim['firstTouch'].values.sum() == 0:
    #     tContact = 0
    # else
    #     tContact = 





print('contactDF done.')
print('all done.')


