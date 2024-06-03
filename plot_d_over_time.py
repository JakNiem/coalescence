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

# import imageio
# import matplotlib.cm as cm
# import matplotlib.animation as animation


def approxIndexRow(df, approxIndex):
    return abs(df.index.to_series()-approxIndex).idxmin()





## find all sims:
dirList = os.listdir('.')
folders = [f for f in dirList if os.path.isdir(f)]
sims = [f for f in folders if f.startswith('T')]
sims.sort()
print(sims)







## CENTRAL LINE VISUALIZATION:
plt.figure(figsize = (18.5, 10.5))
plt.grid()

for simdir in sims:
    [T, r, dInital, v1v2] =  re.findall(r"[-+]?(?:\d*\.*\d+)", simdir)
    # re.findall(r"[-+]?(?:\d*\.*\d+)", "Current Level: -13.2db or 14.2 or 3")


    print('------------------')
    print(f'directory: {simdir}')

    dirList = os.listdir(simdir)
    coalSamplingFiles = [f for f in dirList if f.startswith('coalSampling_')]
    # print(dirList)
    if len(coalSamplingFiles) == 0:
        print(f'no relevant files found in directory {simdir}.')
    else:
        coalSamplingFiles = sorted(coalSamplingFiles)
        if len(coalSamplingFiles) > 1:
            print(f'{len(coalSamplingFiles)} files beginning with "coalSampling_" found in dir {simdir}. Using first file, named {coalSamplingFiles[0]}')
        


        # load and transpose DataFrame
        caolSamplePath = os.path.join(simdir, coalSamplingFiles[0])
        rhoDF = pd.read_csv(caolSamplePath, delimiter='\s+')   #  index (rows)= simstep;  colums = y-coordinate  
        rhoDF.set_index('simstep', drop = True, inplace= True)
        rhoDF = rhoDF.transpose()
        rhoDF.index = rhoDF.index.map(float)
        


        # calculating rhoLiq & rhoVap:
        rhoDF_rolling = rhoDF.rolling(window = 20, center = True, axis='rows').mean()
        idDropcenter1Approx = (int) (len(rhoDF)/3)
        idDropcenter2Approx = (int) (2*len(rhoDF)/3)
        rhoLiq = np.mean([rhoDF_rolling.iloc[idDropcenter1Approx].mean(), rhoDF_rolling.iloc[idDropcenter2Approx].mean()])

        rhoDF_rolling = rhoDF.rolling(window = 3, center = True, axis='rows').mean()
        rhoVap = np.mean([rhoDF_rolling.iloc[1].mean(), rhoDF_rolling.iloc[-2].mean()])
        rhoGibbs = (rhoLiq+rhoVap)/2

        print(f'{simdir}: rhoLiq  = {rhoLiq}, rhoVap = {rhoVap}, rhoGibbs = {rhoGibbs}')


        





        # plot of rho on y-axis for each sim
        showplot = False
        # showplot = True
        if(showplot):
            rhoDF_plot = rhoDF.rolling(window = 3, center = True, axis='rows').mean()
            rhoDF_plot.plot(legend=False)
            plt.axhline(y=rhoLiq, color='k', linestyle='--')
            plt.axhline(y=rhoVap, color='k', linestyle='--')
            plt.axhline(y=(rhoLiq+rhoVap)/2, color='k', linestyle='--')
            plt.grid()
            plt.show()



    
        # plot of development of rho in contact point
        rhoDF_rolling = rhoDF.rolling(window = 7, center = True, axis='rows').mean()
        idSymPlane = (int) (len(rhoDF)/2)

        rhoY_symPlane = rhoDF_rolling.iloc[idSymPlane]

        #####criterion: first simstep, where rho>rhoGibbs
        # stepsWithContact= (rhoY_symPlane>rhoGibbs)[rhoY_symPlane>rhoGibbs]
        # if(len(stepsWithContact)==0):
        #     touch = float('nan')
        # else:
        #     touch =stepsWithContact.index.tolist()[0]
         
        ####criterion: last simstep, where spheres don't touch
        mask = rhoY_symPlane>rhoGibbs
        touch = mask[mask==False].index[-1] +5

        showplot = False
        # showplot = True
        if(showplot):
            plt.plot(touch,dInital,'o', label = simdir) 
            print(rhoDF.index[-1])
            plt.axvline(x= rhoDF.columns[-1], color = 'k')
            plt.grid()
            plt.legend()
            plt.title('d_inital over simstep of first touch')
            plt.savefig('d_inital_over_simstepOfTouch')


        # plot of development of rho in contact point
        showplot = False
        showplot = True
        if(showplot):
            plt.axhline(y=(rhoLiq+rhoVap)/2, color = 'k', linestyle='--')
            plt.plot(rhoDF_rolling.iloc[idSymPlane], label = simdir)
            plt.grid()
            # plt.legend(loc = 'lower right')
            plt.legend()
            plt.title('rho at contact point over time (simstep)')
            plt.savefig('rhoContact_over_time')

















