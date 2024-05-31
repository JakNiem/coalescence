#%%
import os
from site import execsitecustomize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math

import re


postprocDF = pd.read_csv("postprocessing_dataScrap.csv")


print('generating tContactDF ...')
contactList = []
for dir in postprocDF['dir'].unique():
    print(f'dir: {dir}')
    simDF = postprocDF[postprocDF['dir'] == dir]
    T = simDF.iloc[0]['T']
    r = simDF.iloc[0]['r']
    d_init = simDF.iloc[0]['d_init']
    n = simDF.iloc[0]['n']

    # print(simDF)
    d0 = simDF.iloc[0].d

    tContact = simDF[simDF['d']==0].t

    # print(f'tContact for {dir}: \n {tContact}')

    if tContact.size == 0:
        tFirstContact = float('nan')
    else:
        tFirstContact = tContact.iloc[0]
    print(f'tFirstContact for {dir}: \n {tFirstContact}')

    contactList.append({'dir': dir.strip(), 'T':T, 'r':r, 'd_init': d_init, 'n': n, 'd0':d0, 'tTouch':tFirstContact})

print(contactList)
contactDF = pd.DataFrame(contactList)
print('contactDF:')
print(contactDF)
contactDF.to_csv("postprocessing_contactDF.csv", sep = ',')


    # if sim['firstTouch'].values.sum() == 0:
    #     tContact = 0
    # else
    #     tContact = 





print('contactDF done.')