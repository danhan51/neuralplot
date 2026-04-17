from scipy.io import loadmat
import os
import numpy as np
import pandas as pd
from neuralplot import loadNeuralplot, Neuralplot #custom class for quad data
import tdt
import matplotlib.pyplot as plt
import pickle

animal_date_dict = {
    'Pancho': ['260305','260310'],
    'Diego': ['260304','260306']
    
}

for animal, dates in animal_date_dict.items():
    for date in dates:
        print(f'Doing {animal} {date}...')
        basedir = f'/home/danhan/code/prims_fixation_final/dat'
        os.makedirs(basedir,exist_ok=True)
        nplot = loadNeuralplot(animal, date)

        with open(f'{basedir}/{animal}_{date}_stims_df.pkl','wb') as f:
            pickle.dump(nplot.Dat,f)

        print('...Done')
