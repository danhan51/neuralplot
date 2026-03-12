from scipy.io import loadmat
import os
import numpy as np
import pandas as pd
from neuralplot import loadNeuralplot, Neuralplot #custom class for quad data
import tdt
import matplotlib.pyplot as plt

animal_date_dict = {
    # 'Pancho': ['260217','260218','260219'],
    'Diego': ['260211']
}

for animal, dates in animal_date_dict.items():
    for date in dates:
        print(f'Doing {animal} {date}...')
        basedir = f'/home/danhan/code/data/plots/fob_theo/{animal}/{date}'
        nplot = loadNeuralplot(animal, date)

        from neuralplot import REGIONS
        for r in REGIONS:
            channels = nplot.getChannelNumOrRegionName(r, return_as = 'list')

            params = {
            'fixation_success_binary': [True],
            }

            savedir = f'{basedir}/rasters_each_site/{r}'

            if not os.path.exists(savedir):
                os.makedirs(savedir)
            print(f'Plotting rasters for each unit in {r}...')
            for channel in channels:
                fig_dict = nplot.plotRaster(channel,params, window = (0.4,1.0))
                for index, fig in fig_dict.items():
                    fig.savefig(f'{savedir}/{r}_{channel}_{index}.png')
                    plt.close(fig)
                plt.close('all')
            print('... Done')
            print(f'Plotting PSTH for each {r} site accross whole day...')
            for channel in channels:                                                    
                fig_dict = nplot.plotPSTH([channel], params, group_by='fixation_success_binary')

                savedir = f'{basedir}/psth_whole_day/{r}'
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                for index,fig in fig_dict.items():
                    fig.savefig(f'{savedir}/{r}_{channel}_{index}.png')
                    plt.close(fig)
                plt.close('all')
            print('... Done')
