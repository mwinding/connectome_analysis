# %%
import os
import sys
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:
    pass

from pymaid_creds import url, name, password, token
import pymaid
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

rm = pymaid.CatmaidInstance(url, name, password, token)
adj = pd.read_csv('VNC_interaction/data/axon-dendrite.csv', header = 0, index_col = 0)
inputs = pd.read_csv('VNC_interaction/data/input_counts.csv', index_col = 0)
inputs = pd.DataFrame(inputs.values, index = inputs.index, columns = ['axon_input', 'dendrite_input'])
pairs = pd.read_csv('VNC_interaction/data/pairs-2020-10-26.csv', header = 0) # import pairs

sens_asc_mat_thresh = pd.read_csv('VNC_interaction/plots/individual_asc_paths/ascending_identity_2-hops.csv', header=0, index_col=0)

# %%
# plotting neurons

def plot_pair(num, neurons, cns, neuropil, segments, view):
    if(view == 'side'):
        fig, ax = pymaid.plot2d([neurons, cns], method='3d_complex', color = 'grey', linewidth=1.5, connectors=True)
        ax.azim=0
        ax.dist = 5 # zoom
        pymaid.plot2d(neuropil, method='3d_complex', ax=ax)
        for segment in segments:
            pymaid.plot2d(segment, method='3d_complex', ax=ax)
        plt.show()
        fig.savefig(f'VNC_interaction/plots/individual_asc_paths/{num}_{neurons[0].skeleton_id}_morphology_side.pdf')

    if(view == 'front'):
        fig, ax = pymaid.plot2d([neurons, cns] ,method='3d_complex', color = 'grey', linewidth=1.5, connectors=True)
        ax.azim = 90
        ax.dist = 2.5 # zoom
        pymaid.plot2d(neuropil, method='3d_complex', ax=ax)
        plt.show()
        fig.savefig(f'VNC_interaction/plots/individual_asc_paths/{num}_{neurons[0].skeleton_id}_morphology_front.pdf')
    
    if(view == 'top'):
        fig, ax = pymaid.plot2d([neurons, cns] ,method='3d_complex', color = 'grey', linewidth=1.5, connectors=True)
        ax.elev=90
        ax.dist = 5 # zoom
        pymaid.plot2d(neuropil, method='3d_complex', ax=ax)
        for segment in segments:
            pymaid.plot2d(segment, method='3d_complex', ax=ax)
        plt.show()
        fig.savefig(f'VNC_interaction/plots/individual_asc_paths/{num}_{neurons[0].skeleton_id}_morphology_top.pdf')


ascendings = [int(x) for x in sens_asc_mat_thresh.columns]
asc_pairs = [pairs[pairs.leftid==x].loc[:, ['leftid', 'rightid']].values for x in ascendings]
asc_pairs = [list(x) for sublist in asc_pairs for x in sublist]

#neuron_left = pymaid.get_neurons(asc_pairs[0][0])
#neuron_right = pymaid.get_neurons(asc_pairs[0][1])

cns = pymaid.get_volume('cns')
neuropil = pymaid.get_volume('PS_Neuropil_manual')
T1_left = pymaid.get_volume('T1_left')
T1_right = pymaid.get_volume('T1_right')
T2_left = pymaid.get_volume('T2_left')
T2_right = pymaid.get_volume('T2_right')
T3_left = pymaid.get_volume('T3_left')
T3_right = pymaid.get_volume('T3_right')
A1_left = pymaid.get_volume('A1_left')
A1_right = pymaid.get_volume('A1_right')
A2_left = pymaid.get_volume('A2_left')
A2_right = pymaid.get_volume('A2_right')
A3_left = pymaid.get_volume('A3_left')
A3_right = pymaid.get_volume('A3_right')

# Set color and alpha of volumes
cns.color = (250, 250, 250, .05)
neuropil.color = (250, 250, 250, .1)

T1_left.color = (0, 50, 250, .025)
T1_right.color = (0, 50, 250, .025)

T2_left.color = (0, 150, 250, .05)
T2_right.color = (0, 150, 250, .05)

T3_left.color = (0, 250, 250, .075)
T3_right.color = (0, 250, 250, .075)

A1_left.color = (0, 250, 0, .075)
A1_right.color = (0, 250, 0, .075)

A2_left.color = (200, 100, 0, .05)
A2_right.color = (200, 100, 0, .05)

A3_left.color = (250, 50, 0, .025)
A3_right.color = (250, 50, 0, .025)

segments = [T1_left, T1_right,
            T2_left, T2_right,
            T3_left, T3_right,
            A1_left, A1_right,
            A2_left, A2_right,
            A3_left, A3_right]

from tqdm import tqdm
'''
for i in tqdm(range(0, len(asc_pairs))):
    neurons = pymaid.get_neurons(asc_pairs[i])
    plot_pair(i, neurons, cns, neuropil, segments, 'side')
'''
for i in tqdm(range(0, len(asc_pairs))):
    neurons = pymaid.get_neurons(asc_pairs[i])
    plot_pair(i, neurons, cns, neuropil, segments, 'front')

for i in tqdm(range(0, len(asc_pairs))):
    neurons = pymaid.get_neurons(asc_pairs[i])
    plot_pair(i, neurons, cns, neuropil, segments, 'top')

# %%
