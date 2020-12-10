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
        fig.savefig(f'VNC_interaction/plots/individual_asc_paths/{num}_{neurons[0].skeleton_id}_morphology_side.png', dpi=200)

    if(view == 'front'):
        fig, ax = pymaid.plot2d([neurons, cns] ,method='3d_complex', color = 'grey', linewidth=1.5, connectors=True)
        ax.azim = 90
        ax.dist = 2.5 # zoom
        pymaid.plot2d(neuropil, method='3d_complex', ax=ax)
        plt.show()
        fig.savefig(f'VNC_interaction/plots/individual_asc_paths/{num}_{neurons[0].skeleton_id}_morphology_front.png', dpi=200)
    
    if(view == 'top'):
        fig, ax = pymaid.plot2d([neurons, cns] ,method='3d_complex', color = 'grey', linewidth=1.5, connectors=True)
        ax.elev=90
        ax.dist = 5 # zoom
        pymaid.plot2d(neuropil, method='3d_complex', ax=ax)
        for segment in segments:
            pymaid.plot2d(segment, method='3d_complex', ax=ax)
        plt.show()
        fig.savefig(f'VNC_interaction/plots/individual_asc_morpho/{num}_{neurons[0].skeleton_id}_morphology_top.png', dpi=200)


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

#neurons = pymaid.get_neurons(asc_pairs[0])
#plot_pair(0, neurons, cns, neuropil, segments, 'side')

for i in range(0, len(asc_pairs)):
    neurons = pymaid.get_neurons(asc_pairs[i])
    plot_pair(i, neurons, cns, neuropil, segments, 'side')

for i in range(0, len(asc_pairs)):
    neurons = pymaid.get_neurons(asc_pairs[i])
    plot_pair(i, neurons, cns, neuropil, segments, 'front')

for i in range(0, len(asc_pairs)):
    neurons = pymaid.get_neurons(asc_pairs[i])
    plot_pair(i, neurons, cns, neuropil, segments, 'top')

# %%
# plot dendritic synapses

SEZ_left = pymaid.get_volume('SEZ_left')
SEZ_right = pymaid.get_volume('SEZ_right')
A4_left = pymaid.get_volume('A4_left')
A4_right = pymaid.get_volume('A4_right')
A5_left = pymaid.get_volume('A5_left')
A5_right = pymaid.get_volume('A5_right')
A6_left = pymaid.get_volume('A6_left')
A6_right = pymaid.get_volume('A6_right')
A7_left = pymaid.get_volume('A7_left')
A7_right = pymaid.get_volume('A7_right')
A8_left = pymaid.get_volume('A8_left')
A8_right = pymaid.get_volume('A8_right')

# calculate edges of each segment
def volume_edges(vol_left, vol_right):
    vol_min = np.mean([vol_left.bbox[2,0], vol_right.bbox[2,0]])
    vol_max = np.mean([vol_left.bbox[2,1], vol_right.bbox[2,1]])
    return(vol_min, vol_max)

SEZ_min, SEZ_max = volume_edges(SEZ_left, SEZ_right)
T1_min, T1_max = volume_edges(T1_left, T1_right)
T2_min, T2_max = volume_edges(T2_left, T2_right)
T3_min, T3_max = volume_edges(T3_left, T3_right)
A1_min, A1_max = volume_edges(A1_left, A1_right)
A2_min, A2_max = volume_edges(A2_left, A2_right)
A3_min, A3_max = volume_edges(A3_left, A3_right)
A4_min, A4_max = volume_edges(A4_left, A4_right)
A5_min, A5_max = volume_edges(A5_left, A5_right)
A6_min, A6_max = volume_edges(A6_left, A6_right)
A7_min, A7_max = volume_edges(A7_left, A7_right)
A8_min, A8_max = volume_edges(A8_left, A8_right)

SEZ_T1 = np.mean([SEZ_max, T1_min])
T1_T2 = np.mean([T1_max, T2_min])
T2_T3 = np.mean([T2_max, T3_min])
T3_A1 = np.mean([T3_max, A1_min])
A1_A2 = np.mean([A1_max, A2_min])
A2_A3 = np.mean([A2_max, A3_min])
A3_A4 = np.mean([A3_max, A4_min])
A4_A5 = np.mean([A4_max, A5_min])
A5_A6 = np.mean([A5_max, A6_min])
A6_A7 = np.mean([A6_max, A7_min])
A7_A8 = np.mean([A7_max, A8_min])
end_A8_neuropil = neuropil.bbox[2,1]

boundary_z = [SEZ_min, SEZ_T1, T1_T2, T2_T3,
                T3_A1, A1_A2, A2_A3,
                A3_A4, A4_A5, A5_A6,
                A6_A7, A7_A8, end_A8_neuropil]

def plot_pair_split(num, neurons, min_z, max_z, bin_num, draw_boundaries):
    cut1 = pymaid.cut_neuron(neurons[0], 'mw axon split')
    outputs1 = cut1[0].connectors[cut1[0].connectors['relation']==0] # axon outputs
    inputs1 = cut1[1].connectors[cut1[1].connectors['relation']==1] # dendrite inputs

    cut2 = pymaid.cut_neuron(neurons[1], 'mw axon split')
    outputs2 = cut2[0].connectors[cut2[0].connectors['relation']==0] # axon outputs
    inputs2 = cut2[1].connectors[cut2[1].connectors['relation']==1] # dendrite inputs

    fig, axs = plt.subplots(1,1, figsize=(1.115, 0.25))
    ax = axs
    sns.distplot(list(outputs1.z)+list(outputs2.z), color = 'crimson', kde = False, kde_kws = {'shade': True}, ax=ax, bins=range(int(min_z), int(max_z), int(max_z/bin_num)))
    sns.distplot(list(inputs1.z)+list(inputs2.z), color = 'royalblue', kde = False, kde_kws = {'shade': True}, ax=ax, bins=range(int(min_z), int(max_z), int(max_z/bin_num)))
    #sns.distplot(outputs1.x, color = 'crimson', kde=False, bins = 30)
    
    ax.set(xlim=(min_z,max_z), ylim=(0, 100), xlabel='', xticks=([])) # set x/y axis limits

    # draw boundaries between segments
    for boundary in draw_boundaries:
        plt.axvline(x=int(boundary), lw = 0.1, color = 'k')

    # change width of border lines
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.25)

    # change size of yticks
    plt.tick_params(axis='both', which='major', labelsize=5, length=1.5, width=0.25)

    fig.savefig(f'VNC_interaction/plots/individual_asc_morpho/synapse-distribution_{num}_{neurons[0].skeleton_id}.pdf', bbox_inches = 'tight')

ascendings = [int(x) for x in sens_asc_mat_thresh.columns]
asc_pairs = [pairs[pairs.leftid==x].loc[:, ['leftid', 'rightid']].values for x in ascendings]

for i, pair in enumerate(asc_pairs):
    neurons = pymaid.get_neurons(pair)
    plot_pair_split(i, neurons, min_z = cns.bbox[2,0], max_z = cns.bbox[2,1], bin_num = 50, draw_boundaries = boundary_z)
    
# %%
