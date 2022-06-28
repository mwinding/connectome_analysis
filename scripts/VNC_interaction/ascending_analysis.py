# %%
from pymaid_creds import url, name, password, token
from data_settings import data_date, data_date_A1_brain, pairs_path
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)
import navis
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from contools import Promat, Celltype, Celltype_Analyzer

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

rm = pymaid.CatmaidInstance(url, token, name, password)
select_neurons = pymaid.get_skids_by_annotation(['mw A1 neurons paired', 'mw dVNC'])
select_neurons = select_neurons + Celltype_Analyzer.get_skids_from_meta_annotation('mw A1 sensories')
ad_edges_A1 = Promat.pull_edges(type_edges='ad', threshold=0.01, data_date=data_date_A1_brain, pairs_combined=False, select_neurons=select_neurons)
pairs = Promat.get_pairs(pairs_path=pairs_path)

# load sensory types
A1_MN = pymaid.get_skids_by_annotation('mw A1 MN')

A1_proprio = pymaid.get_skids_by_annotation('mw A1 proprio')
A1_chordotonal = pymaid.get_skids_by_annotation('mw A1 chordotonals')
A1_noci = pymaid.get_skids_by_annotation('mw A1 noci')
A1_classII_III = pymaid.get_skids_by_annotation('mw A1 classII_III')
A1_external = pymaid.get_skids_by_annotation('mw A1 external sensories')
A1_unk = pymaid.get_skids_by_annotation('mw A1 unknown sensories')
dVNC = pymaid.get_skids_by_annotation('mw dVNC')

# determining hops from each sensory modality for each ascending neuron (using all hops)
names = ['us-MN', 'ds-Proprio', 'ds-Noci', 'ds-Chord', 'ds-ClassII_III', 'ds-ES', 'ds-unk']
general_names = ['pre-MN', 'Proprio', 'Noci', 'Chord', 'ClassII_III', 'ES', 'unk']
exclude = A1_MN + A1_proprio + A1_chordotonal + A1_noci + A1_classII_III + A1_external + A1_unk + dVNC

hops = 5
us_MN = Promat.upstream_multihop(edges=ad_edges_A1, sources=A1_MN, hops=hops, exclude=exclude, pairs=pairs)
ds_proprio = Promat.downstream_multihop(edges=ad_edges_A1, sources=A1_proprio, hops=hops, exclude=exclude, pairs=pairs)
ds_noci = Promat.downstream_multihop(edges=ad_edges_A1, sources=A1_noci, hops=hops, exclude=exclude, pairs=pairs)
ds_chord = Promat.downstream_multihop(edges=ad_edges_A1, sources=A1_chordotonal, hops=hops, exclude=exclude, pairs=pairs)
ds_classII_III = Promat.downstream_multihop(edges=ad_edges_A1, sources=A1_classII_III, hops=hops, exclude=exclude, pairs=pairs)
ds_external = Promat.downstream_multihop(edges=ad_edges_A1, sources=A1_external, hops=hops, exclude=exclude, pairs=pairs)
ds_unk = Promat.downstream_multihop(edges=ad_edges_A1, sources=A1_unk, hops=hops, exclude=exclude, pairs=pairs)

VNC_layers = [us_MN, ds_proprio, ds_noci, ds_chord, ds_classII_III, ds_external, ds_unk]
cat_order = ['pre-MN', 'Proprio', 'Noci', 'Chord', 'ClassII_III', 'ES', 'Unknown']

# %%
# identities of ascending neurons

# load ascending and sort by modality
A1_ascending = Celltype_Analyzer.get_skids_from_meta_annotation('mw A1 ascending', split=True)
ascending_pairs = [Promat.extract_pairs_from_list(skids, pairs)[0] for skids in A1_ascending[0]]

for i in range(len(ascending_pairs)):
    ascending_pairs[i]['type'] = [A1_ascending[1][i].replace('mw A1 ascending ', '')]*len(ascending_pairs[i])

ascending_pairs = pd.concat(ascending_pairs)
ascending_pairs = ascending_pairs.reset_index(drop=True)

VNC_layers = [[A1_MN]+us_MN, [A1_proprio]+ds_proprio, [A1_noci]+ds_noci, [A1_chordotonal]+ds_chord, [A1_classII_III]+ds_classII_III, [A1_external]+ds_external, [A1_unk]+ds_unk]
A1_ascending = Celltype_Analyzer.get_skids_from_meta_annotation('mw A1 ascending', split=False)
ascending_layers,ascending_skids = Celltype_Analyzer.layer_id(VNC_layers, general_names, A1_ascending)
sens_asc_mat, sens_asc_mat_plotting = Promat.hop_matrix(ascending_skids.T, general_names, ascending_pairs.leftid, include_start=True)

# only include 1 or 2-hop connections that are relevant
sens_asc_mat_plotting[sens_asc_mat_plotting<(hops-1)]=0

# remove coloring from 2-hop or greater connections to proprio (not relevant)
sens_asc_mat_plotting.iloc[:, 0:4][sens_asc_mat_plotting.iloc[:, 0:4]==4]=0

A00c_other_segments = [2511238, 2816457, 1042538]
cols = [x for x in sens_asc_mat_plotting.columns if x not in A00c_other_segments]

sort = [15315491, 14522404, 12998937, 8059283, 2123422, 10929797, 8644484, 7766016, 10949382, 5343578, 4595609, 3220616, 4206755, 4555763, 21250110, 18458734, 11455472, 3564452, 5606265, 8057753]

fig, ax = plt.subplots(1,1)
annots = sens_asc_mat.astype(int).astype(str)
annots[annots=='0']=''
sns.heatmap(sens_asc_mat_plotting.loc[['Proprio','Noci','Chord','ClassII_III','ES'], sort], annot=annots.loc[['Proprio','Noci','Chord','ClassII_III','ES'], sort], fmt='s', cmap='Blues', square=True, ax=ax)
plt.savefig('plots/ascending_identity_plot.pdf', format='pdf', bbox_inches='tight')

# %%

# setting up volumes for future cells
cns = pymaid.get_volume('cns')
neuropil = pymaid.get_volume('PS_Neuropil_manual')
SEZ_left = pymaid.get_volume('SEZ_left')
SEZ_right = pymaid.get_volume('SEZ_right')
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

mult = 0.8

# Set color and alpha of volumes
cns.color = (250, 250, 250, 0.1)
neuropil.color = (250, 250, 250, 0.1)

SEZ_left.color = (0, 0, 250, .05*mult)
SEZ_right.color = (0, 0, 250, .05*mult)

T1_left.color = (0, 0, 250, .03*mult)
T1_right.color = (0, 0, 250, .03*mult)

T2_left.color = (0, 250, 250, .075*mult)
T2_right.color = (0, 250, 250, .075*mult)

T3_left.color = (0, 250, 250, .04*mult)
T3_right.color = (0, 250, 250, .04*mult)

A1_left.color = (0, 250, 0, .075*mult)
A1_right.color = (0, 250, 0, .075*mult)

A2_left.color = (0, 250, 0, .04*mult)
A2_right.color = (0, 250, 0, .04*mult)

A3_left.color = (250, 250, 0, .08*mult)
A3_right.color = (250, 250, 0, .08*mult)

A4_left.color = (250, 250, 0, .04*mult)
A4_right.color = (250, 250, 0, .04*mult)

A5_left.color = (250, 0, 0, .06*mult)
A5_right.color = (250, 0, 0, .06*mult)

A6_left.color = (250, 0, 0, .03*mult)
A6_right.color = (250, 0, 0, .03*mult)

A7_left.color = (250, 0, 150, .05*mult)
A7_right.color = (250, 0, 150, .05*mult)

A8_left.color = (250, 0, 150, .025*mult)
A8_right.color = (250, 0, 150, .025*mult)

# %%
# plotting neurons

def plot_pair(num, neurons, cns, neuropil, segments, view, method, neurons_present=True):

    if(neurons_present):
        fig, ax = navis.plot2d([neurons, cns], method=method, color = '#444140', linewidth=1.5, connectors=True, cn_size=2)
    if(neurons_present==False):
        fig, ax = navis.plot2d([cns], method=method)

    if(view == 'side'):
        ax.azim= 0
        ax.dist = 5.2 # zoom
        navis.plot2d(neuropil, method=method, ax=ax)
        for segment in segments:
            navis.plot2d(segment, method=method, ax=ax)
        plt.show()
        if(neurons_present):
            fig.savefig(f'plots/individual-asc-morpho_{num}-{neurons[0].skeleton_id}-morphology-{view}.png', dpi=200)
        if(neurons_present==False):
            fig.savefig(f'plots/individual-asc-morpho_CNS-morphology-{view}.png', dpi=200)

    if(view == 'front'):
        ax.azim = 90
        ax.dist = 5.2 # zoom
        navis.plot2d(neuropil, method=method, ax=ax)
        for segment in segments:
            navis.plot2d(segment, method=method, ax=ax)
        plt.show()
        if(neurons_present):
            fig.savefig(f'plots/individual-asc-morpho_{num}-{neurons[0].skeleton_id}-morphology-{view}.png', dpi=200)
        if(neurons_present==False):
            fig.savefig(f'plots/individual-asc-morpho_CNS-morphology-{view}.png', dpi=200)
 
    if(view == 'top'):
        ax.elev=90
        ax.dist = 5.2 # zoom
        navis.plot2d(neuropil, method=method, ax=ax)
        for segment in segments:
            navis.plot2d(segment, method=method, ax=ax)
        plt.show()
        if(neurons_present):
            fig.savefig(f'plots/individual-asc-morpho_{num}-{neurons[0].skeleton_id}-morphology-{view}.png', dpi=200)
        if(neurons_present==False):
            fig.savefig(f'plots/individual-asc-morpho_CNS-morphology-{view}.png', dpi=200)
 
ascending_pairs = ascending_pairs.set_index('leftid', drop=False)
asc_pairs = [list(x[0:2]) for x in ascending_pairs.loc[sort].values]

segments = [T1_left, T1_right,
            T2_left, T2_right,
            T3_left, T3_right,
            A1_left, A1_right,
            A2_left, A2_right,
            A3_left, A3_right,
            A4_left, A4_right,
            A5_left, A5_right,
            A6_left, A6_right,
            A7_left, A7_right,
            A8_left, A8_right]

#neurons = pymaid.get_neurons(asc_pairs[0])
#plot_pair(0, neurons, cns, neuropil, segments, 'side')
method='3d_complex'

for i in range(0, len(asc_pairs)):
    neurons = pymaid.get_neurons(asc_pairs[i])
    plot_pair(i, neurons, cns, neuropil, segments, 'side', method)
'''
for i in range(0, len(asc_pairs)):
    neurons = pymaid.get_neurons(asc_pairs[i])
    plot_pair(i, neurons, cns, neuropil, segments, 'front', method)
'''
for i in range(0, len(asc_pairs)):
    neurons = pymaid.get_neurons(asc_pairs[i])
    plot_pair(i, neurons, cns, neuropil, segments, 'top', method)

segments = [SEZ_left, SEZ_right,
            T1_left, T1_right,
            T2_left, T2_right,
            T3_left, T3_right,
            A1_left, A1_right,
            A2_left, A2_right,
            A3_left, A3_right,
            A4_left, A4_right,
            A5_left, A5_right,
            A6_left, A6_right,
            A7_left, A7_right,
            A8_left, A8_right]

method='3d_complex'
#plot_pair('', [], cns, [], segments, 'side', method=method, neurons_present=False)
#plot_pair('', [], cns, [], segments, 'front', method=method,neurons_present=False)
#plot_pair('', [], cns, [], segments, 'top', method=method,neurons_present=False)

# %%
# plot dendritic synapses

SEZ_left = pymaid.get_volume('SEZ_left')
SEZ_right = pymaid.get_volume('SEZ_right')

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
neuropil_max = neuropil.bbox[2,1]
neuropil_min = neuropil.bbox[2,0]

boundary_z = [neuropil_min, SEZ_T1, T1_T2, T2_T3,
                T3_A1, A1_A2, A2_A3,
                A3_A4, A4_A5, A5_A6,
                A6_A7, A7_A8, neuropil_max]

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
