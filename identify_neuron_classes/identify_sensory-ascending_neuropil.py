#%%
import os
os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
import pymaid
import connectome_tools.cluster_analysis as clust
import connectome_tools.celltype as ct
import connectome_tools.process_graph as pg
import connectome_tools.process_matrix as pm
import navis

rm = pymaid.CatmaidInstance(url, token, name, password)

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

#%%
# load previously generated adjacency matrices
# see 'network_analysis/generate_all_edges.py'

brain_inputs = [x for sublist in [pymaid.get_skids_by_annotation(annot) for annot in pymaid.get_annotated('mw brain inputs and ascending').name] for x in sublist]
brain = pymaid.get_skids_by_annotation('mw brain neurons') + brain_inputs

adj_names = ['ad', 'aa', 'dd', 'da']
adj_ad, adj_aa, adj_dd, adj_da = [pd.read_csv(f'data/adj/all-neurons_{name}.csv', index_col = 0).rename(columns=int) for name in adj_names]
adj_ad = adj_ad.loc[np.intersect1d(adj_ad.index, brain), np.intersect1d(adj_ad.index, brain)]
adj_aa = adj_aa.loc[np.intersect1d(adj_aa.index, brain), np.intersect1d(adj_aa.index, brain)]
adj_dd = adj_dd.loc[np.intersect1d(adj_dd.index, brain), np.intersect1d(adj_dd.index, brain)]
adj_da = adj_da.loc[np.intersect1d(adj_da.index, brain), np.intersect1d(adj_da.index, brain)]
adjs = [adj_ad, adj_aa, adj_dd, adj_da]

# load input counts
inputs = pd.read_csv('data/graphs/inputs.csv', index_col=0)
outputs = pd.read_csv('data/graphs/outputs.csv', index_col=0)

# process to produce %input pairwise matrix
mat_ad, mat_aa, mat_dd, mat_da = [pm.Adjacency_matrix(adj=adj, input_counts=inputs, mat_type=adj_names[i]) for i, adj in enumerate(adjs)]
# %%
# load appropriate sensory and ascending types
input_types = [[annot.replace('mw ', ''), pymaid.get_skids_by_annotation(annot)] for annot in pymaid.get_annotated('mw brain inputs and ascending').name]
input_types = pd.DataFrame(input_types, columns = ['type', 'source'])

all_ascending = [pymaid.get_skids_by_annotation(annot) for annot in pymaid.get_annotated('mw brain ascendings').name]
all_ascending = [x for sublist in all_ascending for x in sublist]

input_types = input_types.append(pd.DataFrame([['A1 ascendings all', all_ascending]], columns = ['type', 'source']))
input_types = input_types.reset_index(drop=True)

# %%
# identify all 2nd-order neurons
threshold = 0.01
layers = [mat_ad.downstream_multihop(source=skids, threshold=threshold, hops=1, exclude=brain_inputs)[0] for skids in input_types.source]
input_types['order2'] = layers

# look at overlap between order2 neurons
cts = [ct.Celltype(input_types.type.iloc[i] + ' 2nd-order', input_types.order2.iloc[i]) for i in range(0, len(input_types.index))]
cts_analyze = ct.Celltype_Analyzer(cts)
sns.heatmap(cts_analyze.compare_membership(sim_type='iou').iloc[0:10, 0:10])

# export IDs
[pymaid.add_annotations(input_types.order2.iloc[i], f'mw brain 2nd_order {input_types.type.iloc[i]}') for i in range(0, 10)]
pymaid.add_meta_annotations([f'mw brain 2nd_order {input_types.type.iloc[i]}' for i in range(0, 10)], 'mw brain inputs 2nd_order')

# %%
# visualize neurons in adjacency
#all_order2 = [x for sublist in [input_types.order2.iloc[i] for i in range(0,10)] for x in sublist]
summed_adj = pd.read_csv(f'data/adj/all-neurons_all-all.csv', index_col = 0).rename(columns=int)
#plt.imshow(summed_adj.loc[np.intersect1d(summed_adj.index, all_order2),np.intersect1d(summed_adj.index, all_order2)], vmax=1, cmap='Reds')

# identify all PNs / LNs
def identify_LNs(summed_adj, adj_aa, skids, input_skids, outputs, pairs = pm.Promat.get_pairs(), sort = True):
    mat = summed_adj.loc[np.intersect1d(summed_adj.index, skids), np.intersect1d(summed_adj.index, skids)]
    mat = mat.sum(axis=1)

    mat_axon = adj_aa.loc[np.intersect1d(adj_aa.index, skids), np.intersect1d(adj_aa.index, input_skids)]
    mat_axon = mat_axon.sum(axis=1)

    # convert to % outputs
    skid_percent_output = []
    for skid in skids:
        skid_output = 0
        output = sum(outputs.loc[skid, :])
        if(output != 0):
            if(skid in mat.index):
                skid_output = skid_output + mat.loc[skid]/output
            if(skid in mat_axon.index):
                skid_output = skid_output + mat_axon.loc[skid]/output

        skid_percent_output.append([skid, skid_output])

    skid_percent_output = pm.Promat.convert_df_to_pairwise(pd.DataFrame(skid_percent_output, columns=['skid', 'percent_output_intragroup']).set_index('skid'))

    LNs = skid_percent_output.groupby('pair_id').sum()      
    LNs = LNs[np.array([x for sublist in (LNs>=1).values for x in sublist])]    
    return(list(LNs.index), skid_percent_output)

LNs = [identify_LNs(summed_adj, adj_aa, input_types.order2.iloc[i], input_types.source.iloc[i], outputs)[0] for i in range(0, len(input_types))]
LNs_data = [identify_LNs(summed_adj, adj_aa, input_types.order2.iloc[i], input_types.source.iloc[i], outputs)[1] for i in range(0, len(input_types))]

# %%
#

neuropil = pymaid.get_volume('PS_Neuropil_manual')
neuropil.color = (250, 250, 250, .05)
colors = ['#00753F', '#1D79B7', '#5D8C90', '#D88052', '#FF8734', '#E55560', '#F9EB4D', '#8C7700', '#9467BD','#D88052', '#A52A2A']

for i, skids in enumerate(input_types.order2):
    neurons = pymaid.get_neurons(skids)

    fig, ax = navis.plot2d(x=[neurons, neuropil], connectors_only=False, color=colors[i], alpha=0.5)
    ax.azim = -90
    ax.elev = -90
    ax.dist = 3.5
    plt.show()
    fig.savefig(f'identify_neuron_classes/plots/morpho_{input_types.type.iloc[i]}.pdf', format='pdf')


# %%
# identify all 3rd-order neurons
# identify all PNs / LNs



# %%
# plot 2nd-order, 3-order based on A-D, A-A, etc.