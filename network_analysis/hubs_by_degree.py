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

rm = pymaid.CatmaidInstance(url, token, name, password)

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

# %%
# load previously generated edge lists with pairwise threshold
# see 'network_analysis/generate_all_edges.py'

adj_names = ['ad', 'aa', 'dd', 'da']
edge_lists = [pd.read_csv(f'data/edges_threshold/pairwise-threshold_{name}_all-edges.csv', index_col = 0) for name in adj_names]
Gad, Gaa, Gdd, Gda = [pg.Analyze_Nx_G(edge_list, graph_type='directed', split_pairs=True) for edge_list in edge_lists]
Gs = [Gad, Gaa, Gdd, Gda]
# %%
# extract in-degree / out-degree data for each neuron and identify hubs
threshold = 20
G_hubs = [obj.get_node_degrees(hub_threshold=threshold) for obj in Gs]

# %%
# plot data

for i, hubs in enumerate(G_hubs):
    hubs_plot = hubs.groupby(['in_degree', 'out_degree']).count().iloc[:, 0].reset_index()
    hubs_plot.columns = ['in_degree', 'out_degree', 'count']

    hub_type = []
    for index in range(0, len(hubs_plot)):
        if((hubs_plot.iloc[index, :].in_degree>=threshold) & (hubs_plot.iloc[index, :].out_degree<threshold)):
            hub_type.append('in_hub')
        if((hubs_plot.iloc[index, :].out_degree>=threshold) & (hubs_plot.iloc[index, :].in_degree<threshold)):
            hub_type.append('out_hub')
        if((hubs_plot.iloc[index, :].in_degree>=threshold) & (hubs_plot.iloc[index, :].out_degree>=threshold)):
            hub_type.append('in_out_hub')
        if((hubs_plot.iloc[index, :].in_degree<threshold) & (hubs_plot.iloc[index, :].out_degree<threshold)):
            hub_type.append('non-hub')

    hubs_plot['type']=hub_type
    #hubs_plot['count']=np.log(hubs_plot['count'])

    fig, ax = plt.subplots(1,1, figsize=(2,2))
    sns.scatterplot(data=hubs_plot, x='in_degree', y='out_degree', hue='type', size='count', ax=ax, 
                    sizes=(0.3, 8), edgecolor='none', alpha=0.8)
    ax.set(ylim=(-3, 100), xlim=(-3, 100))
    ax.axvline(x=19.5, color='grey', linewidth=0.25, alpha=0.5)
    ax.axhline(y=19.5, color='grey', linewidth=0.25, alpha=0.5)
    ax.legend().set_visible(False)
    plt.savefig(f'network_analysis/plots/hubs_{adj_names[i]}.pdf', format='pdf', bbox_inches='tight')

    fig, ax = plt.subplots(1,1, figsize=(2,2))
    sns.scatterplot(data=hubs_plot, x='in_degree', y='out_degree', hue='type', size='count', ax=ax, 
                    sizes=(0.3, 8), edgecolor='none', alpha=0.8)
    ax.set(ylim=(-3, 100), xlim=(-3, 100))
    ax.axvline(x=19.5, color='grey', linewidth=0.25, alpha=0.5)
    ax.axhline(y=19.5, color='grey', linewidth=0.25, alpha=0.5)
    plt.savefig(f'network_analysis/plots/hubs_{adj_names[i]}_legend.pdf', format='pdf', bbox_inches='tight')

# %%
# determine pairwise hubs

def pairwise_hubs(hubs_df):

    hubs_df = pm.Promat.convert_df_to_pairwise(hubs_df)

    pair_ids = [x[0] for x in hubs_df.loc[('pairs'), :].index]
    pair_ids = list(np.unique(pair_ids))

    for pair in pair_ids:
        test = hubs_df.loc[('pairs', pair), :]
        if(test.iloc[0, :].type==test.iloc[1, :].type):
            continue
        else:
            print(f'recalculating hubs: pair {pair}')
            if((np.mean(test.in_degree)>=threshold) & (np.mean(test.out_degree)<threshold)):
                hubs_df.loc[('pairs', pair), 'type'] = ['in_hub', 'in_hub']
            if((np.mean(test.in_degree)<threshold) & (np.mean(test.out_degree)>=threshold)):
                hubs_df.loc[('pairs', pair), 'type'] = ['out_hub', 'out_hub']
            if((np.mean(test.in_degree)>=threshold) & (np.mean(test.out_degree)>=threshold)):
                hubs_df.loc[('pairs', pair), 'type'] = ['in_out_hub', 'in_out_hub']
            if((np.mean(test.in_degree)<threshold) & (np.mean(test.out_degree)<threshold)):
                hubs_df.loc[('pairs', pair), 'type'] = ['non-hub', 'non-hub']
            print(f'Result is: {hubs_df.loc[(slice(None), pair), :].type.iloc[0]}') 
            print(f'based on in_degrees:{list(hubs_df.loc[(slice(None), pair), :].in_degree.values)}')
            print(f'         out_degrees: {list(hubs_df.loc[(slice(None), pair), :].out_degree.values)}\n')

    return(hubs_df)

G_hubs = [pairwise_hubs(G_hub) for G_hub in G_hubs]
'''
# export hubs
for i, hubs in enumerate(G_hubs):

    if('in_hub' in hubs.type.values):
        in_hubs = hubs.reset_index().groupby(['type', 'skid']).count().loc[('in_hub', slice(None))].index
        pymaid.add_annotations(in_hubs.values, f'mw {adj_names[i]} hubs_in')
        pymaid.add_meta_annotations(f'mw {adj_names[i]} hubs_in', 'mw hubs')

    if('out_hub' in hubs.type.values):
        out_hubs = hubs.reset_index().groupby(['type', 'skid']).count().loc[('out_hub', slice(None))].index
        pymaid.add_annotations(out_hubs.values, f'mw {adj_names[i]} hubs_out')
        pymaid.add_meta_annotations(f'mw {adj_names[i]} hubs_out', 'mw hubs')

    if('in_out_hub' in hubs.type.values):
        in_out_hubs = hubs.reset_index().groupby(['type', 'skid']).count().loc[('in_out_hub', slice(None))].index
        pymaid.add_annotations(in_out_hubs.values, f'mw {adj_names[i]} hubs_in_out')
        pymaid.add_meta_annotations(f'mw {adj_names[i]} hubs_in_out', 'mw hubs')
'''
# %%
# cell type identification
# ad hubs
celltypes_data, celltypes = ct.Celltype_Analyzer.default_celltypes()
orange = sns.color_palette()[1]
green = sns.color_palette()[2]
red = sns.color_palette()[3]

# plot all hub-types at once
for adj_name in adj_names:
    try: in_hubs_ct = ct.Celltype('Out Hubs', pymaid.get_skids_by_annotation(f'mw {adj_name} hubs_out'), orange)
    except: in_hubs_ct = ct.Celltype('Out Hubs', [], orange)

    try: out_hubs_ct = ct.Celltype('In Hubs', pymaid.get_skids_by_annotation(f'mw {adj_name} hubs_in'), green)
    except: out_hubs_ct = ct.Celltype('In Hubs', [], green)

    try: in_out_hubs_ct = ct.Celltype('In-Out Hubs', pymaid.get_skids_by_annotation(f'mw {adj_name} hubs_in_out'), red)
    except: in_out_hubs_ct = ct.Celltype('In-Out Hubs', [], red)

    hubs = ct.Celltype_Analyzer([in_hubs_ct, in_out_hubs_ct, out_hubs_ct])
    hubs.set_known_types(celltypes)
    hubs.plot_memberships(f'network_analysis/plots/{adj_name}_hubs_celltypes.pdf', (0.67*len(hubs.Celltypes),2), ylim=(0,1))

# %%
# location in cluster structure

cluster_level = 6
size = (2,0.5)

for adj_name in adj_names:
    try: ct.plot_marginal_cell_type_cluster(size, ct.Celltype(f'{adj_name} Out Hubs', pymaid.get_skids_by_annotation(f'mw {adj_name} hubs_out')), orange, cluster_level, f'network_analysis/plots/{adj_name}-out-hubs_celltypes-clusters{cluster_level}.pdf', all_celltypes = celltypes)
    except: print('no annotation')

    try: ct.plot_marginal_cell_type_cluster(size, ct.Celltype(f'{adj_name} In Hubs', pymaid.get_skids_by_annotation(f'mw {adj_name} hubs_in')), green, cluster_level, f'network_analysis/plots/{adj_name}-in-hubs_celltypes-clusters{cluster_level}.pdf', all_celltypes = celltypes)
    except: print('no annotation')
        
    try: ct.plot_marginal_cell_type_cluster(size, ct.Celltype(f'{adj_name} In-Out Hubs', pymaid.get_skids_by_annotation(f'mw {adj_name} hubs_in_out')), red, cluster_level, f'network_analysis/plots/{adj_name}-in-out-hubs_celltypes-clusters{cluster_level}.pdf', all_celltypes = celltypes)
    except: print('no annotation')

cluster_level = 3
size = (0.5,0.5)

for adj_name in adj_names:
    try: ct.plot_marginal_cell_type_cluster(size, ct.Celltype(f'{adj_name} All Hubs', pymaid.get_skids_by_annotation(f'mw {adj_name} all_hubs')), orange, cluster_level, f'network_analysis/plots/{adj_name}-all-hubs_celltypes-clusters{cluster_level}.pdf', all_celltypes = celltypes)
    except: print('no annotation')
    
# %%
# plot some examples of hubs
import math 
import navis

ad_out = 3157405
ad_in_out = 9423829
ad_in = 7388340

aa_out = 13988443
aa_in_out = 16561284
aa_in = 12299621

dd_out = 18500988
dd_out_2 = 8311264

da_out = 8311264

neurons_to_plot = [ad_out, ad_in_out, ad_in, 
                    aa_out, aa_in_out, aa_in,
                    dd_out, dd_out_2,
                    da_out]

neuropil = pymaid.get_volume('PS_Neuropil_manual')
neuropil.color = (250, 250, 250, .075)
color = '#5D8B90'

# hub types
n_cols = 4
n_rows = math.ceil(len(neurons_to_plot)/n_cols) # round up to determine how many rows there should be
alpha = 1

fig = plt.figure(figsize=(n_cols*2, n_rows*2))
gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=0, hspace=0)
axs = np.empty((n_rows, n_cols), dtype=object)


for i, skid in enumerate(neurons_to_plot):
    neurons = pymaid.get_neurons(skid)

    inds = np.unravel_index(i, shape=(n_rows, n_cols))
    ax = fig.add_subplot(gs[inds], projection="3d")
    axs[inds] = ax
    navis.plot2d(x=[neurons, neuropil], connectors_only=False, color='gray', alpha=0.5, ax=ax)
    navis.plot2d(x=[neurons, neuropil], connectors_only=True, color=color, alpha=alpha, ax=ax)
    ax.azim = -90
    ax.elev = -90
    ax.dist = 6
    ax.set_xlim3d((-4500, 110000))
    ax.set_ylim3d((-4500, 110000))

fig.savefig(f'network_analysis/plots/morpho_hubs.png', format='png', dpi=300, transparent=True)
