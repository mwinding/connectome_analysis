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
                    sizes=(0.3, 10), edgecolor='none', alpha=0.9)
    ax.set(ylim=(-3, 100), xlim=(-3, 100))
    ax.axvline(x=19.5, color='grey', linewidth=0.25, alpha=0.5)
    ax.axhline(y=19.5, color='grey', linewidth=0.25, alpha=0.5)
    ax.legend().set_visible(False)
    plt.savefig(f'network_analysis/plots/hubs_{adj_names[i]}.pdf', format='pdf', bbox_inches='tight')

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

# export hubs
'''
for i, hubs in enumerate(G_hubs):

    if('in_hub' in hubs.type.values):
        in_hubs = hubs.reset_index().groupby(['type', 'skid']).count().loc[('in_hub', slice(None))].index
        pymaid.add_annotations(in_hubs.values, f'mw {adj_names[i]} hubs_in')

    if('out_hub' in hubs.type.values):
        out_hubs = hubs.reset_index().groupby(['type', 'skid']).count().loc[('out_hub', slice(None))].index
        pymaid.add_annotations(out_hubs.values, f'mw {adj_names[i]} hubs_out')

    if('in_out_hub' in hubs.type.values):
        in_out_hubs = hubs.reset_index().groupby(['type', 'skid']).count().loc[('in_out_hub', slice(None))].index
        pymaid.add_annotations(in_out_hubs.values, f'mw {adj_names[i]} hubs_in_out')
'''
# %%
# location in cluster structure
# ad hubs
celltypes_data, celltypes = ct.Celltype_Analyzer.default_celltypes()
orange = sns.color_palette()[1]
green = sns.color_palette()[2]
red = sns.color_palette()[3]

# plot all hub-types at once
in_hubs_ct = ct.Celltype('Out Hubs', pymaid.get_skids_by_annotation('mw ad hubs_out'), orange)
out_hubs_ct = ct.Celltype('In Hubs', pymaid.get_skids_by_annotation('mw ad hubs_in'), green)
in_out_hubs_ct = ct.Celltype('In-Out Hubs', pymaid.get_skids_by_annotation('mw ad hubs_in_out'), red)

hubs = ct.Celltype_Analyzer([in_hubs_ct, in_out_hubs_ct, out_hubs_ct])
hubs.set_known_types(celltypes)
hubs.plot_memberships('network_analysis/plots/ad_hubs_celltypes.pdf', (0.67*len(hubs.Celltypes),2))

ct.plot_marginal_cell_type_cluster((2,1), ct.Celltype('A-D Out Hubs', pymaid.get_skids_by_annotation('mw ad hubs_out')), orange, 'lvl7_labels', 'network_analysis/plots/ad-out-hubs_celltypes-clusters.pdf')
ct.plot_marginal_cell_type_cluster((2,1), ct.Celltype('A-D In Hubs', pymaid.get_skids_by_annotation('mw ad hubs_in')), green, 'lvl7_labels', 'network_analysis/plots/ad-in-hubs_celltypes-clusters.pdf')
ct.plot_marginal_cell_type_cluster((2,1), ct.Celltype('A-D In-Out Hubs', pymaid.get_skids_by_annotation('mw ad hubs_in_out')), red, 'lvl7_labels', 'network_analysis/plots/ad-in-out-hubs_celltypes-clusters.pdf')

# %%
# location in cluster structure
# aa hubs

# plot all hub-types at once
in_hubs_ct = ct.Celltype('Out Hubs', pymaid.get_skids_by_annotation('mw aa hubs_out'), orange)
out_hubs_ct = ct.Celltype('In Hubs', pymaid.get_skids_by_annotation('mw aa hubs_in'), green)
in_out_hubs_ct = ct.Celltype('In-Out Hubs', pymaid.get_skids_by_annotation('mw aa hubs_in_out'), red)

hubs = ct.Celltype_Analyzer([in_hubs_ct, in_out_hubs_ct, out_hubs_ct])
hubs.set_known_types(celltypes)
hubs.plot_memberships('network_analysis/plots/aa_hubs_celltypes.pdf', (0.67*len(hubs.Celltypes),2))

ct.plot_marginal_cell_type_cluster((2,1), ct.Celltype('A-A Out Hubs', pymaid.get_skids_by_annotation('mw aa hubs_out')), orange, 'lvl7_labels', 'network_analysis/plots/aa-out-hubs_celltypes-clusters.pdf')
ct.plot_marginal_cell_type_cluster((2,1), ct.Celltype('A-A In Hubs', pymaid.get_skids_by_annotation('mw aa hubs_in')), green, 'lvl7_labels', 'network_analysis/plots/aa-in-hubs_celltypes-clusters.pdf')
ct.plot_marginal_cell_type_cluster((2,1), ct.Celltype('A-A In-Out Hubs', pymaid.get_skids_by_annotation('mw aa hubs_in_out')), red, 'lvl7_labels', 'network_analysis/plots/aa-in-out-hubs_celltypes-clusters.pdf')

# %%
# location in cluster structure
# dd hubs

# plot all hub-types at once
in_hubs_ct = ct.Celltype('Out Hubs', pymaid.get_skids_by_annotation('mw dd hubs_out'), orange)

hubs = ct.Celltype_Analyzer([in_hubs_ct])
hubs.set_known_types(celltypes)
hubs.plot_memberships('network_analysis/plots/dd_hubs_celltypes.pdf', (0.67*len(hubs.Celltypes),2))

ct.plot_marginal_cell_type_cluster((2,1), ct.Celltype('D-D Out Hubs', pymaid.get_skids_by_annotation('mw dd hubs_out')), orange, 'lvl7_labels', 'network_analysis/plots/dd-out-hubs_celltypes-clusters.pdf')

# %%
# location in cluster structure
# da hubs

# plot all hub-types at once
in_hubs_ct = ct.Celltype('Out Hubs', pymaid.get_skids_by_annotation('mw da hubs_out'), orange)
out_hubs_ct = ct.Celltype('In Hubs', pymaid.get_skids_by_annotation('mw da hubs_in'), green)

hubs = ct.Celltype_Analyzer([in_hubs_ct, out_hubs_ct])
hubs.set_known_types(celltypes)
hubs.plot_memberships('network_analysis/plots/da_hubs_celltypes.pdf', (0.67*len(hubs.Celltypes),2))

ct.plot_marginal_cell_type_cluster((2,1), ct.Celltype('D-A Out Hubs', pymaid.get_skids_by_annotation('mw da hubs_out')), orange, 'lvl7_labels', 'network_analysis/plots/da-out-hubs_celltypes-clusters.pdf')
ct.plot_marginal_cell_type_cluster((2,1), ct.Celltype('D-A In Hubs', pymaid.get_skids_by_annotation('mw da hubs_in')), green, 'lvl7_labels', 'network_analysis/plots/da-in-hubs_celltypes-clusters.pdf')

# %%
