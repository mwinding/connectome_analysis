#%%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
import pymaid
from contools import Celltype, Celltype_Analyzer, Promat, Analyze_Nx_G, Analyze_Cluster
from data_settings import data_date

import networkx as nx

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
edge_lists = [Promat.pull_edges(type_edges=name, data_date=data_date, pairs_combined=False, threshold=0.01) for name in adj_names]
Gad, Gaa, Gdd, Gda = [Analyze_Nx_G(edge_list, graph_type='directed', split_pairs=True) for edge_list in edge_lists]
Gs = [Gad, Gaa, Gdd, Gda]


# %%
# extract in-degree / out-degree data for each neuron and identify hubs
threshold = 20
G_hubs = [obj.get_node_degrees(hub_threshold=threshold) for obj in Gs]

# in/out-degree of MB-FBNs
FBN = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain MB-FBNs')
print(np.mean(G_hubs[0]))
print(np.mean(G_hubs[0].loc[FBN]))

# %%
# compare hubs to unfiltered hubs
# this chunk is for a comment received during editing

Gad_unfiltered = nx.readwrite.graphml.read_graphml('data/graphs/Gad.graphml', node_type=int)
Gad_unfiltered = pg.Analyze_Nx_G(edges=[], graph_type='directed', graph=Gad_unfiltered)

Gaa_unfiltered = nx.readwrite.graphml.read_graphml('data/graphs/Gaa.graphml', node_type=int)
Gaa_unfiltered = pg.Analyze_Nx_G(edges=[], graph_type='directed', graph=Gaa_unfiltered)

Gdd_unfiltered = nx.readwrite.graphml.read_graphml('data/graphs/Gdd.graphml', node_type=int)
Gdd_unfiltered = pg.Analyze_Nx_G(edges=[], graph_type='directed', graph=Gdd_unfiltered)

Gda_unfiltered = nx.readwrite.graphml.read_graphml('data/graphs/Gda.graphml', node_type=int)
Gda_unfiltered = pg.Analyze_Nx_G(edges=[], graph_type='directed', graph=Gda_unfiltered)

all_degrees = Gad_unfiltered.get_node_degrees()
threshold = np.round(all_degrees.mean()+1.5*all_degrees.std().mean())[0]

Gs_unfiltered = [Gad_unfiltered, Gaa_unfiltered, Gdd_unfiltered, Gda_unfiltered]
G_hubs_unfiltered = [obj.get_node_degrees(hub_threshold=threshold) for obj in Gs_unfiltered]

skids_unfiltered = [list(hub[hub.type!='non-hub'].index) for hub in G_hubs_unfiltered]
skids_filtered = [list(hub[hub.type!='non-hub'].index) for hub in G_hubs]

skids = skids_unfiltered + skids_filtered

names = ['unfiltered_ad-hub', 'unfiltered_aa-hub', 'unfiltered_dd-hub', 'unfiltered_da-hub',
            'filtered_ad-hub', 'filtered_aa-hub', 'filtered_dd-hub', 'filtered_da-hub']

hub_ctas = [Celltype(name=names[i], skids=skids[i]) for i in range(len(skids))]
hub_ctas = Celltype_Analyzer(hub_ctas)
hub_ctas.compare_membership(sim_type='cosine')

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

print(np.mean(G_hubs[0].groupby('pair_id').mean().out_degree)+1.5*np.std(G_hubs[0].groupby('pair_id').mean().out_degree))
print(np.mean(G_hubs[0].groupby('pair_id').mean().in_degree)+1.5*np.std(G_hubs[0].groupby('pair_id').mean().in_degree))
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
celltypes_data, celltypes = Celltype_Analyzer.default_celltypes()
blue = sns.color_palette()[0]
orange = sns.color_palette()[1]
green = sns.color_palette()[2]
red = sns.color_palette()[3]

# plot all hub-types at once
for adj_name in adj_names:
    try: in_hubs_ct = Celltype('Out Hubs', pymaid.get_skids_by_annotation(f'mw {adj_name} hubs_out'), orange)
    except: in_hubs_ct = Celltype('Out Hubs', [], orange)

    try: out_hubs_ct = Celltype('In Hubs', pymaid.get_skids_by_annotation(f'mw {adj_name} hubs_in'), green)
    except: out_hubs_ct = Celltype('In Hubs', [], green)

    try: in_out_hubs_ct = Celltype('In-Out Hubs', pymaid.get_skids_by_annotation(f'mw {adj_name} hubs_in_out'), red)
    except: in_out_hubs_ct = Celltype('In-Out Hubs', [], red)

    hubs = Celltype_Analyzer([in_hubs_ct, in_out_hubs_ct, out_hubs_ct])
    hubs.set_known_types(celltypes)
    hubs.plot_memberships(f'network_analysis/plots/{adj_name}_hubs_celltypes.pdf', (0.67*len(hubs.Celltypes),2), ylim=(0,1))


ad_hubs_ct = Celltype('a-d', pymaid.get_skids_by_annotation(f'mw ad all_hubs'), blue)
aa_hubs_ct = Celltype('a-a', pymaid.get_skids_by_annotation(f'mw aa all_hubs'), orange)
dd_hubs_ct = Celltype('d-d', pymaid.get_skids_by_annotation(f'mw dd all_hubs'), green)
da_hubs_ct = Celltype('d-a', pymaid.get_skids_by_annotation(f'mw da all_hubs'), red)

hubs = Celltype_Analyzer([ad_hubs_ct, aa_hubs_ct, dd_hubs_ct, da_hubs_ct])
hubs.set_known_types(celltypes)
hubs.plot_memberships(f'network_analysis/plots/all-hubs_celltypes.pdf', (0.67*len(hubs.Celltypes),2), ylim=(0,1))
hubs.plot_memberships(f'network_analysis/plots/all-hubs_celltypes_raw.pdf', (0.67*len(hubs.Celltypes),2), ylim=(0,525), raw_num=True)


# %%
# location in cluster structure

def plot_marginal_cell_type_cluster(size, particular_cell_type, particular_color, cluster_level, path, all_celltypes=None):

    # all cell types plot data
    if(all_celltypes==None):
        _, all_celltypes = Celltype_Analyzer.default_celltypes()
        
    clusters = Celltype_Analyzer.get_skids_from_meta_annotation(f'mw brain clusters level {cluster_level}', split=True, return_celltypes=True)
    cluster_analyze = Celltype_Analyzer(clusters)

    cluster_analyze.set_known_types(all_celltypes)
    celltype_colors = [x.get_color() for x in cluster_analyze.get_known_types()]
    all_memberships = cluster_analyze.memberships()
    all_memberships = all_memberships.iloc[[0,1,2,3,4,5,6,7,8,9,10,11,12,17,13,14,15,16], :] # switching order so unknown is not above outputs and RGNs before pre-outputs
    celltype_colors = [celltype_colors[i] for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,17,13,14,15,16]] # switching order so unknown is not above outputs and RGNs before pre-outputs
    
    # particular cell type data
    cluster_analyze.set_known_types([particular_cell_type])
    membership = cluster_analyze.memberships()

    # plot
    fig = plt.figure(figsize=size) 
    fig.subplots_adjust(hspace=0.1)
    gs = plt.GridSpec(4, 1)

    ax = fig.add_subplot(gs[0:3, 0])
    ind = np.arange(0, len(cluster_analyze.Celltypes))
    ax.bar(ind, membership.iloc[0, :], color=particular_color)
    ax.set(xlim = (-1, len(ind)), ylim=(0,1), xticks=([]), yticks=([]), title=particular_cell_type.get_name())

    ax = fig.add_subplot(gs[3, 0])
    ind = np.arange(0, len(cluster_analyze.Celltypes))
    ax.bar(ind, all_memberships.iloc[0, :], color=celltype_colors[0])
    bottom = all_memberships.iloc[0, :]
    for i in range(1, len(all_memberships.index)):
        plt.bar(ind, all_memberships.iloc[i, :], bottom = bottom, color=celltype_colors[i])
        bottom = bottom + all_memberships.iloc[i, :]
    ax.set(xlim = (-1, len(ind)), ylim=(0,1), xticks=([]), yticks=([]))
    ax.axis('off')
    ax.axis('off')

    plt.savefig(path, format='pdf', bbox_inches='tight')

cluster_level = 7
size = (2,0.5)
adj_names = ['ad', 'aa', 'dd', 'da']
_, celltypes = Celltype_Analyzer.default_celltypes()

blue = sns.color_palette()[0]
orange = sns.color_palette()[1]
green = sns.color_palette()[2]
red = sns.color_palette()[3]
colors = [blue, orange, green, red]

# plot ad, aa, dd, da hubs (each type: in, out, in/out) within cluster level 7
for adj_name in adj_names:
    try: Celltype_Analyzer.plot_marginal_cell_type_cluster(size, Celltype(f'{adj_name} Out Hubs', pymaid.get_skids_by_annotation(f'mw {adj_name} hubs_out')), orange, cluster_level, f'plots/network-analysis_{adj_name}-out-hubs_celltypes-clusters{cluster_level}.pdf', all_celltypes = celltypes)
    except: print('no annotation')

    try: Celltype_Analyzer.plot_marginal_cell_type_cluster(size, Celltype(f'{adj_name} In Hubs', pymaid.get_skids_by_annotation(f'mw {adj_name} hubs_in')), green, cluster_level, f'plots/network-analysis_{adj_name}-in-hubs_celltypes-clusters{cluster_level}.pdf', all_celltypes = celltypes)
    except: print('no annotation')
        
    try: Celltype_Analyzer.plot_marginal_cell_type_cluster(size, Celltype(f'{adj_name} In-Out Hubs', pymaid.get_skids_by_annotation(f'mw {adj_name} hubs_in_out')), red, cluster_level, f'plots/network-analysis_{adj_name}-in-out-hubs_celltypes-clusters{cluster_level}.pdf', all_celltypes = celltypes)
    except: print('no annotation')

# plot all ad, aa, dd, and da hubs within cluster level 4
cluster_level = 4
size = (0.5,0.5)

for i, adj_name in enumerate(adj_names):
    plot_marginal_cell_type_cluster(size, Celltype(f'{adj_name} All Hubs', pymaid.get_skids_by_annotation(f'mw {adj_name} all_hubs')), colors[i], cluster_level, f'plots/network-analysis_{adj_name}-all-hubs_celltypes-clusters{cluster_level}.pdf', all_celltypes = celltypes)

# plot a-d hubs (in, out, in/out hubs) within cluster level 4
cluster_level = 4
size = (0.5,0.5)
adj_name = 'ad'

plot_marginal_cell_type_cluster(size, Celltype(f'{adj_name} Out Hubs', pymaid.get_skids_by_annotation(f'mw {adj_name} hubs_out')), orange, cluster_level, f'plots/network-analysis_{adj_name}-out-hubs_celltypes-clusters{cluster_level}.pdf', all_celltypes = celltypes)
plot_marginal_cell_type_cluster(size, Celltype(f'{adj_name} In Hubs', pymaid.get_skids_by_annotation(f'mw {adj_name} hubs_in')), green, cluster_level, f'plots/network-analysis_{adj_name}-in-hubs_celltypes-clusters{cluster_level}.pdf', all_celltypes = celltypes)
plot_marginal_cell_type_cluster(size, Celltype(f'{adj_name} In-Out Hubs', pymaid.get_skids_by_annotation(f'mw {adj_name} hubs_in_out')), red, cluster_level, f'plots/network-analysis_{adj_name}-in-out-hubs_celltypes-clusters{cluster_level}.pdf', all_celltypes = celltypes)

# %%
# plot cell type memberships of ad hubs
_, celltypes = Celltype_Analyzer.default_celltypes()

ad_hubs = [Celltype('ad out hubs', pymaid.get_skids_by_annotation('mw ad hubs_out')),
            Celltype('ad in-out hubs', pymaid.get_skids_by_annotation('mw ad hubs_in_out')),
            Celltype('ad in hubs', pymaid.get_skids_by_annotation('mw ad hubs_in'))]

celltype_annots = pymaid.get_annotated('mw brain simple groups').name
celltype_skids = [Celltype_Analyzer.get_skids_from_meta_annotation(annot) for annot in celltype_annots]
celltype_names = [x.replace('mw brain ', '') for x in celltype_annots.values]
celltypes_full = [Celltype(celltype_names[i], celltype_skids[i]) for i in range(len(celltype_names))]

ad_hubs_cta = Celltype_Analyzer(ad_hubs)
ad_hubs_cta.set_known_types(celltypes_full)
ad_hubs_cta.memberships()

ad_hubs_cta.set_known_types(celltypes)
ad_hubs_cta.memberships()

official_order = ['sensories', 'ascendings', 'PNs', 'PNs-somato', 'LNs', 'LHNs', 'FFNs', 'MBINs', 'KCs', 'MBONs', 'MB-FBNs', 'CNs', 'pre-dSEZs', 'pre-dVNCs', 'RGNs', 'dSEZs', 'dVNCs', 'unknown']
bar_df = ad_hubs_cta.memberships(raw_num=True).iloc[:, 1].loc[official_order]

# gave priority to other celltypes over LNs, performed in CATMAID
# *** WARNING: hardcoded ***
bar_df.loc['LNs'] = 0
bar_df.loc['MB-FBNs'] = bar_df.loc['MB-FBNs'] + 4
bar_df.loc['LHNs'] = bar_df.loc['LHNs'] + 2

# switch priority of CNs over LHNs and MBONs, new class of CN/FBNs
# will use pre-dSEZs to mean CNs and CNs to mean CN/MB-FBNs, so that the order works out
# will need to manually change the colors
bar_df['pre-dSEZs'] = bar_df['pre-dSEZs'] + 4
bar_df['LHNs'] = bar_df['LHNs'] - 4
bar_df['pre-dSEZs'] = bar_df['pre-dSEZs'] + 2
bar_df['MBONs'] = bar_df['MBONs'] - 2
bar_df['CNs'] = bar_df['CNs'] + 6
bar_df['MB-FBNs'] = bar_df['MB-FBNs'] - 6

# pull official celltype colors
colors = list(pymaid.get_annotated('mw brain simple colors').name)
colors_names = [x.name.values[0] for x in list(map(pymaid.get_annotated, colors))] # use order of colors annotation for now
color_sort = [np.where(x.replace('mw brain ', '')==np.array(official_order))[0][0] for x in colors_names]
colors = [element for _, element in sorted(zip(color_sort, colors))]
colors = colors + ['tab:gray']

# donut plot of cell types
fig, ax = plt.subplots(1,1,figsize=(2,2))
ax.pie(bar_df, colors=colors)

my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

plt.savefig('plots/hubs_ad-in-out_celltypes.pdf', format='pdf', bbox_inches='tight')

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

# %%
# targeted questions about MB neurons

# how many a-d in-out hubs are downstream of MB or upstream of DANs (FBNs, FFNs, MBONs)
ad_inout_hub = pymaid.get_skids_by_annotation('mw ad hubs_in_out')
MBONs = pymaid.get_skids_by_annotation('mw MBON')
FBNs = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain MB-FBNs')
FFNs = pymaid.get_skids_by_annotation('mw FFN')
MB_types = list(np.unique(MBONs + FBNs + FFNs))

fraction_hubs = len(np.intersect1d(MB_types, ad_inout_hub))/len(ad_inout_hub)
print(f'{fraction_hubs:.2f} of a-d in-out hubs were MBONs, MB-FBNs, or MB-FFNs')

# how many MB-FBNs, MB-FFNs, MBONs are in-out hubs?
ad_inout_hub = pymaid.get_skids_by_annotation('mw ad hubs_in_out')
fraction_MBON_hubs = len(np.intersect1d(MBONs, ad_inout_hub))/len(MBONs)
fraction_FBN_hubs = len(np.unique(np.intersect1d(FBNs, ad_inout_hub)))/len(FBNs)
fraction_FFN_hubs = len(np.intersect1d(FFNs, ad_inout_hub))/len(FFNs)
print(f'{fraction_MBON_hubs:.2f} of MBONs are a-d in-out hubs')
print(f'{fraction_FBN_hubs:.2f} of MB-FBNs are a-d in-out hubs')
print(f'{fraction_FFN_hubs:.2f} of MB-FFNs are a-d in-out hubs')

######
# how does in-degree/out-degree of MB-FBNs compare to other in-out hubs?
FBN_pairids = pm.Promat.load_pairs_from_annotation('MB-FBN', pm.Promat.get_pairs(), return_type='all_pair_ids', skids=FBNs, use_skids=True)
ad_hubs = G_hubs[0]

# identify FBN and non-FBN ad hubs
FBN_hubs = ad_hubs.loc[('pairs', FBN_pairids, FBN_pairids)]
non_FBNs = np.setdiff1d([x[1] for x in ad_hubs.index], FBN_pairids)
other_hubs = ad_hubs.loc[('pairs', non_FBNs, non_FBNs)]

# identify FBN and non-FBN ad in-out hubs
FBN_hubs_inout = FBN_hubs[FBN_hubs.type=='in_out_hub'].drop(['in_hub', 'out_hub', 'in_out_hub'], axis=1)
other_hubs_inout = other_hubs[other_hubs.type=='in_out_hub'].drop(['in_hub', 'out_hub', 'in_out_hub'], axis=1)
FBN_hubs_inout['type'] = ['FBN_in-out-hub']*len(FBN_hubs_inout.index)
other_hubs_inout['type'] = ['nonFBN_in-out-hub']*len(other_hubs_inout.index)

df = pd.concat([FBN_hubs_inout, other_hubs_inout])

fig, ax = plt.subplots(1,1, figsize=(4,4))
sns.scatterplot(x=df.in_degree, y=df.out_degree, hue=df.type, ax=ax)
plt.savefig('network_analysis/plots/in-out-degree_FBNs.pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1, figsize=(4,4))
sns.barplot(y=df.in_degree, x=df.type, ax=ax)
ax.set(ylim = (0,50))
plt.savefig('network_analysis/plots/in-degree_FBNs.pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1, figsize=(4,4))
sns.barplot(y=df.out_degree, x=df.type, ax=ax)
ax.set(ylim = (0,50))
plt.savefig('network_analysis/plots/out-degree_FBNs.pdf', bbox_inches='tight')

print(f'FBN in-degree: {np.mean(FBN_hubs_inout.in_degree):.2f} +/- {np.std(FBN_hubs_inout.in_degree):.2f}')
print(f'Non-FBN in-degree: {np.mean(other_hubs_inout.in_degree):.2f} +/- {np.std(other_hubs_inout.in_degree):.2f}')
print(f'FBN out-degree: {np.mean(FBN_hubs_inout.out_degree):.2f} +/- {np.std(FBN_hubs_inout.out_degree):.2f}')
print(f'Non-FBN out-degree: {np.mean(other_hubs_inout.out_degree):.2f} +/- {np.std(other_hubs_inout.out_degree):.2f}')

#####
# how does in-degree/out-degree of MB neurons compare to other in-out hubs?
MB_pairids = pm.Promat.load_pairs_from_annotation('MB-types', pm.Promat.get_pairs(), return_type='all_pair_ids', skids=MB_types, use_skids=True)
ad_hubs = G_hubs[0]

# identify FBN and non-FBN ad hubs
MB_hubs = ad_hubs.loc[('pairs', MB_pairids, MB_pairids)]
non_MB = np.setdiff1d([x[1] for x in ad_hubs.index], MB_pairids)
other_hubs = ad_hubs.loc[('pairs', non_MB, non_MB)]

# identify FBN and non-FBN ad in-out hubs
MB_hubs_inout = MB_hubs[MB_hubs.type=='in_out_hub'].drop(['in_hub', 'out_hub', 'in_out_hub'], axis=1)
other_hubs_inout = other_hubs[other_hubs.type=='in_out_hub'].drop(['in_hub', 'out_hub', 'in_out_hub'], axis=1)
MB_hubs_inout['type'] = ['MB_in-out-hub']*len(MB_hubs_inout.index)
other_hubs_inout['type'] = ['nonMB_in-out-hub']*len(other_hubs_inout.index)

df = pd.concat([MB_hubs_inout, other_hubs_inout])

fig, ax = plt.subplots(1,1, figsize=(4,4))
sns.scatterplot(x=df.in_degree, y=df.out_degree, hue=df.type, ax=ax)
plt.savefig('network_analysis/plots/in-out-degree_MB-types.pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1, figsize=(4,4))
sns.barplot(y=df.in_degree, x=df.type, ax=ax)
ax.set(ylim = (0,50))
plt.savefig('network_analysis/plots/in-degree_MB-types.pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1, figsize=(4,4))
sns.barplot(y=df.out_degree, x=df.type, ax=ax)
ax.set(ylim = (0,50))
plt.savefig('network_analysis/plots/out-degree_MB-types.pdf', bbox_inches='tight')

print(f'MB in-degree: {np.mean(MB_hubs_inout.in_degree):.2f} +/- {np.std(MB_hubs_inout.in_degree):.2f}')
print(f'Non-MB in-degree: {np.mean(other_hubs_inout.in_degree):.2f} +/- {np.std(other_hubs_inout.in_degree):.2f}')
print(f'MB out-degree: {np.mean(MB_hubs_inout.out_degree):.2f} +/- {np.std(MB_hubs_inout.out_degree):.2f}')
print(f'Non-MB out-degree: {np.mean(other_hubs_inout.out_degree):.2f} +/- {np.std(other_hubs_inout.out_degree):.2f}')