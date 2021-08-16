#%%
import os
os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import connectome_tools.cluster_analysis as clust
import connectome_tools.celltype as ct
import connectome_tools.process_graph as pg
import connectome_tools.process_matrix as pm
import connectome_tools.cascade_analysis as casc
import navis

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

ad_edges = pd.read_csv('data/edges_threshold/ad_all-paired-edges.csv', index_col=0)
pairs = pm.Promat.get_pairs()
_, celltypes = ct.Celltype_Analyzer.default_celltypes()

# %%
# asymmetrical bilateral types

p_asymmetrical_bi_pairids = pm.Promat.load_pairs_from_annotation('mw bilateral axon partially asymmetrical neurons', pairs, return_type='all_pair_ids')
asymmetrical_bi_pairids = pm.Promat.load_pairs_from_annotation('mw bilateral axon asymmetrical neurons', pairs, return_type='all_pair_ids')

# sort by sort-walk
p_asymmetrical_bi_pairids = pm.Promat.walk_sort_skids(p_asymmetrical_bi_pairids, pairs)
asymmetrical_bi_pairids = pm.Promat.walk_sort_skids(asymmetrical_bi_pairids, pairs)

# plot asymmetrical bilaterals
ct.chromosome_plot(
                    df = pm.Promat.find_all_partners_hemispheres(asymmetrical_bi_pairids, ad_edges),
                    path = 'interhemisphere/plots/asymmetrical-bilateral-axon_chromosome-plots',
                    celltypes = celltypes
                )

ct.chromosome_plot(
                    df = pm.Promat.find_all_partners_hemispheres(p_asymmetrical_bi_pairids, ad_edges),
                    path = 'interhemisphere/plots/paritally-asymmetrical-bilateral-axon_chromosome-plots',
                    celltypes = celltypes
                )

# plot upstreams in a simpler way and combine complex downstream and simple upstream manually
ct.chromosome_plot(
                    df = pm.Promat.find_all_partners(asymmetrical_bi_pairids, ad_edges),
                    path = 'interhemisphere/plots/asymmetrical-bilateral-axon_chromosome-plots_simple',
                    celltypes = celltypes,
                    simple=True,
                    spacer_num=2
                )

ct.chromosome_plot(
                    df = pm.Promat.find_all_partners(p_asymmetrical_bi_pairids, ad_edges),
                    path = 'interhemisphere/plots/paritally-asymmetrical-bilateral-axon_chromosome-plots_simple',
                    celltypes = celltypes,
                    simple=True,
                    spacer_num=2
                )

# %%
# minor ipsi/bi/contra cell types
 
ipsi_bi = list(np.intersect1d(pymaid.get_skids_by_annotation('mw ipsilateral axon'), pymaid.get_skids_by_annotation('mw bilateral dendrite')))
bi_bi = list(np.intersect1d(pymaid.get_skids_by_annotation('mw bilateral axon'), pymaid.get_skids_by_annotation('mw bilateral dendrite')))
contra_bi = list(np.intersect1d(pymaid.get_skids_by_annotation('mw contralateral axon'), pymaid.get_skids_by_annotation('mw bilateral dendrite')))
contra_contra = list(np.intersect1d(pymaid.get_skids_by_annotation('mw contralateral axon'), pymaid.get_skids_by_annotation('mw contralateral dendrite')))
contra_contra = np.intersect1d(contra_contra, pymaid.get_skids_by_annotation('mw brain neurons'))

ipsi_bi_pairids = pm.Promat.load_pairs_from_annotation('ipsi_bi', pairs, return_type='all_pair_ids', skids=ipsi_bi, use_skids=True)
bi_bi_pairids = pm.Promat.load_pairs_from_annotation('ipsi_bi', pairs, return_type='all_pair_ids', skids=bi_bi, use_skids=True)
contra_bi_pairids = pm.Promat.load_pairs_from_annotation('ipsi_bi', pairs, return_type='all_pair_ids', skids=contra_bi, use_skids=True)
contra_contra_pairids = pm.Promat.load_pairs_from_annotation('contra_contra', pairs, return_type='all_pair_ids', skids=contra_contra, use_skids=True)

# sort by sort-walk
ipsi_bi_pairids = pm.Promat.walk_sort_skids(ipsi_bi_pairids, pairs)
bi_bi_pairids = pm.Promat.walk_sort_skids(bi_bi_pairids, pairs)
contra_bi_pairids = pm.Promat.walk_sort_skids(contra_bi_pairids, pairs)
contra_contra_pairids = pm.Promat.walk_sort_skids(contra_contra_pairids, pairs)

# plot minor partners
ct.chromosome_plot(
                    df = pm.Promat.find_all_partners_hemispheres(contra_bi_pairids, ad_edges),
                    path = 'interhemisphere/plots/contra-bi_chromosome-plots',
                    celltypes = celltypes
                )

ct.chromosome_plot(
                    df = pm.Promat.find_all_partners_hemispheres(bi_bi_pairids, ad_edges),
                    path = 'interhemisphere/plots/bi-bi_chromosome-plots',
                    celltypes = celltypes
                )

ct.chromosome_plot(
                    df = pm.Promat.find_all_partners_hemispheres(ipsi_bi_pairids, ad_edges),
                    path = 'interhemisphere/plots/ipsi-bi_chromosome-plots',
                    celltypes = celltypes
                )

ct.chromosome_plot(
                    df = pm.Promat.find_all_partners(contra_contra_pairids, ad_edges),
                    path = 'interhemisphere/plots/contra-contra_chromosome-plots',
                    celltypes = celltypes,
                    simple=True
                )
# %%
