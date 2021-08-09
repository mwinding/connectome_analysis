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

# %%
# identify and annotate partner loops

contra = list(np.intersect1d(pymaid.get_skids_by_annotation('mw contralateral axon'), pymaid.get_skids_by_annotation('mw ipsilateral dendrite')))
bi = list(np.intersect1d(pymaid.get_skids_by_annotation('mw bilateral axon'), pymaid.get_skids_by_annotation('mw ipsilateral dendrite')))

exclude = pymaid.get_skids_by_annotation('mw A1 neurons paired') + pymaid.get_skids_by_annotation('gorogoro')
partner_loops = ad_edges[(ad_edges.upstream_pair_id == ad_edges.downstream_pair_id) & (ad_edges.type=='contralateral') & ([True if x not in exclude else False for x in ad_edges.upstream_pair_id])]
partner_loops.reset_index(inplace=True, drop=True)

partner_loop_pairs = pm.Promat.get_paired_skids(list(partner_loops.upstream_pair_id), pairs)

# annotate neurons
#pymaid.add_annotations(list(partner_loop_pairs.leftid) + list(partner_loop_pairs.rightid), 'mw partner loops')

# %%
# location in cluster structure
celltypes_data, celltypes = ct.Celltype_Analyzer.default_celltypes()
loops = pymaid.get_skids_by_annotation('mw partner loops')
contra_loops = list(np.intersect1d(pymaid.get_skids_by_annotation('mw partner loops'), pymaid.get_skids_by_annotation('mw contralateral axon')))
bi_loops = list(np.intersect1d(pymaid.get_skids_by_annotation('mw partner loops'), pymaid.get_skids_by_annotation('mw bilateral axon')))

cluster_level = 7
size = (2,0.5)
blue = sns.color_palette()[0]
orange = sns.color_palette()[1]
green = sns.color_palette()[2]

ct.plot_marginal_cell_type_cluster(size, ct.Celltype('Partner-Loops', loops), blue, cluster_level, 'interhemisphere/plots/partner-loops-in-cluster.pdf', all_celltypes = celltypes)
ct.plot_marginal_cell_type_cluster(size, ct.Celltype('Bilateral-Partner-Loops', bi_loops), orange, cluster_level, 'interhemisphere/plots/bilateral-partner-loops-in-cluster.pdf', all_celltypes = celltypes)
ct.plot_marginal_cell_type_cluster(size, ct.Celltype('Contra-Partner-Loops', contra_loops), green, cluster_level, 'interhemisphere/plots/contra-partner-loops-in-cluster.pdf', all_celltypes = celltypes)

# %%
# cell types of loops

loops_ct = ct.Celltype('Partner-Loops', loops, blue)
bi_loops_ct = ct.Celltype('Bilateral-Partner-Loops', bi_loops, orange)
contra_loops_ct = ct.Celltype('Contra-Partner-Loops', contra_loops, green)

loops_analyze = ct.Celltype_Analyzer([loops_ct, bi_loops_ct, contra_loops_ct])
loops_analyze.set_known_types(celltypes)
loops_analyze.plot_memberships(f'interhemisphere/plots/partner-loops_celltypes.pdf', (0.67*len(loops_analyze.Celltypes),2), ylim=(0,1))
loops_analyze.plot_memberships(f'interhemisphere/plots/partner-loops_celltypes_count.pdf', (0.67*len(loops_analyze.Celltypes),2), raw_num=True)

# %%
# upstream and downstream of partner loops

contra_loops = list(np.intersect1d(pymaid.get_skids_by_annotation('mw partner loops'), pymaid.get_skids_by_annotation('mw contralateral axon')))
bi_loops = list(np.intersect1d(pymaid.get_skids_by_annotation('mw partner loops'), pymaid.get_skids_by_annotation('mw bilateral axon')))

contra_loops_pairs = pm.Promat.load_pairs_from_annotation(annot='contralateral loops', pairList=pairs, skids=contra_loops, use_skids=True)
bi_loops_pairs = pm.Promat.load_pairs_from_annotation(annot='bilateral loops', pairList=pairs, skids=bi_loops, use_skids=True)

contra_loop_partners = pm.Promat.find_all_partners(contra_loops_pairs.leftid, ad_edges)
bi_loop_partners = pm.Promat.find_all_partners(bi_loops_pairs.leftid, ad_edges)

# %%
# plot partners
celltypes_data, celltypes = ct.Celltype_Analyzer.default_celltypes()

# all us/ds from contra loops combined
loops = pymaid.get_skids_by_annotation('mw partner loops')
loops_pairs = pm.Promat.load_pairs_from_annotation(annot='loops', pairList=pairs, skids=loops, use_skids=True)
loop_partners = pm.Promat.find_all_partners(loops_pairs.leftid, ad_edges)

all_loops_us = list(np.unique([x for sublist in list(loop_partners.upstream) for x in sublist]))
all_loops_ds = list(np.unique([x for sublist in list(loop_partners.downstream) for x in sublist]))
contra_loops_us = list(np.unique([x for sublist in list(contra_loop_partners.upstream) for x in sublist]))
contra_loops_ds = list(np.unique([x for sublist in list(contra_loop_partners.downstream) for x in sublist]))
bi_loops_us = list(np.unique([x for sublist in list(bi_loop_partners.upstream) for x in sublist]))
bi_loops_ds = list(np.unique([x for sublist in list(bi_loop_partners.downstream) for x in sublist]))

upstream_ct = ct.Celltype_Analyzer(
                    [ct.Celltype('all_loops_us', all_loops_us),
                    ct.Celltype('bi_loops_us', bi_loops_us),
                    ct.Celltype('contra_loops_us', contra_loops_us)]
                )

downstream_ct = ct.Celltype_Analyzer(
                    [ct.Celltype('all_loops_ds', all_loops_ds),
                    ct.Celltype('bi_loops_ds', bi_loops_ds),
                    ct.Celltype('contra_loops_ds', contra_loops_ds)]
                )

upstream_ct.set_known_types(celltypes)
downstream_ct.set_known_types(celltypes)
upstream_ct.plot_memberships(f'interhemisphere/plots/all-partner-loops-upstream_celltypes.pdf', (0.67*len(upstream_ct.Celltypes),2), ylim=(0,1))
downstream_ct.plot_memberships(f'interhemisphere/plots/all-partner-loops-downstream_celltypes.pdf', (0.67*len(downstream_ct.Celltypes),2), ylim=(0,1))

# individual contra loops
contra_loop_upstream_ct = []
contra_loop_downstream_ct = []
for i in contra_loop_partners.index:
    pairid = contra_loop_partners.source_pairid[i]
    us = contra_loop_partners.loc[i, 'upstream']
    ds = contra_loop_partners.loc[i, 'downstream']

    contra_loop_upstream_ct.append(ct.Celltype(f'{pairid}-upstream-l', us))
    contra_loop_upstream_ct.append(ct.Celltype(f'{pairid}-upstream-r', us))
    contra_loop_upstream_ct.append(ct.Celltype(f'{pairid}-spacer', [])) # add these blank columns for formatting purposes only
    contra_loop_upstream_ct.append(ct.Celltype(f'{pairid}-spacer2', [])) # add these blank columns for formatting purposes only

    contra_loop_downstream_ct.append(ct.Celltype(f'{pairid}-downstream-l', ds))
    contra_loop_downstream_ct.append(ct.Celltype(f'{pairid}-downstream-r', ds))
    contra_loop_downstream_ct.append(ct.Celltype(f'{pairid}-spacer', [])) # add these blank columns for formatting purposes only
    contra_loop_downstream_ct.append(ct.Celltype(f'{pairid}-spacer2', [])) # add these blank columns for formatting purposes only

contra_loop_upstream_ct = ct.Celltype_Analyzer(contra_loop_upstream_ct)
contra_loop_downstream_ct = ct.Celltype_Analyzer(contra_loop_downstream_ct)

contra_loop_upstream_ct.set_known_types(celltypes)
contra_loop_downstream_ct.set_known_types(celltypes)
contra_loop_upstream_ct.plot_memberships(f'interhemisphere/plots/contra-partner-loops-upstream_celltypes.pdf', (0.67*len(contra_loop_upstream_ct.Celltypes),2), ylim=(0,1))
contra_loop_downstream_ct.plot_memberships(f'interhemisphere/plots/contra-partner-loops-downstream_celltypes.pdf', (0.67*len(contra_loop_downstream_ct.Celltypes),2), ylim=(0,1))

# individual bilateral loops
bi_loop_upstream_ct = []
bi_loop_downstream_ct = []
for i in bi_loop_partners.index:
    pairid = bi_loop_partners.source_pairid[i]
    us = bi_loop_partners.loc[i, 'upstream']
    ds = bi_loop_partners.loc[i, 'downstream']

    bi_loop_upstream_ct.append(ct.Celltype(f'{pairid}-upstream-l', us))
    bi_loop_upstream_ct.append(ct.Celltype(f'{pairid}-upstream-r', us))
    bi_loop_upstream_ct.append(ct.Celltype(f'{pairid}-spacer', [])) # add these blank columns for formatting purposes only
    bi_loop_upstream_ct.append(ct.Celltype(f'{pairid}-spacer2', [])) # add these blank columns for formatting purposes only

    bi_loop_downstream_ct.append(ct.Celltype(f'{pairid}-downstream-l', ds))
    bi_loop_downstream_ct.append(ct.Celltype(f'{pairid}-downstream-r', ds))
    bi_loop_downstream_ct.append(ct.Celltype(f'{pairid}-spacer', [])) # add these blank columns for formatting purposes only
    bi_loop_downstream_ct.append(ct.Celltype(f'{pairid}-spacer2', [])) # add these blank columns for formatting purposes only

bi_loop_upstream_ct = ct.Celltype_Analyzer(bi_loop_upstream_ct)
bi_loop_downstream_ct = ct.Celltype_Analyzer(bi_loop_downstream_ct)

bi_loop_upstream_ct.set_known_types(celltypes)
bi_loop_downstream_ct.set_known_types(celltypes)
bi_loop_upstream_ct.plot_memberships(f'interhemisphere/plots/bi-partner-loops-upstream_celltypes.pdf', (0.67*len(bi_loop_upstream_ct.Celltypes),2), ylim=(0,1))
bi_loop_downstream_ct.plot_memberships(f'interhemisphere/plots/bi-partner-loops-downstream_celltypes.pdf', (0.67*len(bi_loop_downstream_ct.Celltypes),2), ylim=(0,1))

# %%
# annotate us/ds partners of partner-loop neurons

# export partners to CATMAID for bilateral_diff and bilateral_pdiff
for i in contra_loop_partners.index:
    skid = contra_loop_partners.loc[i].source_pairid
    pymaid.add_annotations(contra_loop_partners.loc[i].upstream, f'mw {skid} upstream partners')
    pymaid.add_annotations(contra_loop_partners.loc[i].downstream, f'mw {skid} downstream partners')

    pymaid.add_meta_annotations(f'mw {skid} upstream partners', 'mw partner-loops-contralateral partners')
    pymaid.add_meta_annotations(f'mw {skid} downstream partners', 'mw partner-loops-contralateral partners')
    pymaid.add_meta_annotations(f'mw {skid} upstream partners', 'mw partner-loops partners')
    pymaid.add_meta_annotations(f'mw {skid} downstream partners', 'mw partner-loops partners')

for i in bi_loop_partners.index:
    skid = bi_loop_partners.loc[i].source_pairid
    pymaid.add_annotations(bi_loop_partners.loc[i].upstream, f'mw {skid} upstream partners')
    pymaid.add_annotations(bi_loop_partners.loc[i].downstream, f'mw {skid} downstream partners')

    pymaid.add_meta_annotations(f'mw {skid} upstream partners', 'mw partner-loops-bilateral partners')
    pymaid.add_meta_annotations(f'mw {skid} downstream partners', 'mw partner-loops-bilateral partners')
    pymaid.add_meta_annotations(f'mw {skid} upstream partners', 'mw partner-loops partners')
    pymaid.add_meta_annotations(f'mw {skid} downstream partners', 'mw partner-loops partners')
# %%
