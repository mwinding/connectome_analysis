#%%

import pymaid
from pymaid_creds import url, name, password, token
rm = pymaid.CatmaidInstance(url, token, name, password)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

from data_settings import data_date, pairs_path
from contools import Promat, Prograph, Celltype, Celltype_Analyzer, Analyze_Nx_G

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

edges = Promat.pull_edges('ad', threshold=0.01, data_date=data_date, pairs_combined=False)
pairs = Promat.get_pairs(pairs_path=pairs_path)

# identify contralateral sens neurons and contra-contra neurons to flip their left/right identities
neurons_to_flip = list(np.intersect1d(pymaid.get_skids_by_annotation('mw contralateral axon'), pymaid.get_skids_by_annotation('mw contralateral dendrite')))

# define left and right neurons from a hemispheric propagation perspective, flip left/right identity as appropriate
left, right = Promat.get_hemis('mw left', 'mw right', neurons_to_flip=neurons_to_flip)

# %%
# load ipsi, bilateral, contra left/right

ipsi_axon = pymaid.get_skids_by_annotation('mw ipsilateral axon') + list(np.intersect1d(pymaid.get_skids_by_annotation('mw contralateral axon'), pymaid.get_skids_by_annotation('mw contralateral dendrite')))
bilat_axon = pymaid.get_skids_by_annotation('mw bilateral axon')
contra_axon = np.setdiff1d(pymaid.get_skids_by_annotation('mw contralateral axon'), pymaid.get_skids_by_annotation('mw contralateral dendrite'))
brain = pymaid.get_skids_by_annotation('mw brain neurons')

ipsi_axon = list(np.intersect1d(ipsi_axon, brain))
bilat_axon = list(np.intersect1d(bilat_axon, brain))
contra_axon = list(np.intersect1d(contra_axon, brain))

ipsi_axon_left = Celltype('ipsi-left', list(np.intersect1d(ipsi_axon, left)))
ipsi_axon_right = Celltype('ipsi-right', list(np.intersect1d(ipsi_axon, right)))
bilat_axon_left = Celltype('bilat-left', list(np.intersect1d(bilat_axon, left)))
bilat_axon_right = Celltype('bilat-right', list(np.intersect1d(bilat_axon, right)))
contra_axon_left = Celltype('contra-left', list(np.intersect1d(contra_axon, left)))
contra_axon_right = Celltype('contra-right', list(np.intersect1d(contra_axon, right)))

cta = Celltype_Analyzer([ipsi_axon_left, bilat_axon_left, contra_axon_left, contra_axon_right, bilat_axon_right, ipsi_axon_right])

# %%
# connection probability

cta_connect_prob = cta.connection_prob(edges=edges, pairs_combined=False)
cta_connect_prob
sns.heatmap(cta_connect_prob, cmap='Blues', square=True)
# %%
# connection probability without pair loops

pairloops = Celltype_Analyzer.get_skids_from_meta_annotation('mw pair loops')

ipsi_axon_npl = list(np.setdiff1d(ipsi_axon, pairloops))
bilat_axon_npl = list(np.setdiff1d(bilat_axon, pairloops))
contra_axon_npl = list(np.setdiff1d(contra_axon, pairloops))

ipsi_axon_npl_left = Celltype('ipsi-left', list(np.intersect1d(ipsi_axon_npl, left)))
ipsi_axon_npl_right = Celltype('ipsi-right', list(np.intersect1d(ipsi_axon_npl, right)))
bilat_axon_npl_left = Celltype('bilat-left', list(np.intersect1d(bilat_axon_npl, left)))
bilat_axon_npl_right = Celltype('bilat-right', list(np.intersect1d(bilat_axon_npl, right)))
contra_axon_npl_left = Celltype('contra-left', list(np.intersect1d(contra_axon_npl, left)))
contra_axon_npl_right = Celltype('contra-right', list(np.intersect1d(contra_axon_npl, right)))

cta_no_pl = Celltype_Analyzer([ipsi_axon_npl_left, bilat_axon_npl_left, contra_axon_npl_left, 
                            contra_axon_npl_right, bilat_axon_npl_right, ipsi_axon_npl_right])
cta_no_pl_connect_prob = cta_no_pl.connection_prob(edges=edges, pairs_combined=False)
cta_no_pl_connect_prob
sns.heatmap(cta_no_pl_connect_prob, cmap='Blues', square=True)

# %%
# connections per neuron type

# contra
edges_test = edges.copy()
edges_test = edges_test.set_index('upstream_skid', drop=False)
edges_contra = edges_test.loc[np.intersect1d(edges_test.index, contra_axon)]
edges_contra = edges_contra.set_index('downstream_skid', drop=False)
edges_contra_contra = edges_contra.loc[np.intersect1d(edges_contra.index, contra_axon)]

num_contra_partners = np.mean(edges_contra_contra.groupby('upstream_skid').count().downstream_skid)
std_contra_partners = np.std(edges_contra_contra.groupby('upstream_skid').count().downstream_skid)

# ipsi
edges_test = edges.copy()
edges_test = edges_test.set_index('upstream_skid', drop=False)
edges_ipsi = edges_test.loc[np.intersect1d(edges_test.index, ipsi_axon)]
edges_ipsi = edges_ipsi.set_index('downstream_skid', drop=False)
edges_ipsi_contra = edges_ipsi.loc[np.intersect1d(edges_ipsi.index, contra_axon)]

num_ipsi_partners = np.mean(edges_ipsi_contra.groupby('upstream_skid').count().downstream_skid)
std_ipsi_partners = np.std(edges_ipsi_contra.groupby('upstream_skid').count().downstream_skid)

print(f'Individual contralateral neurons synapse onto {num_contra_partners:.1f}+/-{std_contra_partners:.1f} contralateral neurons,')
print(f'while individual ipsilateral neurons synapsed onto {num_ipsi_partners:.1f}+/-{std_ipsi_partners:.1f} contralateral neurons.')
# %%
