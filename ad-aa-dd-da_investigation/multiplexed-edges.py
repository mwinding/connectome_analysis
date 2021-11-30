# %%
import os
os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import connectome_tools.process_matrix as pm
import connectome_tools.celltype as ct
import navis

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

# %%
# load edges and involved neurons

medges3 = pd.read_csv('data/multiplexed_edges/threeplex-edges.csv', index_col=0)
medges4 = pd.read_csv('data/multiplexed_edges/fourplex-edges.csv', index_col=0)

nodes3 = np.unique(list(medges3.index) + list(medges3.target))
nodes4 = np.unique(list(medges4.index) + list(medges4.target))

pre_nodes3 = np.unique(list(medges3.index))
post_nodes3 = np.unique(list(medges3.target))
pre_nodes4 = np.unique(list(medges4.index))
post_nodes4 = np.unique(list(medges4.target))

celltypes3 = ct.Celltype_Analyzer([
    ct.Celltype('threeplex_nodes', nodes3),
    ct.Celltype('threeplex_pre-nodes', pre_nodes3),
    ct.Celltype('threeplex_post-nodes', post_nodes3)
])

celltypes4 = ct.Celltype_Analyzer([
    ct.Celltype('fourplex_nodes', nodes4),
    ct.Celltype('fourplex_pre-nodes', pre_nodes4),
    ct.Celltype('fourplex_post-nodes', post_nodes4)
])

# %%
# characterize cell types in each category

_, celltypes = ct.Celltype_Analyzer.default_celltypes()

celltypes3.set_known_types(celltypes)
celltypes4.set_known_types(celltypes)

celltypes3.memberships()
celltypes4.memberships()

# %%
# are strong a-d connections in these edges?

# generating tuples of edges for strong a-d connections and multiplexed connection types (threeplex and fourplex)
ad_edges = pd.read_csv('data/edges_threshold/pairwise-threshold_ad_all-edges.csv', index_col=0)
ad_edges_raw = list(map(lambda x: (x[0], x[1]), zip(ad_edges.upstream_skid, ad_edges.downstream_skid)))
medges3_raw = list(map(lambda x: (x[0], x[1]), zip(medges3.index, medges3.target)))
medges4_raw = list(map(lambda x: (x[0], x[1]), zip(medges4.index, medges4.target)))

# identify threeplex edges containing strong a-d edge
medges3_df = pd.DataFrame(medges3_raw, columns=['upstream_skid', 'downstream_skid'])
medges3_df['strong_ad'] = [1 if edge in ad_edges_raw else 0 for edge in medges3_raw]

# identify fourplex edges containing strong a-d edge
medges4_df = pd.DataFrame(medges4_raw, columns=['upstream_skid', 'downstream_skid'])
medges4_df['strong_ad'] = [1 if edge in ad_edges_raw else 0 for edge in medges4_raw]

print(f'{medges4_df.strong_ad.sum()/len(medges4_df.index):.2f} of fourplex edges contain strong a-d edges')
print(f'{medges3_df.strong_ad.sum()/len(medges3_df.index):.2f} of fourplex edges contain strong a-d edges')

# %%
