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

rm = pymaid.CatmaidInstance(url, token, name, password)

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

# %%

ct.plot_cell_types_cluster('lvl7_labels', 'cluster_analysis/plots/celltypes-clusters.pdf')
ct.plot_marginal_cell_type_cluster((2,1), ct.Celltype('MBONs', pymaid.get_skids_by_annotation('mw MBON')), 'green', 'lvl7_labels', 'cluster_analysis/plots/MBON_celltypes-clusters.pdf')

# %%

celltypes_data, celltypes = ct.Celltype_Analyzer.default_celltypes()
in_hubs = ct.Celltype('In Hubs', pymaid.get_skids_by_annotation('mw hubs_in'), 'green')
in_hubs.plot_cell_type_memberships(celltypes)

# plot all hub-types at once
in_hubs_ct = ct.Celltype('Out Hubs', pymaid.get_skids_by_annotation('mw hubs_out'), 'orange')
out_hubs_ct = ct.Celltype('In Hubs', pymaid.get_skids_by_annotation('mw hubs_in'), 'green')
in_out_hubs_ct = ct.Celltype('In-Out Hubs', pymaid.get_skids_by_annotation('mw hubs_in_out'), 'red')

hubs = ct.Celltype_Analyzer([in_hubs_ct, in_out_hubs_ct, out_hubs_ct])
hubs.set_known_types(celltypes)
hubs.plot_memberships()
# %%
