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
# load edges using brain inputs, brain neurons, and brain accessory neurons

brain_access = pymaid.get_skids_by_annotation('mw brain neurons') + pymaid.get_skids_by_annotation('mw brain accessory neurons') + ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain inputs')
ad_edges = pm.Promat.pull_edges(type_edges='ad', pairs_combined=True, select_neurons=brain_access)
aa_edges = pm.Promat.pull_edges(type_edges='aa', pairs_combined=True, select_neurons=brain_access)
dd_edges = pm.Promat.pull_edges(type_edges='dd', pairs_combined=True, select_neurons=brain_access)
da_edges = pm.Promat.pull_edges(type_edges='da', pairs_combined=True, select_neurons=brain_access)


# %%
# number of edges and neurons

_, celltypes = ct.Celltype_Analyzer.default_celltypes()

ad_cells = list(np.unique(list(ad_edges.upstream_pair_id) + list(ad_edges.upstream_pair_id)))
aa_cells = list(np.unique(list(aa_edges.upstream_pair_id) + list(aa_edges.upstream_pair_id)))
dd_cells = list(np.unique(list(dd_edges.upstream_pair_id) + list(dd_edges.upstream_pair_id)))
da_cells = list(np.unique(list(da_edges.upstream_pair_id) + list(da_edges.upstream_pair_id)))

cells_ct = [ct.Celltype('ad_cells', ad_cells), ct.Celltype('aa_cells', aa_cells),
            ct.Celltype('dd_cells', dd_cells), ct.Celltype('da_cells', da_cells)]

cells_ct = ct.Celltype_Analyzer(cells_ct)
cells_ct.set_known_types(celltypes)
cells_ct.memberships(raw_num=True)
