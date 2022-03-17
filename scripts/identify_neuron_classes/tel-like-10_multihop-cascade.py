# %%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import connectome_tools.process_matrix as pm

# %%
neurons = pymaid.get_skids_by_annotation('nr Tel-like 10') # Tel-like 10

# use pregenerated edge list
edges = pd.read_csv('data/edges_threshold/pairwise-threshold_ad_all-edges.csv', index_col=0)

# downstream 3-hops of Tel-like 10
downstream = pm.Promat.downstream_multihop(edges=edges, sources=neurons, hops=3)
[pymaid.add_annotations(skids, f'nr Tel-like 10 downstream {i+1}-hop') for i, skids in enumerate(downstream)]

# identify dVNcs in 3-hops downstream of Tel-like 10
dVNC = pymaid.get_skids_by_annotation('mw dVNC')
downstream_dVNCs = [list(np.intersect1d(dVNC, skids)) for skids in downstream]
[pymaid.add_annotations(skids, f'nr Tel-like 10 downstream-dVNCs {i+1}-hop') for i, skids in enumerate(downstream_dVNCs)]

# %%