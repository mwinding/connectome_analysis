# %%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

from contools import Promat
from data_settings import data_date, pairs_path

# %%
neurons = pymaid.get_skids_by_annotation('nr Tel-like 10') # Tel-like 10

# use pregenerated edge list
edges = Promat.pull_edges(type_edges='ad', threshold=0.01, data_date=data_date, pairs_combined=True)

# downstream 3-hops of Tel-like 10
downstream = Promat.downstream_multihop(edges=edges, sources=neurons, hops=3, pairs_combined=True)
#[pymaid.add_annotations(skids, f'nr Tel-like 10 downstream {i+1}-hop') for i, skids in enumerate(downstream)]

# identify dVNcs in 3-hops downstream of Tel-like 10
dVNC = pymaid.get_skids_by_annotation('mw dVNC')
downstream_dVNCs = [list(np.intersect1d(dVNC, skids)) for skids in downstream]
#[pymaid.add_annotations(skids, f'nr Tel-like 10 downstream-dVNCs {i+1}-hop') for i, skids in enumerate(downstream_dVNCs)]

# %%