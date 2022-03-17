# %%

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

adj_ad = pm.Promat.pull_adj(type_adj='ad', subgraph='brain and accesssory')

# %%
# load 2nd- and 3rd-order modalities

order = ['olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'thermo-warm', 'thermo-cold', 'visual', 'noci', 'mechano-Ch', 'mechano-II/III', 'proprio', 'respiratory']
o2_annot = ['mw ' + name + ' 2nd_order' for name in order]
o3_annot = ['mw ' + name + ' 3rd_order' for name in order]

o2 = [pymaid.get_skids_by_annotation(annot) for annot in o2_annot]
o3 = [pymaid.get_skids_by_annotation(annot) for annot in o3_annot]

# %%
# ratio of cells between 2nd- and 3rd-order

ratio = [f'{len(o3[i])/len(o2[i]):.2f}' for i in range(len(o2))]

