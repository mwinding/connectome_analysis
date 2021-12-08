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
# pull sensory modalities

order2 = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain inputs 2nd_order')
order3 = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain inputs 3rd_order')
order4 = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain inputs 4th_order')
order5 = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain inputs 5th_order')

dVNC = pymaid.get_skids_by_annotation('mw dVNC')
dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')
RGN = pymaid.get_skids_by_annotation('mw RGN')

sensory_circuit_ct = [ct.Celltype('order2', order2), ct.Celltype('order3', order3), ct.Celltype('order4', order4), ct.Celltype('order5', order5)]
outputs_ct = [ct.Celltype('dVNC', dVNC), ct.Celltype('dSEZ', dSEZ), ct.Celltype('RGN', RGN)]
outputs_ct = ct.Celltype_Analyzer(outputs_ct)
outputs_ct.set_known_types(sensory_circuit_ct)