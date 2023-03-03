# %%

from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from contools import Celltype, Celltype_Analyzer, Promat
import navis

celltypes_df, celltypes = Celltype_Analyzer.default_celltypes()

# old names and new replacement names
replace_dict = {'FFNs': 'MB-FFNs',
                'pre-dSEZs': 'pre-DN-SEZs',
                'pre-dVNCs': 'pre-DN-VNCs',
                'dSEZs': 'DN-SEZs',
                'dVNCs': 'DN-VNCs'}

# replace names with updated ones in dataFrame
for i in range(len(celltypes_df.index)):
    name = celltypes_df.iloc[i, 0]
    if (name in replace_dict.keys()):
        celltypes_df.iloc[i, 0] = replace_dict[celltypes_df.iloc[i, 0]]

# replace names with updated ones in Celltype objects
for i in range(len(celltypes)):
    name = celltypes[i].name
    if (name in replace_dict.keys()):
        celltypes[i].name = replace_dict[celltypes[i].name]  

# %%
# write the binary celltypes

[pymaid.add_annotations(x.skids, f'ctd {x.name}') for x in celltypes]
pymaid.add_meta_annotations([f'ctd {x.name}' for x in celltypes], 'mw brain celltypes discrete')

pymaid.clear_cache()
brain_neurons = np.setdiff1d(pymaid.get_skids_by_annotation('mw brain neurons'), pymaid.get_skids_by_annotation(['mw brain very incomplete', 'mw partially differentiated']))
other = np.setdiff1d(brain_neurons, Celltype_Analyzer.get_skids_from_meta_annotation('mw brain celltypes discrete'))

pymaid.add_annotations(other, 'ctd other')
pymaid.add_meta_annotations('ctd other', 'mw brain celltypes discrete')

# %%
# write overlapping celltypes

celltype_names = ['sensories', 'PNs', 'ascendings', 'PNs-somato', 'LNs', 'LHNs', 'FFNs', 'MBINs', 'KCs', 'MBONs', 'MB-FBNs', 'CNs', 'pre-dSEZs', 'pre-dVNCs', 'RGNs', 'dSEZs', 'dVNCs']
annots = ['mw brain ' + x for x in celltype_names]

for i in range(len(celltype_names)):
    name = celltype_names[i]
    if (name in replace_dict.keys()):
        celltype_names[i] = replace_dict[celltype_names[i]]     

all_celltypes = [Celltype(celltype_names[i], Celltype_Analyzer.get_skids_from_meta_annotation(annot)) for i, annot in enumerate(annots)]

[pymaid.add_annotations(x.skids, f'ct {x.name}') for x in all_celltypes]
pymaid.add_meta_annotations([f'ct {x.name}' for x in celltypes], 'mw brain celltypes')

pymaid.clear_cache()
brain_neurons = np.setdiff1d(pymaid.get_skids_by_annotation('mw brain neurons'), pymaid.get_skids_by_annotation(['mw brain very incomplete', 'mw partially differentiated']))
other = list(np.setdiff1d(brain_neurons, Celltype_Analyzer.get_skids_from_meta_annotation('mw brain celltypes')))

pymaid.add_annotations(other, 'ct other')
pymaid.add_meta_annotations('ct other', 'mw brain celltypes')
# %%
