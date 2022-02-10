# %%

from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import connectome_tools.process_matrix as pm
import connectome_tools.celltype as ct
import navis

celltypes_df, celltypes = ct.Celltype_Analyzer.default_celltypes()
celltypes_df.iloc[6, 0] = 'MB-FFNs'
celltypes[6].name = 'MB-FFNs'

# %%
# write the binary celltypes

[pymaid.add_annotations(x.skids, f'ctd {x.name}') for x in celltypes]
pymaid.add_meta_annotations([f'ctd {x.name}' for x in celltypes], 'mw brain celltypes discrete')

pymaid.clear_cache()
other = list(np.setdiff1d(pymaid.get_skids_by_annotation('mw brain neurons'),
                        ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain celltypes discrete')))

pymaid.add_annotations(other, 'ctd other')
pymaid.add_meta_annotations('ctd other', 'mw brain celltypes discrete')

# %%
# write overlapping celltypes

celltype_names = ['sensories', 'PNs', 'ascendings', 'PNs-somato', 'LNs', 'LHNs', 'FFNs', 'MBINs', 'KCs', 'MBONs', 'MB-FBNs', 'CNs', 'pre-dSEZs', 'pre-dVNCs', 'RGNs', 'dSEZs', 'dVNCs']
annots = ['mw brain ' + x for x in celltype_names]
celltype_names = ['sensories', 'PNs', 'ascendings', 'PNs-somato', 'LNs', 'LHNs', 'MB-FFNs', 'MBINs', 'KCs', 'MBONs', 'MB-FBNs', 'CNs', 'pre-dSEZs', 'pre-dVNCs', 'RGNs', 'dSEZs', 'dVNCs']

all_celltypes = [ct.Celltype(celltype_names[i], ct.Celltype_Analyzer.get_skids_from_meta_annotation(annot)) for i, annot in enumerate(annots)]

[pymaid.add_annotations(x.skids, f'ct {x.name}') for x in all_celltypes]
pymaid.add_meta_annotations([f'ct {x.name}' for x in celltypes], 'mw brain celltypes')

pymaid.clear_cache()
other = list(np.setdiff1d(pymaid.get_skids_by_annotation('mw brain neurons'),
                        ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain celltypes')))

pymaid.add_annotations(other, 'ct other')
pymaid.add_meta_annotations('ct other', 'mw brain celltypes')
# %%
