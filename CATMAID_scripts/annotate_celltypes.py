# %%
import os
import sys
os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory

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

# %%
#

[pymaid.add_annotations(x.skids, f'mw exclusive-celltype {x.name}') for x in celltypes]
pymaid.add_meta_annotations([f'mw exclusive-celltype {x.name}' for x in celltypes], 'mw exclusive celltypes')

other = list(np.setdiff1d(pymaid.get_skids_by_annotation('mw brain neurons'),
                        ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw exclusive celltypes')))

pymaid.add_annotations(other, 'mw exclusive-celltype other')
pymaid.add_meta_annotations('mw exclusive-celltype other', 'mw exclusive celltypes')
