# %%
import os
import sys
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:
    pass

import pymaid
import numpy as np
import pandas as pd

A1 = pymaid.get_skids_by_annotation('mw A1 neurons')
A1_names = pymaid.get_names(A1)

A1_names_array = []
for skid in A1:
    name = A1_names[str(skid)]
    A1_names_array.append([skid, name])
    
A1_df = pd.DataFrame(A1_names_array, columns = ['skid', 'name'])
# %%

A1_names_short = pd.DataFrame([[A1_df.skid[i], name[:5]] for i, name in enumerate(A1_df.name)],
                                columns = ['skid', 'name'])
matches = []
for entry in A1_names_short:
    indices = np.where([entry.name==x for x in A1_names_short.name])[0]
    if(len(lndices)==2):
        matches = [indices[0], indices[1], ]