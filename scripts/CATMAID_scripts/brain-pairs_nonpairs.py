#%%

from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import numpy as np
import pandas as pd
import connectome_tools.process_matrix as pm

pairs = pm.Promat.get_pairs()

# %%
# identify brain neurons in pair-list; brain neurons not in pair-list
# output as CSVs

brain = pymaid.get_skids_by_annotation('mw brain neurons')

brain_pairs = pd.DataFrame([list(x) for x in pairs.values if x[0] in brain], columns=['left', 'right'])
brain_unpaired = pd.DataFrame(np.setdiff1d(brain, list(brain_pairs.left) + list(brain_pairs.right)), columns=['unpaired'])

brain_pairs.to_csv('data/pairs/brain-pairs.csv')
brain_unpaired.to_csv('data/pairs/brain-unpaired.csv')

# %%
