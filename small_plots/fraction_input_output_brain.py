#%%
import os
import sys
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/maggot_models/')
    sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
except:
    pass


from pymaid_creds import url, name, password, token
import pymaid

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'arial'

rm = pymaid.CatmaidInstance(url, name, password, token)

brain = pymaid.get_skids_by_annotation('mw brain neurons')
sensories = [pymaid.get_skids_by_annotation(x) for x in pymaid.get_annotated('mw brain inputs and ascending').name]
sensories = [x for sublist in sensories for x in sublist]

outputs = [pymaid.get_skids_by_annotation(x) for x in pymaid.get_annotated('mw brain outputs').name]
outputs = [x for sublist in outputs for x in sublist]

brain = list(np.setdiff1d(brain, outputs))
# %%
# plot number brain inputs, interneurons, outputs

fig, ax = plt.subplots(1,1,figsize=(2.5,5))
sns.barplot(x=['Inputs', 'Interneurons', 'Outputs'], y=[len(sensories), len(brain), len(outputs)], ax=ax)
plt.xticks(rotation=45, ha='right')
plt.savefig('small_plots/plots/general-neuron-counts.pdf', format='pdf', bbox_inches='tight')
# %%
