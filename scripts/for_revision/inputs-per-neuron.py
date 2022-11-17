# %%
# identify clusters with higher than 2*std mean axonic IO or dendritic OI

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
from data_settings import data_date, pairs_path
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

from contools import Celltype, Celltype_Analyzer, Promat

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 6
plt.rcParams['font.family'] = 'arial'

axon_inputs = pd.read_csv('data/adj/inputs_' + data_date + '.csv', index_col=0)
axon_outputs = pd.read_csv('data/adj/outputs_' + data_date + '.csv', index_col=0)
input_output = pd.concat([axon_inputs, axon_outputs], axis=1)

inputs = input_output.dendrite_input + input_output.axon_input
# %%
# 

brain_neurons = pymaid.get_skids_by_annotation(['mw brain neurons', 'mw brain accessory neurons'])
remove_neurons = pymaid.get_skids_by_annotation(['mw brain very incomplete', 'mw partially differentiated', 'mw motor'])
brain_neurons = list(np.setdiff1d(brain_neurons, remove_neurons)) # remove neurons that are incomplete or partially differentiated (as well as SEZ motor neurons)

sns.histplot(inputs.loc[brain_neurons], bins=40)

less100 = sum((inputs.loc[brain_neurons])<=100)/len(brain_neurons)
between100_200 = sum(((inputs.loc[brain_neurons])>100) & (inputs.loc[brain_neurons]<=200))/len(brain_neurons)
more200 = sum((inputs.loc[brain_neurons])>200)/len(brain_neurons)