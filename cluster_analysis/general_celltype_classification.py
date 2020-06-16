#%%
import os
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:
    pass

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pymaid as pymaid
import glob as gl

# identify list of neuron-groups to import
neuron_groups = gl.glob('data/color_iso-d=8/*.json')

# load skids of each neuron class
clusters = map(lambda x : pd.read_json(x), neuron_groups)
l_cluster = list(clusters)
# %%
df.skids[df.annotation == "sensory" | df.annotation == "motor"].unique()