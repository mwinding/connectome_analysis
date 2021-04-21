#%%
import os
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
except:
    pass

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
import pymaid
import connectome_tools.cluster_analysis as clust
import connectome_tools.celltype as ct

rm = pymaid.CatmaidInstance(url, token, name, password)
celltypes = ct.Celltype_Analyzer.default_celltypes()

# %%
