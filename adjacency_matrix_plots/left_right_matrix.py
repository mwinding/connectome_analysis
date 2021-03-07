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
import math
#from pymaid_creds import url, name, password, token
#import pymaid

#rm = pymaid.CatmaidInstance(url, token, name, password)
#left = pymaid.get_skids_by_annotation('mw left')
#right = pymaid.get_skids_by_annotation('mw right')
'''

fig, ax = plt.subplots(1,1,figsize=(4,4))
sns.heatmap(matrix, ax = ax, cmap='OrRd')
plt.savefig('adjacency_matrix/plots/matrix.pdf', bbox_inches='tight', transparent = True)
'''
# %%
#matrix = np.random.randint(50, size=(2500, 2500))
matrix = pd.read_csv('data/G-pair-sorted.csv', header = 0, index_col = 0)

fig, ax = plt.subplots(1,1,figsize=(4,4))
#sns.heatmap(matrix, ax = ax, cmap='OrRd')

plt.imshow(matrix, cmap='OrRd', interpolation='none', vmax = 20)
plt.savefig('adjacency_matrix_plots/plots/matrix.pdf', bbox_inches='tight', transparent = True)

# %%
