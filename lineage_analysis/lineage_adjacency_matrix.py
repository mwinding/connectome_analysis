#%%
import os
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:
    pass

#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import connectome_tools.process_matrix as promat
import pymaid
from pymaid_creds import url, name, password, token
import string

rm = pymaid.CatmaidInstance(url, name, password, token)
lineages = pymaid.get_annotated('Volker')

# %%
# added escape character to '*' in each lineage annotation
# this is required so pymaid doesn't break

for i in np.arange(0, len(lineages['name']), 1):
    if("*" in lineages['name'][i]):
        lineages['name'][i] = '%s%s' %('\\',lineages['name'][i])
        print(lineages['name'][i])

# %%
# identify skids per lineage

lineage_skids = []
for i in lineages['name']:
    skids = pymaid.get_skids_by_annotation(i)
    lineage_skids.append(skids)

# %%
# identifying the indices of the adjacency matrix that are part of each lineage
matrix = pd.read_csv('data/G-pair-sorted.csv', header = 0, index_col = 0)

lineage_index = []
for i in lineage_skids:
    index = []
    for j in np.arange(0, len(matrix.index), 1):
        if(matrix.index[j] in i):
            index.append(j)
    
    lineage_index.append(index)

# summing up all elements of a lineage by rows and columns
lineage_rows = []
for i in lineage_index:
    summed_rows = matrix.iloc[i, :].sum(axis=0)
    lineage_rows.append(summed_rows)
    
lineage_rows_df = pd.DataFrame(lineage_rows)

lineage_rows_columns = []
for i in lineage_index:
    summed_rows_columns = lineage_rows_df.iloc[:, i].sum(axis=1)
    lineage_rows_columns.append(summed_rows_columns)
    
lineage_rows_columns_df = pd.DataFrame(lineage_rows_columns)

# %%
# plotting adjacency matrix of lineages

fig, ax = plt.subplots(1,1,figsize=(4,4))

sns.heatmap(lineage_rows_columns_df, cmap='OrRd', vmax = 100)
#sns.clustermap(lineage_rows_columns_df, metric='correlation')

#lineage_rows_columns_df.to_csv('lineage_analysis/lineage_matrix.csv')

# %%
