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

seg_index = pd.read_csv('da_investigation/csv/segregationindex_mw_mixed_axondendrite_2020_06_05.csv', header = 0, index_col = 0)
pairs = pd.read_csv('data/pairs-2020-05-08.csv', header = 0) # import pairs

# %%
# arrange segregation index by pairs

def identify_pair(skid, pairList):

    pair_skid = []
    
    if(skid in pairList["leftid"].values):
        pair_skid = pairList["rightid"][pairList["leftid"]==skid].iloc[0]

    if(skid in pairList["rightid"].values):
        pair_skid = pairList["leftid"][pairList["rightid"]==skid].iloc[0]

    return(pair_skid)


seg_index.index = seg_index['skeleton_id']
cols = ['leftid', 'rightid', 'segregation_left', 'segregation_right', 'segregation_average']
seg_paired = []

for i in range(0, len(pairs['leftid'])):
    if((pairs['leftid'][i] in seg_index.index) & (pairs['rightid'][i] in seg_index.index)):
        left_identifier = pairs['leftid'][i]
        left_seg = seg_index.loc[left_identifier]['Segregation index']
        
        right_identifier = identify_pair(pairs['leftid'][i], pairs)
        right_seg = seg_index.loc[right_identifier]['Segregation index']
                
        seg_paired.append([left_identifier, right_identifier, left_seg, right_seg, (left_seg + right_seg)/2])

seg_paired = pd.DataFrame(seg_paired, columns= cols)


# %%
6219445