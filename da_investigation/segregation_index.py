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

mixed = pymaid.get_skids_by_annotation("mw mixed axon/dendrite")
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
    if((pairs['leftid'][i] in seg_index.index) & (pairs['rightid'][i] in seg_index.index) & (pairs['leftid'][i] in mixed) & (pairs['rightid'][i] in mixed)):
        left_identifier = pairs['leftid'][i]
        left_seg = seg_index.loc[left_identifier]['Segregation index']
        
        right_identifier = identify_pair(pairs['leftid'][i], pairs)
        right_seg = seg_index.loc[right_identifier]['Segregation index']
                
        seg_paired.append([left_identifier, right_identifier, left_seg, right_seg, (left_seg + right_seg)/2])

seg_paired = pd.DataFrame(seg_paired, columns= cols)


# %%
seg_paired = seg_paired.iloc[seg_paired['segregation_average'].sort_values(ascending=False).index, :]

seg_paired10 = seg_paired[seg_paired['segregation_average'] > 0.10]
# completed 35, 40, 25, 42, 1, 29, 2, 37, 32, 5, 41, 33, 27, 13
# unmodified 4985759	9291474	as "mw mixed axon/dendrite"

# 15770140	12049797 are interesting bilateral neurons

seg_paired5 = seg_paired[seg_paired['segregation_average'] > 0.05]
# completed 22, 30, 0, 8, 26, 6, 14, 24, 25, 20, 7, 19, 28

seg_paired_5 = seg_paired[seg_paired['segregation_average'] < 0.05]
# unmodified index  leftid  rightid; notes
# unmodified 13	8814524	17340973; check later
# unmodified 18	11009492	17304269; check later	
# unmodified 11	8338584	4195012; strange MBON-h1
# unmodified 3	7802210	16797672; strange MBON-i1
# unmodified 12	8798010	4230749; strange MBON-h2

# unmodified 8955607; unpaired and perhaps not complete

# unmodified 5	7939979	8198317; probably actually unsplittable
# unmodified 17	11007420	16884962; probably actually unsplittable
# unmodified 10	8311264	8198238; probably actually unsplittable
# unmodified 29	5030808	10831005; probably actually unsplittable
# unmodified 4	7939890	6557581; probably actually unsplittable (Broad T3)
# unmodified 4620453; unpaired, probably actually unsplittable (duplicate of AVL011 PN)

# completed 16, 21, 2, 15, 23, 27, 9

# 16283590 removed mw mixed axon/dendrite; not sure about split though 
# 17588623 removed mw mixed axon/dendrite; not sure about split though
# %%
