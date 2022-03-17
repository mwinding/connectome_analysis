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
import connectome_tools.process_matrix as pm
import connectome_tools.cascade_analysis as casc

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.family'] = 'arial'

rm = pymaid.CatmaidInstance(url, token, name, password)

brain = pymaid.get_skids_by_annotation('mw brain neurons')

lineages = pymaid.get_annotated('Volker').name
lineages = [('\\' + x) if x[0]=='*' in x else x for x in lineages]
remove_these = [ 'DPLal',
                'BLP1/2_l akira',
                'BLAd_l akira',
                'BLD5/6_l akira',
                'DPMl_l',
                '\\*DPLal1-3_l akira',
                '\\*DPLal1-3_r akira',
                '\\*DPLc_ant_med_r akira',
                '\\*DPLm_r akira',
                '\\*DPMl12_post_r akira',
                '\\*DPMpl3_r akira',
                'unknown lineage']
lineages = [x for x in lineages if x not in remove_these ]
lineages.sort()
lineage_skids = [list(np.intersect1d(pymaid.get_skids_by_annotation(x), brain)) for x in lineages]

lineage_counts = pd.DataFrame(zip(lineages, [len(x) for x in lineage_skids]), columns = ['lineage', 'count'])

lineage_diffs = []
for i in np.arange(0, len(lineage_counts), 2):
    diff = abs(lineage_counts.loc[i, 'count'] - lineage_counts.loc[i+1, 'count'])
    lineage_diffs.append([lineage_counts.loc[i, 'lineage'], diff])

lineage_diffs = pd.DataFrame(lineage_diffs, columns = ['lineage', 'diffs'])

fig, ax = plt.subplots(1,1,figsize=(4,4))
sns.swarmplot(y=lineage_diffs.diffs, orient='v', ax=ax)
ax.set_yticks(range(0,16))
ax.set(ylabel = 'Left-right difference in member counts', title='Lineage Comparison')
plt.savefig('lineage_analysis/plots/different-between-lineages-left-right_2021-04-06b.pdf', bbox_inches='tight', transparent = True)

# %%
