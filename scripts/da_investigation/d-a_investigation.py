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

connectors = pd.read_csv('data/connectors.csv', header = 0, index_col = 0)

dd = connectors.loc[(connectors['presynaptic_type'] == 'dendrite') & (connectors['postsynaptic_type'] == 'dendrite')]
da = connectors.loc[(connectors['presynaptic_type'] == 'dendrite') & (connectors['postsynaptic_type'] == 'axon')]
ad = connectors.loc[(connectors['presynaptic_type'] == 'axon') & (connectors['postsynaptic_type'] == 'dendrite')]
aa = connectors.loc[(connectors['presynaptic_type'] == 'axon') & (connectors['postsynaptic_type'] == 'axon')]
ud = connectors.loc[(connectors['presynaptic_type'] == 'unspli') & (connectors['postsynaptic_type'] == 'dendrite')]
ua = connectors.loc[(connectors['presynaptic_type'] == 'unspli') & (connectors['postsynaptic_type'] == 'axon')]
uu = connectors.loc[(connectors['presynaptic_type'] == 'unspli') & (connectors['postsynaptic_type'] == 'unspli')]
au = connectors.loc[(connectors['presynaptic_type'] == 'axon') & (connectors['postsynaptic_type'] == 'unspli')]
du = connectors.loc[(connectors['presynaptic_type'] == 'dendrite') & (connectors['postsynaptic_type'] == 'unspli')]

brain = pymaid.get_skids_by_annotation('mw brain neurons')
left = pymaid.get_skids_by_annotation('mw left')
right = pymaid.get_skids_by_annotation('mw right')
# %%
# are da synapses a mistake?
from numpy import random as rand

def membership(list1, list2):
    set1 = set(list1)
    return [item in set1 for item in list2]

da_brain = da[membership(brain, da['presynaptic_to'])]
da_brain.index = np.arange(0, len(da_brain), 1) # reset indices

# make 100 random indices to manually check da synapses
rand_index = rand.random_integers(0, len(da), size = 100)

da_selected = da.iloc[rand_index, :]
da_selected.to_csv('da_investigation/csv/random100_da_synapses.csv')

# %%
# how many da synapses are contributed by usplit neurons? should u be a separate channel?

print('Synapses contributed by\na-d: %i\na-a: %i\nd-d: %i\nd-a: %i\n' 
                    %(len(ad) + len(au),
                    len(aa),
                    len(dd) + len(ud) + len(uu) + len(du),
                    len(da) + len(ua)))


print('Synapses contributed by\na-d: %i\na-u: %i\n\na-a: %i\n\nd-d: %i\nu-d: %i\nu-u: %i\nd-u: %i\n\nd-a: %i\nu-a: %i\n' 
                    %(len(ad),
                    len(au),
                    len(aa),
                    len(dd),
                    len(ud),
                    len(uu),
                    len(du),
                    len(da),
                    len(ua),))


# %%
# are da synapses allows ipsilateral?

projectome = pd.read_csv('data/projectome.csv')

# %%
# are da synapses mirror of ad? (on same partners?)