#%%
import os
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:
    pass

# %%
import matplotlib.pyplot as plt
import seaborn as sns

from graspy.plot import gridplot, heatmap
from graspy.utils import binarize, pass_to_ranks
from src.data import load_metagraph
from src.visualization import CLASS_COLOR_DICT, adjplot

sns.set_context("talk")

mg = load_metagraph("G", version="2020-04-23", path = '/Volumes/GoogleDrive/My\ Drive/python_code/maggot_models/data/processed/')
mg.calculate_degrees(inplace=True)

adj = mg.adj  # adjacency matrix from the "mg" object
meta = mg.meta  # dataframe of node metadata

# %%
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
adjplot(
    adj,
    meta=meta,
    sort_class="hemisphere",  # group by hemisphere, this is a key for column in "meta"
    plot_type="scattermap",  # plot dots instead of a heatmap
    sizes=(1, 1),  # min and max sizes for dots, so this is effectively binarizing
    item_order="Pair ID",  # order by pairs (some have no pair here so don't look same)
    ax=ax,
)