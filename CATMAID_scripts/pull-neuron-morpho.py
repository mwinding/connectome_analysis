# %%
# details of my specific repo
import os
import sys
os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory

# %%
# start demo here
import pymaid
from pymaid_creds import url, name, password, token # my private CATMAID credentials
rm = pymaid.CatmaidInstance(url, token, name, password)

skids = [992646, 15599768]
pair_morpho = pymaid.get_neurons(skids)
pair_morpho.nodes.loc[:, ['x', 'y', 'z']]