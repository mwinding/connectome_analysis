import sys
sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')

import pymaid
import pyoctree
from pymaid_creds import url, name, password, token
import numpy as np
import pandas as pd
import re
from itertools import chain

rm = pymaid.CatmaidInstance(url, name, password, token)

# breaking all brain neurons into 200 neuron chunks for blender import
# blender needs skids separated by commas
brain = pymaid.get_annotated('mw brain neurons')
brain_skids = brain['skeleton_ids']

skids = []
for i in brain_skids:
    skids = i + skids

size = 200
chunk_num = int(len(skids)/size)
remainder = len(skids)%size

chunks = []
for i in range(0, chunk_num):
    chunks.append(skids[(i*size):((i+1)*size)])

    if(i == (chunk_num-1)):
        chunks.append(skids[((i+1)*size):(((i+1)*size)+remainder)])

# verify that contents of list of lists are correct
# check total number of entries and whether each are unique
##verify = list(chain.from_iterable(chunks))
#print(len(verify))
#print(pd.Series(verify).unique)

print('--------------')
print('--------------')
print('list below...')
print('--------------')
print('--------------')


print(chunks[1])