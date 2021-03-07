import sys
sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')

import pymaid
import pyoctree
from pymaid_creds import url, name, password, token
import numpy as np
import pandas as pd
import re

rm = pymaid.CatmaidInstance(url, token, name, password)

# save old skids/annotation correspondences
annotated = pymaid.get_annotated('MB nomenclature')
r = re.compile(r"CN-\d+")
CNs = list(filter(r.match, annotated['name']))

skid_list = []
for i in CNs:
    neurons = pymaid.get_annotated(i)
    for j in neurons['skeleton_ids']:
        skid_list.append([j, i])

CNs_df = pd.DataFrame(skid_list, columns = ['skid', 'annotation'])
print(CNs_df)

CNs_df.to_csv('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/scripts/CNs.csv')

# I didn't have permissions to add new annotations, so I asked Chris to finish this part for me

#pymaid.add_annotations(['12704069', '2966602'], ['test', 'test2'])
'''
for i in CNs:
    old_annot = '%s%s' %('old_', i)
    neurons = pymaid.get_annotated(i)
    #print(neurons)
    pymaid.add_annotations(neurons.skeleton_ids[0], old_annot)
    #print('annotation:%s' %i)
    print('%s changed to %s' %(i, old_annot))

    #skids = pymaid.get_annotated(i)
    #print('Skids annotated by %s:' %i)
    #print(skids['skeleton_ids'])
    break
'''