import sys
sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')

import pymaid
import pyoctree
from pymaid_creds import url, name, password, token
import numpy as np
import pandas as pd
import re
import natsort as ns

rm = pymaid.CatmaidInstance(url, token, name, password)

# save old skids/annotation correspondences
annotated = pymaid.get_annotated('MB nomenclature')
r = re.compile(r"CN-\d+")
CNs = list(filter(r.match, annotated['name']))

# natural sorting of CN names
CNs = ns.natsorted(CNs)

correspondences = []
for i in CNs:
    pair_details = pymaid.get_annotation_details('annotation:%s' %i)
    r = re.compile(r"old_CN-\d+")
    old_CN_name = list(filter(r.match, pair_details['annotation']))
    
    if(old_CN_name[0] != old_CN_name[1]):
        print("Warning! old_CN names don't match! Something is wrong with the annotations...")
    
    print('%s was renamed as %s' %(old_CN_name[0], i))
    correspondences.append([old_CN_name[0], i])

#print(correspondences)
correspondences = pd.DataFrame(data = correspondences, columns = ['old_name', 'new_name'])
print(correspondences)

correspondences.to_csv('CATMAID_scripts/old-new-CN-names.csv')
