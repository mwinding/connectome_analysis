import sys
sys.path.append("/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/")

import pymaid
import pyoctree
from pymaid_creds import url, name, password, token

#pymaid.clear_cache()
rm = pymaid.CatmaidInstance(url, token, name, password)

annotated = pymaid.get_annotated('mw neuron groups')
test1 = annotated['name'][3]
test2 = annotated['name'][4]

print(annotated)
print(pymaid.get_skids_by_annotation(test1))
print(pymaid.get_skids_by_annotation(test2))


