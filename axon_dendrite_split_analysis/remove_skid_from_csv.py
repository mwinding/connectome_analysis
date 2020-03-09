import csv
import pandas as pd

skeletons = pd.read_csv('axon_dendrite_data/splittable_skeletons_all.csv', header=0, skipinitialspace=True, keep_default_na = False)
connectors = pd.read_csv('axon_dendrite_data/splittable_connectors_all.csv', header=0, skipinitialspace=True, keep_default_na = False)



print(len(skeletons))
#4663875

skeletons = skeletons[skeletons['skeleton_id']!=4663875]
skeletons = skeletons[skeletons['skeleton_id']!=17360014]
skeletons = skeletons[skeletons['skeleton_id']!=16595001]
skeletons = skeletons[skeletons['skeleton_id']!=16675987]

print(len(skeletons))


print(len(connectors))
connectors = connectors[connectors['skeleton_id']!=4663875]
connectors = connectors[connectors['skeleton_id']!=17360014]
connectors = connectors[connectors['skeleton_id']!=16595001]
connectors = connectors[connectors['skeleton_id']!=16675987]
print(len(connectors))

skeletons.to_csv('axon_dendrite_data/splittable_skeletons_all_mod.csv')
connectors.to_csv('axon_dendrite_data/splittable_connectors_all_mod.csv')
