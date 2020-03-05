import csv
import pandas as pd

skeletons = pd.read_csv('axon_dendrite_data/splittable_skeletons_left2.csv', header=0, skipinitialspace=True, keep_default_na = False)
connectors = pd.read_csv('axon_dendrite_data/splittable_connectors_left2.csv', header=0, skipinitialspace=True, keep_default_na = False)

#print(len(skeletons))
#print(len(skeletons.drop('4663875')))

print(sum(skeletons['skeleton_id']!=4663875))
print(len(skeletons))
#4663875

skeletons = skeletons[skeletons['skeleton_id']!=4663875]

print(len(connectors))
connectors = connectors[connectors['skeleton_id']!=4663875]
print(len(connectors))

skeletons.to_csv('axon_dendrite_data/splittable_skeletons_left2_mod.csv')
connectors.to_csv('axon_dendrite_data/splittable_connectors_left2_mod.csv')