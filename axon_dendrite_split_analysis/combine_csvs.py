import csv
import pandas as pd

# I had to export these skeleton morphologies in groups of ~600 so CATMAID didn't crash
# this script combines them again into one CSV
skeletons_l1 = pd.read_csv('axon_dendrite_split_analysis/splittable_skeletons_left1.csv', header=0, skipinitialspace=True, keep_default_na = False)
skeletons_l2 = pd.read_csv('axon_dendrite_split_analysis/splittable_skeletons_left2.csv', header=0, skipinitialspace=True, keep_default_na = False)
skeletons_r1 = pd.read_csv('axon_dendrite_split_analysis/splittable_skeletons_right1.csv', header=0, skipinitialspace=True, keep_default_na = False)
skeletons_r2 = pd.read_csv('axon_dendrite_split_analysis/splittable_skeletons_right2.csv', header=0, skipinitialspace=True, keep_default_na = False)

connectors_l1 = pd.read_csv('axon_dendrite_split_analysis/splittable_connectors_left1.csv', header=0, skipinitialspace=True, keep_default_na = False)
connectors_l2 = pd.read_csv('axon_dendrite_split_analysis/splittable_connectors_left2.csv', header=0, skipinitialspace=True, keep_default_na = False)
connectors_r1 = pd.read_csv('axon_dendrite_split_analysis/splittable_connectors_right1.csv', header=0, skipinitialspace=True, keep_default_na = False)
connectors_r2 = pd.read_csv('axon_dendrite_split_analysis/splittable_connectors_right2.csv', header=0, skipinitialspace=True, keep_default_na = False)

skeletons = pd.concat([skeletons_l1, skeletons_l2, skeletons_r1, skeletons_r2])
connectors = pd.concat([connectors_l1, connectors_l2, connectors_r1, connectors_r2])

skeletons.to_csv('axon_dendrite_split_analysis/splittable_skeletons_2020-3-3.csv')
connectors.to_csv('axon_dendrite_split_analysis/splittable_connectors_2020-3-3.csv')