import csv
import pandas as pd
'''
# I had to export these skeleton morphologies in groups of ~600 so CATMAID didn't crash
# this script combines them again into one CSV
dist_l1 = pd.read_csv('axon_dendrite_data/splittable_connectdists_left1_norm.csv', header=0, skipinitialspace=True, keep_default_na = False)
dist_l2 = pd.read_csv('axon_dendrite_data/splittable_connectordists_left2_norm.csv', header=0, skipinitialspace=True, keep_default_na = False)
dist_r1 = pd.read_csv('axon_dendrite_data/splittable_connectordists_right1_norm.csv', header=0, skipinitialspace=True, keep_default_na = False)
dist_r2 = pd.read_csv('axon_dendrite_data/splittable_connectordists_right2_norm.csv', header=0, skipinitialspace=True, keep_default_na = False)

connectors = pd.concat([dist_l1, dist_l2, dist_r1, dist_r2])

connectors.to_csv('axon_dendrite_data/splittable_connectordists_all_2020-3-5.csv')


dist_l1r = pd.read_csv('axon_dendrite_data/splittable_connectdists_left1_raw.csv', header=0, skipinitialspace=True, keep_default_na = False)
dist_l2r = pd.read_csv('axon_dendrite_data/splittable_connectordists_left2_raw.csv', header=0, skipinitialspace=True, keep_default_na = False)
dist_r1r = pd.read_csv('axon_dendrite_data/splittable_connectordists_right1_raw.csv', header=0, skipinitialspace=True, keep_default_na = False)
dist_r2r = pd.read_csv('axon_dendrite_data/splittable_connectordists_right2_raw.csv', header=0, skipinitialspace=True, keep_default_na = False)

connectors = pd.concat([dist_l1r, dist_l2r, dist_r1r, dist_r2r])

connectors.to_csv('axon_dendrite_data/splittable_connectordists_all_raw_2020-3-5.csv')

'''

skeleton_l1 = pd.read_csv('axon_dendrite_data/splittable_skeletons_left1.csv', header=0, skipinitialspace=True, keep_default_na = False)
skeleton_l2 = pd.read_csv('axon_dendrite_data/splittable_skeletons_left2.csv', header=0, skipinitialspace=True, keep_default_na = False)
skeleton_r1 = pd.read_csv('axon_dendrite_data/splittable_skeletons_right1.csv', header=0, skipinitialspace=True, keep_default_na = False)
skeleton_r2 = pd.read_csv('axon_dendrite_data/splittable_skeletons_right2.csv', header=0, skipinitialspace=True, keep_default_na = False)

skeletons = pd.concat([skeleton_l1, skeleton_l2, skeleton_r1, skeleton_r2])

skeletons.to_csv('axon_dendrite_data/splittable_skeletons_all.csv')



connectors_l1 = pd.read_csv('axon_dendrite_data/splittable_connectors_left1.csv', header=0, skipinitialspace=True, keep_default_na = False)
connectors_l2 = pd.read_csv('axon_dendrite_data/splittable_connectors_left2.csv', header=0, skipinitialspace=True, keep_default_na = False)
connectors_r1 = pd.read_csv('axon_dendrite_data/splittable_connectors_right1.csv', header=0, skipinitialspace=True, keep_default_na = False)
connectors_r2 = pd.read_csv('axon_dendrite_data/splittable_connectors_right2.csv', header=0, skipinitialspace=True, keep_default_na = False)

connectors = pd.concat([connectors_l1, connectors_l2, connectors_r1, connectors_r2])

connectors.to_csv('axon_dendrite_data/splittable_connectors_all.csv')