import sys
sys.path.append("/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/")

import connectome_tools.process_skeletons as proskel
'''
paths = ['axon_dendrite_data/splittable_skeletons_right2.csv', 
        'axon_dendrite_data/splittable_connectors_right2.csv',
        'axon_dendrite_data/splittable_connectordists_right2_raw.csv',
        'axon_dendrite_data/splittable_connectordists_right2_norm.csv',
        'axon_dendrite_data/splittable_connectordists_right2_norm2.csv']

proskel.connector_dist_batch(paths[0], paths[1], paths[2], paths[3], paths[4]) 
'''

paths2 = ['axon_dendrite_data/splittable_skeletons_left2.csv', 
        'axon_dendrite_data/splittable_connectors_left2.csv',
        'axon_dendrite_data/splittable_connectordists_left2_raw.csv',
        'axon_dendrite_data/splittable_connectordists_left2_norm.csv',
        'axon_dendrite_data/splittable_connectordists_left2_norm2.csv']

proskel.connector_dist_batch(paths2[0], paths2[1], paths2[2], paths2[3], paths2[4]) 

