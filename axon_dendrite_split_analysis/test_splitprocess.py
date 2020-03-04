import sys
sys.path.append("/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/")

import connectome_tools.process_skeletons as proskel

paths = ['axon_dendrite_data/splittable_skeletons_right1.csv', 
        'axon_dendrite_data/splittable_connectors_right1.csv',
        'axon_dendrite_data/splittable_connectors_right1_raw.csv',
        'axon_dendrite_data/splittable_connectors_right1_norm.csv',
        'axon_dendrite_data/splittable_connectors_right1_norm2.csv']

proskel.connector_dist_batch(paths[0], paths[1], paths[2], paths[3], paths[4]) 