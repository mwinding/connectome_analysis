import pandas as pd
import numpy as np
import csv

# import synapses divided across hemispheres
hemisphere_data = pd.read_csv('left_right_hemisphere_data/brain_hemisphere_membership.csv', header = 0)
#print(hemisphere_data)

# import pair list CSV, manually generated
#pairs = pd.read_csv('data/bp-pairs-2020-01-28.csv', header = 0)

# import skids of neurons that cross commissure
commissure_neurons = pd.read_json('left_right_hemisphere_data/cross_commissure-2020-3-2.json')['skeleton_id'].values

#print(type(commissure_neurons[0]))
#print(type(hemisphere_data['skeleton'][0]))
ipsi_neurons = np.setdiff1d(hemisphere_data['skeleton'], commissure_neurons)

ipsi_neurons_bool = pd.Series(hemisphere_data['skeleton'].values).isin(ipsi_neurons)
contra_neurons_bool = ~pd.Series(hemisphere_data['skeleton'].values).isin(ipsi_neurons)

print("IPSI")
print("Postsynaptic Sites")
print(sum(hemisphere_data[ipsi_neurons_bool]['n_inputs_left'].values))
print(sum(hemisphere_data[ipsi_neurons_bool]['n_inputs_right'].values))
print(sum(hemisphere_data[ipsi_neurons_bool]['n_inputs_left'].values)/sum(hemisphere_data[ipsi_neurons_bool]['n_inputs_right'].values))
print("")

print("Presynaptic Sites")
print(sum(hemisphere_data[ipsi_neurons_bool]['n_outputs_left'].values))
print(sum(hemisphere_data[ipsi_neurons_bool]['n_outputs_right'].values))
print(sum(hemisphere_data[ipsi_neurons_bool]['n_outputs_left'].values)/sum(hemisphere_data[ipsi_neurons_bool]['n_outputs_right'].values))
print("")

print("Treenodes")
print(sum(hemisphere_data[ipsi_neurons_bool]['n_treenodes_left'].values))
print(sum(hemisphere_data[ipsi_neurons_bool]['n_treenodes_right'].values))
print(sum(hemisphere_data[ipsi_neurons_bool]['n_treenodes_left'].values)/sum(hemisphere_data[ipsi_neurons_bool]['n_treenodes_right'].values))
print("")
print("")
print("")

print("CONTRA")
print("Postsynaptic Sites")
print(sum(hemisphere_data[contra_neurons_bool]['n_inputs_left'].values))
print(sum(hemisphere_data[contra_neurons_bool]['n_inputs_right'].values))
print(sum(hemisphere_data[contra_neurons_bool]['n_inputs_left'].values)/sum(hemisphere_data[contra_neurons_bool]['n_inputs_right'].values))
print("")

print("Presynaptic Sites")
print(sum(hemisphere_data[contra_neurons_bool]['n_outputs_left'].values))
print(sum(hemisphere_data[contra_neurons_bool]['n_outputs_right'].values))
print(sum(hemisphere_data[contra_neurons_bool]['n_outputs_left'].values)/sum(hemisphere_data[contra_neurons_bool]['n_outputs_right'].values))
print("")

print("Treenodes")
print(sum(hemisphere_data[contra_neurons_bool]['n_treenodes_left'].values))
print(sum(hemisphere_data[contra_neurons_bool]['n_treenodes_right'].values))
print(sum(hemisphere_data[contra_neurons_bool]['n_treenodes_left'].values)/sum(hemisphere_data[contra_neurons_bool]['n_treenodes_right'].values))
print("")

