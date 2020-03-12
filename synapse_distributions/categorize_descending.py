#%%
import sys
sys.path.append("/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/")

import pymaid
import pyoctree
from pymaid_creds import url, name, password, token
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

rm = pymaid.CatmaidInstance(url, name, password, token)

def get_connectors_group(annotation):
    skids = pymaid.get_skids_by_annotation(annotation)
    neurons = pymaid.get_neuron(skids)

    outputs = neurons.connectors[neurons.connectors['relation']==0]
    inputs = neurons.connectors[neurons.connectors['relation']==1]

    return(outputs, inputs)

#%%
dVNC_outputs, dVNC_inputs = get_connectors_group("mw dVNC")
motor_outputs, motor_inputs = get_connectors_group("mw VNC motorneurons A1")
sensory_outputs, sensory_inputs = get_connectors_group("mw VNC sensories A1")
premotor_outputs, premotor_inputs = get_connectors_group("mw VNC PMNs A1")


# %%
# based on border of SEZ/T1 75650, cut off outputs in SEZ/in brain
dVNC_outputs_inVNC = dVNC_outputs[dVNC_outputs['z']>=75650]
motor_inputs_inVNC = motor_inputs[motor_inputs['z']>=75650]
premotor_inputs_inVNC = premotor_inputs[premotor_inputs['z']>=75650]
sensory_outputs_inVNC = sensory_outputs[sensory_outputs['z']>=75650]

descending_axons = dVNC_outputs_inVNC.groupby('skeleton_id').mean()


# %%
# all locations of inputs on MNs and outputs from sensories
fig, ax = plt.subplots(1,1,figsize=(8,8))
#sns.kdeplot(sensory_outputs_inVNC['x'], sensory_outputs_inVNC['y'], cmap="Reds", ax = ax, shade = True, shade_lowest=False, levels = 20)
#sns.kdeplot(motor_inputs_inVNC['x'], motor_inputs_inVNC['y'], cmap="Blues", ax = ax, shade = True, shade_lowest=False, levels = 20)
#sns.scatterplot(premotor_inputs_inVNC['x'], premotor_inputs_inVNC['y'], color="cyan", edgecolor = None, ax = ax, alpha = 0.5)
sns.scatterplot(motor_inputs_inVNC['x'], motor_inputs_inVNC['y'], color="Blue", edgecolor = None, ax = ax, alpha = 0.4)
sns.scatterplot(sensory_outputs_inVNC['x'], sensory_outputs_inVNC['y'], color="Red", edgecolor = None, ax = ax, alpha = 0.4)

sns.scatterplot(descending_axons['x'], descending_axons['y'], color = 'white', ax = ax, marker ="+")
ax.invert_yaxis()
ax.set_facecolor('black')
plt.axis('equal')


#%%
'''
# old implementation
graph = sns.jointplot(test['x'], test['y'], color = 'black')

graph.x = MN_inputs['x']
graph.y = MN_inputs['y']
graph.plot_joint(sns.kdeplot, color = 'blue', alpha = 0.5)

graph.x = sens_outputs['x']
graph.y = sens_outputs['y']
graph.plot_joint(sns.kdeplot, color = 'red', alpha = 0.5)
'''
