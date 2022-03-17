import pymaid as pymaid
#from pymaid_creds import url, name, password, token
import navis.interfaces.blender as b3d
import numpy as np

pymaid.CatmaidInstance(url, token, name, password)
h = b3d.Handler()

cns = pymaid.get_volume('cns')
neuropil = pymaid.get_volume('PS_Neuropil_manual')
h.add(cns)
h.add(neuropil)

skids = pymaid.get_skids_by_annotation('mw CN')
neurons = pymaid.get_neurons(skids)
h.add(neurons)
h.neurons.bevel(.02) # change neuron thickness

# must manually remove pre- and postsynaptic sites in blender



# use the below code if issues loading all neurons as a group

#neurons_loaded = pymaid.get_neurons(skids[0])
#for j in range(1, len(skids)):
#    loaded = pymaid.get_neurons(skids[j])
#    neurons_loaded = neurons_loaded + loaded

#name = 'CNs'
#skids = pymaid.get_skids_by_annotation(f'mw exclusive-celltype {name}')
skids = pymaid.get_skids_by_annotation('mw dVNC')
neurons = pymaid.get_neurons(skids)
h.add(neurons)
h.neurons.bevel(0.02)
 
