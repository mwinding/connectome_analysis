import pymaid as pymaid
from pymaid_creds import url, name, password, token
import navis.interfaces.blender as b3d

pymaid.CatmaidInstance(url, token, name, password)

cns = pymaid.get_volume('cns')
neuropil = pymaid.get_volume('neuropil')

h = b3d.Handler()
h.add(cns)
h.add(neuropil)

'''
import pandas as pd 
import navis
import navis.interfaces.blender as b3d
from os import listdir
from os.path import isfile, join

mypath = '/Users/mwinding/Downloads/catmaid-swc-export(1)'
paths = [f for f in listdir(mypath) if isfile(join(mypath, f))]
paths = [mypath + '/'+ x for x in paths]

neurons = []
for i in range(len(paths)):
    neuron = pd.read_csv(paths[i], sep=' ', header=None)
    neuron.columns = ['node_id', 'id', 'y', 'z', 'x', 'radius', 'parent_id']
    neuron = navis.TreeNeuron(neuron)
    neurons.append(neuron)

h = b3d.Handler()
for neuron in neurons:
    h.add(neuron)
'''