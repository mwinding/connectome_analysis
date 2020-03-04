import networkx as nx 
from networkx.utils import pairwise
import pandas as pd
import numpy as np
import connectome_tools.process_skeletons as proskel
import math
import matplotlib.pyplot as plt
import seaborn as sns

# import and store skeleton morphology data as graph
skeleton_csv = pd.read_csv('data/test_skeleton.csv', header=0, skipinitialspace=True, keep_default_na = False)
graph = proskel.skid_as_networkx_graph(skeleton_csv)
root = proskel.identify_root(graph)

connectors = pd.read_csv('data/test_connectors.csv', header=0, skipinitialspace=True, keep_default_na = False)
connectdists = proskel.connector_dists(graph, connectors, root)


inputs = []
outputs = []
for i in range(0, len(connectdists)):
    if(connectdists[i]['type']=='postsynaptic'):
        inputs.append(connectdists[i]['distance_root'])
    if(connectdists[i]['type']=='presynaptic'):
        outputs.append(connectdists[i]['distance_root'])


fig, ax = plt.subplots(1,1,figsize=(8,4))
#sns.distplot(data = inputs, ax = ax, )
#sns.distplot(data = outputs, ax = ax, )

ax.hist(outputs, density = True)
ax.hist(inputs, density = True)
plt.show()

'''
fig, axs = plt.subplots(2,1,figsize=(8,4))
axs[0].hist(inputs)
axs[1].hist(outputs)
fig.suptitle("text")
plt.show()
'''


'''
# test case for calculating total distance between two treenode ids
path = nx.bidirectional_shortest_path(graph, 1867733, root)
path = list(pairwise(path))

dist = []
for i in path:
    p1 = np.array([graph.nodes(data=True)[i[0]]['x'], graph.nodes(data=True)[i[0]]['y'], graph.nodes(data=True)[i[0]]['z']])
    p2 = np.array([graph.nodes(data=True)[i[1]]['x'], graph.nodes(data=True)[i[1]]['y'], graph.nodes(data=True)[i[1]]['z']])

    dist.append(np.linalg.norm(p1-p2))
    
total_dist = np.sum(dist)
print("from source: %d to root: %d is %d nm" % (1867733, root, total_dist))
'''