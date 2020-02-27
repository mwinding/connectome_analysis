import networkx as nx 
import pandas as pd
import connectome_tools.process_skeletons as proskel

path = 'data/test_skeleton.csv'
graph = proskel.skid_as_networkx_graph(path)
root = proskel.identify_root(graph)

connector_locations = pd.DataFrame(data=None, columns = ['nodeid', 'type', 'distance_root'])


for i in graph.nodes():
    print(i)
#print(nx.bidirectional_shortest_path(graph, 1867733, root))