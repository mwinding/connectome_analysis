#%%

import pandas as pd
import numpy as np
from pymaid_creds import url, name, password, token
import pymaid


rm = pymaid.CatmaidInstance(url, token, name, password)

def identify_open_ends(neuron):
    end_nodes = neuron.nodes[neuron.nodes.type=='end'].node_id.to_list()
    end_tags = neuron.tags['ends']

    tags = ['not a branch', 'uncertain continuation', 'uncertain end', 'gap']
    for tag in tags:
        try:
            nodes = neuron.tags[tag]
            end_tags = end_tags + nodes
        except:
            x=0
    
    open_ends = np.setdiff1d(end_nodes, list(np.intersect1d(end_tags, end_nodes)))
    return(open_ends)

# %%
# identify neurons with open-ends (0% review)

brain_unreviewed = pymaid.get_skids_by_annotation('mw brain unreviewed')

neurons = []
for skid in brain_unreviewed:
    neurons.append(pymaid.get_neurons(skid))

contains_open_ends_no_review=[]
for neuron in neurons:
    try: 
        open_ends = identify_open_ends(neuron)
        if(len(open_ends)>0):
            contains_open_ends_no_review.append(int(neuron.id))
    except:
        print(f'no ends at all? {neuron.id}')

pymaid.add_annotations(contains_open_ends_no_review, 'mw brain open-ends 1')

# partially reviewed neurons
brain_partially_reviewed = pymaid.get_skids_by_annotation('mw brain reviewed <80%')

neurons = []
chunks = list(np.arange(0, len(brain_partially_reviewed), 100)) + [len(brain_partially_reviewed)]
for i in np.arange(1, len(chunks)):
    neurons = pymaid.get_neurons(brain_partially_reviewed[chunks[i-1]:chunks[i]])

    contains_open_ends_partial_review=[]
    for neuron in neurons:
        try:
            open_ends = identify_open_ends(neuron)
            if(len(open_ends)>0):
                contains_open_ends_partial_review.append(int(neuron.id))
        except:
            print(f'no ends at all? {neuron.id}')

    pymaid.add_annotations(contains_open_ends_partial_review, 'mw brain open-ends 2')
    print(f'finished chunk {i}!')

# rest of brain
brain = pymaid.get_skids_by_annotation('mw brain neurons')
brain = np.setdiff1d(brain, brain_unreviewed + brain_partially_reviewed)

neurons = []
chunks = list(np.arange(0, len(brain), 25)) + [len(brain)]
for i in np.arange(1, len(chunks)):
    neurons = pymaid.get_neurons(brain[chunks[i-1]:chunks[i]])

    contains_open_ends_partial_review=[]
    for neuron in neurons:
        try:
            open_ends = identify_open_ends(neuron)
            if(len(open_ends)>0):
                contains_open_ends_partial_review.append(int(neuron.id))
        except:
            print(f'no ends at all? {neuron.id}')

    pymaid.add_annotations(contains_open_ends_partial_review, 'mw brain open-ends 3')
    print(f'finished chunk {i}!')
# %%
# 