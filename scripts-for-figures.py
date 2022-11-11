# %%
# location of scripts to plot each figure in brain connectome paper
import pandas as pd

figure_scripts = [
    [1, 'A', None, 'used Blender'],
    [1, 'B', 'scripts/brain_completeness/brain_simple-synaptic-completeness.py', ['presynaptic site completion', 'postsynaptic site completion', 'all synaptic site completion', 'number of differentiated brain neurons, completed on left and right']],
    [1, 'C', 'scripts/brain_completeness/brain_simple-synaptic-completeness.py', 'number of paired neurons and nonpaired'],
    [1, 'D', None, 'used Illustrator'],
    [1, 'E', None, 'used Illustrator'],
    [1, 'F', 'scripts/small_plots/fraction_cell-types_brain.py', 'plot number brain inputs, interneurons, outputs'],
    [1, 'G', 'scripts/small_plots/fraction_cell-types_brain.py', 'plot number brain inputs, interneurons, outputs'],
    [1, 'H', 'scripts/small_plots/fraction_cell-types_brain.py', 'plot number brain inputs, interneurons, outputs'],
    [2, 'A', None, 'used Blender; numbers direct from CATMAID'],
    [2, 'B', 'axon_dendrite_split_analysis', 'plotting as normalized histogram distribution (note this script likely needs updating)'],
    [2, 'C', 'scripts/small_plots/general stats', 'quantification of different edge and synapse types'],
    [2, 'D', None, 'Plot from Benjamin Pedigo'],
    [2, 'E', 'scripts/small_plots/general stats', 'general stats'],
    [2, 'F', 'scripts/small_plots/feedforward-feedback_signalflow.py', 'seaborn barplots'],
    [2, 'G', None, 'Plot from Benjamin Pedigo'],
    [3, 'A', None, 'Plot from Benjamin Pedigo'],
    [3, 'B', None, 'Plot from Benjamin Pedigo'],
    [3, 'C', None, 'Plot from Benjamin Pedigo'],
    [3, 'D', None, 'Plot from Benjamin Pedigo'],
    [3, 'E', 'scripts/network_analysis/hubs_by_degree.py', 'location in cluster structure'],
    [3, 'F', 'scripts/network_analysis/hubs_by_degree.py (and modified manually)', 'plot cell type memberships of ad hubs'],
    [3, 'G', None, 'Plot from Benjamin Pedigo'],
    [4, 'A', 'identify_neuron_classes/identify_sensory-ascending_neuropil.py', 'plot each type sequentially'],
    [4, 'B', 'identify_neuron_classes/identify_sensory-ascending_neuropil.py', 'intersection between 2nd/3rd/4th neuropils'],
    [4, 'C', 'identify_neuron_classes/identify_sensory-ascending_neuropil.py', 'known cell types per 2nd/3rd order'],
    [4, 'D', None, 'used Illustrator'],
    [4, 'E', 'cascades/cluster_cascades/sensory_cascades_through_clusters.py', 'plot signal of all sensories through clusters (and used Illustrator)'],
    [4, 'F', 'cascades/downstream_sensories_cascades.py', 'how close are descending neurons to sensory?'],
    [4, 'G', 'cascades/cluster_cascades/sensory_cascades_integration_clusters.py', ''],
    [4, 'H', 'cascades/cluster_cascades/sensory_cascades_integration_clusters.py', ''],
    [4, 'I', 'cascades/cluster_cascades/sensory_cascades_integration_clusters.py', ''],
    [4, 'J', 'cascades/cluster_cascades/sensory_cascades_integration_clusters.py', ''],
    [5, 'A', None, 'Plot from Benjamin Pedigo'],
    [5, 'B', 'synapse_distributions/synapse_dist_Br_ipsi-contra.py', 'all'],
    [5, 'C', '', ''],
    [5, 'D', '', ''],
    [5, 'E', '', ''],
    [5, 'F', '', ''],
    [5, 'G', '', ''],
    [5, 'H', '', ''],
    [5, 'I', '', ''],
    [5, 'J', '', ''],
    [6, 'A', None, 'used Illustrator'],
    [6, 'B', '', ''],
    [6, 'C', '', ''],
    [6, 'D', '', ''],
    [6, 'E', '', ''],
    [6, 'F', '', ''],
    [6, 'G', '', ''],
    [6, 'H', '', ''],
    [6, 'I', '', ''],
    [6, 'J', '', ''],
    [6, 'K', '', ''],
    [7, 'A', '', ''],
    [7, 'B', '', ''],
    [7, 'C', '', ''],
    [7, 'D', '', ''],
    [7, 'E', '', ''],
    [7, 'F', '', ''],
    [7, 'G', '', ''],
    [7, 'H', '', ''],
    [7, 'I', '', ''],
    [7, 'J', '', ''],
    [7, 'K', '', '']
]

figure_scripts = pd.DataFrame(figure_scripts, columns = ['figure', 'panel', 'path', 'chunk'])

sup_figure_scripts = [
    ['S1', 'A', None, 'used Illustrator and Photoshop'],
    ['S1', 'B', None, 'used Illustrator'],
    ['S1', 'C', None, 'used Illustrator'],
    ['S1', 'D', None, 'used Illustrator'],
    ['S1', 'E', '', ''],
    ['S1', 'F', '', ''],
    ['S1', 'G', '', ''],
    ['S1', 'H', '', ''],
    ['S2', 'A', 'scripts/VNC_interaction/ascending_analysis.py', 'identities of ascending neurons'],
    ['S2', 'B', 'scripts/VNC_interaction/ascending_analysis.py', 'plotting neurons'],
    ['S2', 'C', 'scripts/VNC_interaction/ascending_analysis.py', 'plotting neurons'],
    ['S2', 'D', 'scripts/VNC_interaction/ascending_analysis.py', 'plotting neurons'],
    ['S2', 'E', 'scripts/VNC_interaction/ascending_analysis.py', 'plotting neurons'],
    ['S2', 'F', 'scripts/VNC_interaction/ascending_analysis.py', 'plotting neurons'],
    ['S3', 'A', None, 'used CATMAID and Illustrator'],
    ['S3', 'B', None, 'used CATMAID and Illustrator'],
    ['S3', 'C', 'scripts/synapse_distributions/synapse_dist_Br_outputs.py', 'used Blender'],
    ['S4', 'A', 'scripts/small_plots/fraction_cell-types_brain.py', 'plot interneurons with overlap allowed'],
    ['S4', 'B', None, 'used Illustrator'],
    ['S4', 'C', 'scripts/small_plots/fraction_cell-types_brain.py', 'plot morphology of each cell type'],
    ['S5', '', '', ''],
    ['S6', '', '', ''],
    ['S7', 'A', None, 'Plot from Benjamin Pedigo'],
    ['S7', 'B', 'small_plots/cluster-level-metrics.py', 'distribution of intracluster similarity'],
    ['S7', 'C', 'small_plots/cluster-level-metrics.py', 'load clusters and plot counts'],
    ['S7', 'D', 'scripts/cluster_analysis/general_celltype_classification.py', 'celltype plot'],
    ['S8', 'A', 'scripts/network_analysis/hubs_by_degree.py', 'location in cluster structure'],
    ['S8', 'B', 'scripts/network_analysis/hubs_by_degree.py', 'location in cluster structure'],
    ['S8', 'C', 'scripts/network_analysis/hubs_by_degree.py', 'location in cluster structure'],
    ['S8', 'D', 'scripts/network_analysis/hubs_by_degree.py', 'location in cluster structure'],
    ['S9', '', '', ''],
    ['S10', '', '', ''],
    ['S11', '', '', ''],
    ['S12', '', '', ''],
    ['S13', '', '', ''],
    ['S14', '', '', ''],
    ['S15', '', '', ''],
    ['S16', '', '', ''],
    ['S17', 'A', 'scripts/interhemisphere/contra_bilateral_loops_detail.py', 'cell types of loops'],
    ['S17', 'B', 'scripts/interhemisphere/contra_bilateral_loops_detail.py', 'location in cluster structure'],
    ['S17', 'C', 'scripts/interhemisphere/contra_bilateral_loops_detail.py', 'plot partners'],
    ['S17', 'D', 'scripts/interhemisphere/contra_bilateral_loops_detail.py', 'plot cell types'],
    ['S18', '', '', ''],
    ['S19', '', '', ''],
    ['S20', '', '', ''],
    ['S21', '', '', ''],
    ['S22', '', '', ''],
    ['S23', '', '', ''],
    ['S24', '', '', ''],
    ['S25', '', '', '']
]

sup_figure_scripts = pd.DataFrame(sup_figure_scripts, columns = ['figure', 'panel', 'path', 'chunk'])


# %%
