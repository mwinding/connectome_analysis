# %%
# location of scripts to plot each figure in brain connectome paper
import pandas as pd

figure_scripts = [[1, 'A', None, 'used Blender'],
                    [1, 'B', 'scripts/brain_completeness/brain_simple-synaptic-completeness.py', ['presynaptic site completion', 'postsynaptic site completion', 'all synaptic site completion', 'number of differentiated brain neurons, completed on left and right']],
                    [1, 'C', 'scripts/brain_completeness/brain_simple-synaptic-completeness.py', 'number of paired neurons and nonpaired'],
                    [1, 'D', None, 'used Illustrator'],
                    [1, 'E', None, 'used Illustrator'],
                    [1, 'F', 'scripts/small_plots/fraction_cell-types_brain.py', 'plot number brain inputs, interneurons, outputs'],
                    [1, 'G', 'scripts/small_plots/fraction_cell-types_brain.py', 'plot number brain inputs, interneurons, outputs'],
                    [1, 'H', 'scripts/small_plots/fraction_cell-types_brain.py', 'plot number brain inputs, interneurons, outputs'],
                    [2, 'A', '', ''],
                    [2, 'B', '', ''],
                    [2, 'C', '', ''],
                    [2, 'D', '', ''],
                    [2, 'E', '', ''],
                    [2, 'F', '', ''],
                    [2, 'G', '', ''],
                    [3, 'A', '', ''],
                    [3, 'B', '', ''],
                    [3, 'C', '', ''],
                    [3, 'D', '', ''],
                    [3, 'E', '', ''],
                    [3, 'F', '', ''],
                    [4, 'A', 'identify_neuron_classes/identify_sensory-ascending_neuropil.py', 'plot each type sequentially'],
                    [4, 'B', 'identify_neuron_classes/identify_sensory-ascending_neuropil.py', 'intersection between 2nd/3rd/4th neuropils'],
                    [4, 'C', 'identify_neuron_classes/identify_sensory-ascending_neuropil.py', 'known cell types per 2nd/3rd order'],
                    [4, 'D', None, None],
                    [4, 'E', 'cascades/cluster_cascades/sensory_cascades_through_clusters.py', 'plot signal of all sensories through clusters'],
                    [4, 'F', 'cascades/downstream_sensories_cascades.py', 'how close are descending neurons to sensory?'],
                    [4, 'G', 'cascades/cluster_cascades/sensory_cascades_integration_clusters.py', ''],
                    [4, 'H', 'cascades/cluster_cascades/sensory_cascades_integration_clusters.py', ''],
                    [4, 'I', 'cascades/cluster_cascades/sensory_cascades_integration_clusters.py', ''],
                    [4, 'J', 'cascades/cluster_cascades/sensory_cascades_integration_clusters.py', ''],
                    [5, 'A', None, None],
                    [5, 'B', 'synapse_distributions/synapse_dist_Br_ipsi-contra.py', 'all'],
                    [6, '', '', ''],
                    [7, '', '', '']]

figure_scripts = pd.DataFrame(figure_scripts, columns = ['figure', 'panel', 'path', 'chunk'])

sup_figure_scripts = [['S1', 'A', '', ''],
                        ['S2', 'A', 'scripts/VNC_interaction/ascending_analysis.py', 'identities of ascending neurons'],
                        ['S2', 'B', 'scripts/VNC_interaction/ascending_analysis.py', 'plotting neurons'],
                        ['S2', 'C', 'scripts/VNC_interaction/ascending_analysis.py', 'plotting neurons'],
                        ['S2', 'D', 'scripts/VNC_interaction/ascending_analysis.py', 'plotting neurons'],
                        ['S2', 'E', 'scripts/VNC_interaction/ascending_analysis.py', 'plotting neurons'],
                        ['S2', 'F', 'scripts/VNC_interaction/ascending_analysis.py', 'plotting neurons']]

sup_figure_scripts = pd.DataFrame(sup_figure_scripts, columns = ['figure', 'panel', 'path', 'chunk'])


# %%
