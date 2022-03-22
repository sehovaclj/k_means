import pytest
import numpy as np
from k_means.resources.input import default_message
from k_means.utils.mapping import Parameters
from k_means.core.data_prep import main_data_engineering


@pytest.fixture(scope='module')
def example_default_parameters():
    return Parameters(default_message)


@pytest.fixture(scope='module')
def example_data_eng():
    data_eng, parameters = main_data_engineering(Parameters(default_message))
    return {
        'data_eng': data_eng,
        'parameters': parameters
    }


@pytest.fixture(scope='module')
def example_results_iter():
    data_eng, parameters = main_data_engineering(Parameters(default_message))
    for i in range(parameters.num_clusters):
        data_eng.clusters[i].append((0, 0))
    counter = -1
    results_iter = {
        'iteration': counter,
        'cluster_plots': [],
        'new_centroids': {}
    }
    for i in range(parameters.num_clusters):
        results_iter['cluster_plots'].append({
            'plot_x_' + str(i + 1): np.array(data_eng.clusters[i])[:, 0],
            'plot_y_' + str(i + 1): np.array(data_eng.clusters[i])[:, 1],
            'plot_c_' + str(i + 1): data_eng.colours[i],
            'plot_label_' + str(i + 1): 'Cluster ' + str(i + 1)
        })
    # add new centroids to results
    results_iter['new_centroids']['x'] = data_eng.centroids_new[:, 0]
    results_iter['new_centroids']['y'] = data_eng.centroids_new[:, 1]
    results_iter['new_centroids']['c'] = 'r'
    results_iter['new_centroids']['marker'] = '*'
    results_iter['new_centroids']['s'] = 200
    results_iter['new_centroids']['label'] = 'Centroids Iter ' + str(counter + 1)
    return results_iter
