"""Module to contain functions used in core algorithm."""
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from typing import List, Dict


def scatter_plot_initial_dists(parameters,
                               data_eng) -> None:
    plt.figure(1)
    # add points to scatter plot
    for i in range(parameters.num_dists):
        if parameters.add_noise:
            if i != parameters.num_dists - 1:
                plt.scatter(data_eng.x_dists[i], data_eng.y_dists[i], label='Dist ' + str(i + 1))
            elif i == parameters.num_dists - 1:
                plt.scatter(data_eng.x_dists[i], data_eng.y_dists[i], label='Noise Dist')
        elif not parameters.add_noise:
            plt.scatter(data_eng.x_dists[i], data_eng.y_dists[i], label='Dist ' + str(i + 1))
    plt.title('Initial distributions/samples in colour')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()


def scatter_plot_initial_dists_no_colour(parameters,
                                         data_eng):
    plots = []
    for i in range(parameters.num_dists):
        plots.append({'plot_' + str(i + 1): plt.scatter(
            data_eng.x_dists[i],
            data_eng.y_dists[i],
            c='k')})
    plt.title('K-means Clustering')
    plt.xlabel('x')
    plt.ylabel('y')
    return plots


def plot_initial_centroids(data_eng):
    plot_cens = plt.scatter(data_eng.initial_centroids[:, 0],
                            data_eng.initial_centroids[:, 1],
                            c='r', marker='*', s=200, label='Initial Centroids')
    leg_cens = plt.legend()
    plt.xlim(left=data_eng.boundaries['min_x'] - 1.0,
             right=data_eng.boundaries['max_x'] + 1.0)
    plt.ylim(bottom=data_eng.boundaries['min_y'] - 1.0,
             top=data_eng.boundaries['max_y'] + 1.0)
    return plot_cens, leg_cens


def append_to_scatter_plot(results_iter,
                           i):
    return {'plot_' + str(i + 1): plt.scatter(
        results_iter['cluster_plots'][i]['plot_x_' + str(i + 1)],
        results_iter['cluster_plots'][i]['plot_y_' + str(i + 1)],
        c=results_iter['cluster_plots'][i]['plot_c_' + str(i + 1)],
        label=results_iter['cluster_plots'][i]['plot_label_' + str(i + 1)])}


def plot_new_centroids(results_iter):
    plot_cens = plt.scatter(
        results_iter['new_centroids']['x'],
        results_iter['new_centroids']['y'],
        c=results_iter['new_centroids']['c'],
        marker=results_iter['new_centroids']['marker'],
        s=results_iter['new_centroids']['s'],
        label=results_iter['new_centroids']['label'])
    return plot_cens


def choose_colours(num_clusters: int) -> np.array:
    colours = np.zeros([num_clusters, 1, 3])
    for i in range(num_clusters):
        colours[i] = [random.random(), random.random(), random.random()]
    return colours
