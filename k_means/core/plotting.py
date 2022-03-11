"""Module to help organize main algorithm."""
import numpy as np
from k_means.utils.plotting import plot_initial_dists, add_centroids_to_plots, \
    remove_plots, choose_colours
from typing import List, Dict


def main_initial_plotting(num_clusters: int,
                          num_dists: int,
                          add_noise: bool,
                          x_dists: list,
                          y_dists: list,
                          pause_length: float,
                          centroids_prev: np.array,
                          boundaries: Dict[str, float]):
    # plot the initial random distributions
    plots = plot_initial_dists(num_dists, add_noise, x_dists, y_dists, pause_length)
    # add centroids to colorless plot
    plot_centroids, legend_centroids = add_centroids_to_plots(centroids_prev, boundaries, pause_length)
    # remove plots from figure
    plots = remove_plots(num_dists, plots)
    # colours for clusters. Randomly chosen. This is for plotting
    colours = choose_colours(num_clusters)
    return plots, plot_centroids, legend_centroids, colours
