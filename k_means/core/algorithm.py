"""Module to run the main k-means algorithm."""
import warnings
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from math import sqrt
from k_means.utils.mapping import map_parameters_from_message
from k_means.core.data_prep import main_data_engineering
from k_means.core.plotting import main_initial_plotting

# we can suppress this warning since we are not threading and plotting will be okay
warnings.filterwarnings("ignore", message="Starting a Matplotlib GUI outside of the main thread will likely fail.")


def main_k_means_algorithm(message):
    # get parameters from post request message
    # TODO: MAKE THIS INTO A CLASS !!!
    num_clusters, num_dists, num_samples, eps, max_iter, add_noise, pause_length, seed \
        = map_parameters_from_message(message)
    # preserving random state
    np.random.seed(seed)
    # main data engineering is first
    x_dists, y_dists, num_dists, matrix_dists, boundaries, centroids_prev = \
        main_data_engineering(num_clusters, num_dists, num_samples, add_noise)
    # plot initial dists and centroids
    plots, plot_centroids, legend_centroids, colours = \
        main_initial_plotting(num_clusters, num_dists, add_noise, x_dists, y_dists, pause_length, centroids_prev,
                              boundaries)

    import pdb
    pdb.set_trace()

    ###################################################################################################################

    # K-means algorithm

    # create empty lists of clusters
    Clusters = []
    for i in range(num_clusters):
        Clusters.append([])

    # Run algorithm until convergence
    counter = 0
    convergence = False
    while not convergence:

        for N in range(num_dists):
            for n in range(num_samples):
                distances = []
                # for each sample, compute distance to each centroid
                for c in range(num_clusters):
                    distances.append(
                        sqrt((A[N][n][0] - centroids_prev[c][0]) ** 2 + (A[N][n][1] - centroids_prev[c][1]) ** 2))
                # find smallest distance
                C_idx = np.array(distances).argmin()
                # append sample, or point, to that cluster
                Clusters[C_idx].append(A[N][n])

            # plot clusters
        plots = None
        plots = []
        for i in range(num_clusters):
            plots.append({'plot_' + str(i + 1): plt.scatter(np.array(Clusters[i])[:, 0], np.array(Clusters[i])[:, 1],
                                                            c=colours[i], label='Cluster ' + str(i + 1))})
        leg_clusters = plt.legend()
        # show plot, and hold for desired timeframe
        plt.pause(pause_length)

        # compute mean of cluster -- and make this our new centroid for that cluster
        centroids_new = centroids_prev.copy()  # creating new variable here, need it for convergence tests
        for i in range(num_clusters):
            centroids_new[i] = [np.array(Clusters[i])[:, 0].mean(), np.array(Clusters[i])[:, 1].mean()]

        # remove old centroids and legend
        plot_cens.remove()
        leg_cens.remove()

        # update iter number
        counter += 1

        # plot new centroids
        plot_cens = plt.scatter(centroids_new[:, 0], centroids_new[:, 1], c='r', marker='*', s=200,
                                label='Centroids Iter ' + str(counter))
        leg_cens = plt.legend()

        plt.pause(pause_length)

        # Now need to test for convergence. Easiest way, also saves most memory, is to test if the centroids distance from the
        # previous centroid is less than epsilon (usually a small value). Hence, if the centroid has not moved or barely moved, the desired number of clusters has been obtained
        distances_bool = []

        for i in range(num_clusters):
            dist_cen = sqrt(
                (centroids_new[i][0] - centroids_prev[i][0]) ** 2 + (centroids_new[i][1] - centroids_prev[i][1]) ** 2)
            if dist_cen <= eps:
                distances_bool.append(True)
            elif dist_cen > eps:
                distances_bool.append(False)

        # if all distances are under a certain eps threshold, end algorithm;
        # if not, assign new centroids as previous, and clear clusters from plots
        if all(distances_bool):
            print('\nConvergence achieved in {} iterations'.format(counter))
            convergence = True
        elif not all(distances_bool):
            centroids_prev = centroids_new
            # clear old clusters from plot (figure 2)
            for i in range(num_clusters):
                plots[i]['plot_' + str(i + 1)].remove()
            # Empty Clusters and create new empty ones
            Clusters = None
            Clusters = []
            for i in range(num_clusters):
                Clusters.append([])
            leg_clusters.remove()

        # also end the algorithm if the maximum number of iterations is reached
        if counter == max_iter:
            print('Convergence timed out by Maximum number of iterations')
            convergence = True

    # return desired variables

    return A, Clusters, centroids_new
    return message
