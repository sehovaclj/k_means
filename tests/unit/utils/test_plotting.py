import matplotlib
import pytest
from k_means.utils.plotting import scatter_plot_initial_dists, scatter_plot_initial_dists_no_colour, \
    plot_initial_centroids, append_to_scatter_plot, plot_new_centroids, clear_old_clusters_and_plot_new_ones


@pytest.mark.utils_plotting
def test_scatter_plot_initial_dists(example_data_eng):
    scatter_plot_initial_dists(parameters=example_data_eng['parameters'],
                               data_eng=example_data_eng['data_eng'])
    assert True


@pytest.mark.utils_plotting
def test_scatter_plot_initial_dists_no_colour(example_data_eng):
    plots = scatter_plot_initial_dists_no_colour(parameters=example_data_eng['parameters'],
                                                 data_eng=example_data_eng['data_eng'])
    assert not not plots


@pytest.mark.utils_plotting
def test_plot_initial_centroids(example_data_eng):
    plot_cens = plot_initial_centroids(example_data_eng['data_eng'])
    assert type(plot_cens) == matplotlib.collections.PathCollection


@pytest.mark.utils_plotting
def test_append_to_scatter_plot(example_results_iter):
    dict_to_append = append_to_scatter_plot(results_iter=example_results_iter, i=0)
    assert not not dict_to_append


@pytest.mark.utils_plotting
def test_plot_new_centroids(example_results_iter):
    plot_cens = plot_new_centroids(example_results_iter)
    assert type(plot_cens) == matplotlib.collections.PathCollection


@pytest.mark.utils_plotting
def test_clear_old_clusters_and_plot_new_ones(example_data_eng, example_results_iter):
    plots = scatter_plot_initial_dists_no_colour(parameters=example_data_eng['parameters'],
                                                 data_eng=example_data_eng['data_eng'])
    plots = clear_old_clusters_and_plot_new_ones(parameters=example_data_eng['parameters'],
                                                 plots=plots,
                                                 results_iter=example_results_iter)
    assert not not plots
