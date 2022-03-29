import pytest
from k_means.utils.data_prep import create_random_dists, dists_as_matrix, dists_min_max, \
    initial_centroids, clusters_list, choose_colours


@pytest.mark.utils_data_prep
def test_create_random_dists(example_default_parameters):
    num_dists = example_default_parameters.num_dists
    x_dists, y_dists, parameters = create_random_dists(example_default_parameters)
    assert not not x_dists
    assert not not y_dists
    if example_default_parameters.add_noise:
        assert parameters.num_dists == num_dists + 1


@pytest.mark.utils_data_prep
def test_dists_as_matrix(example_default_parameters):
    x_dists, y_dists, parameters = create_random_dists(example_default_parameters)
    matrix_dists = dists_as_matrix(parameters.num_dists, x_dists, y_dists)
    # assert proper shape here == num_dists x num_samples x 2 (the 2 here represents the (x,y) coordinate pairing)
    assert matrix_dists.shape == (parameters.num_dists, parameters.num_samples, 2)


@pytest.mark.utils_data_prep
def test_dists_min_max(example_default_parameters):
    x_dists, y_dists, parameters = create_random_dists(example_default_parameters)
    boundaries = dists_min_max(x_dists, y_dists)
    assert not not boundaries


@pytest.mark.utils_data_prep
def test_initial_centroids(example_default_parameters):
    x_dists, y_dists, parameters = create_random_dists(example_default_parameters)
    boundaries = dists_min_max(x_dists, y_dists)
    centroids_prev = initial_centroids(parameters.num_clusters, boundaries)
    assert centroids_prev.shape == (parameters.num_clusters, 2)


@pytest.mark.utils_data_prep
def test_clusters_list(example_default_parameters):
    clusters = clusters_list(example_default_parameters.num_clusters)
    assert len(clusters) == example_default_parameters.num_clusters


@pytest.mark.utils_data_prep
def test_choose_colours(example_default_parameters):
    colours = choose_colours(example_default_parameters.num_clusters)
    assert colours.shape == (example_default_parameters.num_clusters, 1, 3)
