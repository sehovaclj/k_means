import pytest

from k_means.utils.algorithm import calculate_distances, find_new_centroids, check_for_convergence


@pytest.mark.utils_algorithm
def test_calculate_distances(example_data_eng):
    data_eng = calculate_distances(parameters=example_data_eng['parameters'],
                                   data_eng=example_data_eng['data_eng'])
    data_eng_attr = vars(data_eng)
    for key, value in data_eng_attr.items():
        assert data_eng_attr[key] is not None


# this unit test needs to be before the test_find_new_centroids unit test because the latter will alter the centroids
# and then this test will assert convergence=False. Since this test does not alter the centroids in any way, it is
# okay to have this test first.
@pytest.mark.utils_algorithm
def test_check_for_convergence(example_data_eng):
    data_eng, convergence = check_for_convergence(parameters=example_data_eng['parameters'],
                                                  data_eng=example_data_eng['data_eng'],
                                                  counter=0)
    # convergence will be true because to start, we set new centroids == to previous, so convergence will be True,
    # hence this unit tests the epsilon if statement block
    assert convergence


@pytest.mark.utils_algorithm
def test_find_new_centroids(example_data_eng):
    data_eng = find_new_centroids(parameters=example_data_eng['parameters'],
                                  data_eng=example_data_eng['data_eng'])
    data_eng_attr = vars(data_eng)
    for key, value in data_eng_attr.items():
        assert data_eng_attr[key] is not None


@pytest.mark.utils_algorithm
def test_append_to_results(example_results_iter):
    # no need to test anything here, since the example_results_iter is essentially this unit test.
    # If there is anything wrong with example_results_iter, conftest.py will let us know.
    assert True
