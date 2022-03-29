import pytest
from k_means.core.algorithm import k_means_algorithm


@pytest.mark.core_algorithm
def test_k_means_algorithm(example_data_eng):
    results = k_means_algorithm(parameters=example_data_eng['parameters'],
                                data_eng=example_data_eng['data_eng'])
    assert not not results
