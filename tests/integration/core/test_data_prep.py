import pytest

from k_means.core.data_prep import main_data_engineering


@pytest.mark.core_data_prep
def test_main_data_engineering(example_default_parameters):
    num_dists = example_default_parameters.num_dists
    data_eng, parameters = main_data_engineering(example_default_parameters)
    if example_default_parameters.add_noise:
        assert parameters.num_dists == num_dists + 1
    data_eng_attr = vars(data_eng)
    # essentially, we just need to assert that the values are available and not None
    assert not not data_eng_attr.values()
