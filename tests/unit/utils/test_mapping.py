import pytest
from k_means.utils.mapping import Parameters
from k_means.resources.input import default_message


@pytest.mark.utils_mapping
def test_Parameters():
    parameters = Parameters(default_message)
    parameters_attr = vars(parameters)
    assert not not all(parameters_attr.values())
