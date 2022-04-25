import pytest

from k_means.resources.input import default_message
from k_means.utils.mapping import Parameters


@pytest.mark.utils_mapping
def test_parameters():
    parameters = Parameters(default_message)
    parameters_attr = vars(parameters)
    # essentially, we just need to assert that the values are available and not None
    assert not not parameters_attr.values()
