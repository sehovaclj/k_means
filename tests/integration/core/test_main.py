import pytest

from k_means.core.app import run
from k_means.resources.input import default_message


@pytest.mark.core_main
def test_main():
    default_message['ShowPlots'] = False
    run(default_message)
    assert True
