import pytest
from k_means.resources.input import default_message


@pytest.mark.default_message
def test_default_message():
    assert type(default_message['NumberClusters']) == int and default_message['NumberClusters'] > 0
    assert type(default_message['NumberDistributions']) == int and default_message['NumberDistributions'] > 0
    assert type(default_message['NumberSamples']) == int and default_message['NumberSamples'] > 0
    assert type(default_message['EpsilonForConvergence']) == float and \
           0 < default_message['EpsilonForConvergence'] < 1
    assert type(default_message['MaxIterations']) == int and default_message['MaxIterations'] > 0
    assert type(default_message['AddNoise']) == bool
    assert type(default_message['PauseLength']) == float and default_message['PauseLength'] > 0
    assert type(default_message['Seed']) == int and default_message['Seed'] > 0
    return True
