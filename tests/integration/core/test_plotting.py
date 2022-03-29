import pytest
from k_means.core.plotting import plot_simulation


@pytest.mark.core_plotting
def test_plot_simulation(example_results):
    example_results['parameters'].show_plots = False
    plot_simulation(parameters=example_results['parameters'],
                    data_eng=example_results['data_eng'],
                    results=example_results['results'])
    assert True
