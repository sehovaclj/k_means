"""map parameters from message."""
from typing import Dict


class Parameters:
    """Contains parameters.
    Used to organize and clean up the code. Easier to pass one class with many attributes to functions
    than to constantly pass 8 parameters.

    Args:
        message: input dict containing parameters.
            see k_means.resources.input for the default message and parameters.

    """

    def __init__(self,
                 message: Dict[str, any]):
        self.message = message
        self.num_clusters = message['NumberClusters']
        self.num_dists = message['NumberDistributions']
        self.num_samples = message['NumberSamples']
        self.eps = message['EpsilonForConvergence']
        self.max_iter = message['MaxIterations']
        self.add_noise = message['AddNoise']
        self.show_plots = message['ShowPlots']
        self.pause_length = message['PauseLength']
        self.seed = message['Seed']
