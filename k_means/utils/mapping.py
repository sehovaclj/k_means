"""map parameters from message."""
from typing import Dict


class Parameters:
    """Contain message parameters.

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
        self.pause_length = message['PauseLength']
        self.seed = message['Seed']
