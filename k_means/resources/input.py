"""Module that contains the default input parameters. These can be altered at the users desired.

Keys:
    NumberClusters: desired number of clusters to find.
    NumberDistributions: desired number of initial distributions.
    NumberSamples: desired number of samples per distribution.
    EpsilonForConvergence: if the centroids have not moved a distance greater than this parameter,
        we have reached convergence.
    MaxIterations: if we can not seem to find the centroids until convergence,
        wherever the centroids are at this parameter, stop there and use those centroids as final.
    AddNoise: if True, add a random noise distribution.
    PauseLength: for plotting, when we simulate how we found the final clusters and centroids,
        how long do we want to pause the plot before we continue to the next iteration.
    Seed: use the same value to re-run the same simulation, helps preserve the randomness state.

"""

default_message = {
    "NumberClusters": 3,
    "NumberDistributions": 5,
    "NumberSamples": 200,
    "EpsilonForConvergence": 0.01,
    "MaxIterations": 15,
    "ShowPlots": True,
    "AddNoise": True,
    "PauseLength": 0.5,
    "Seed": 11
}
