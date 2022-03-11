"""map parameters from message."""


def map_parameters_from_message(message):
    num_clusters = message['NumberClusters']
    num_dists = message['NumberDistributions']
    num_samples = message['NumberSamples']
    eps = message['EpsilonForConvergence']
    max_iter = message['MaxIterations']
    add_noise = message['AddNoise']
    pause_length = message['PauseLength']
    seed = message['Seed']
    return num_clusters, num_dists, num_samples, eps, max_iter, add_noise, pause_length, seed
