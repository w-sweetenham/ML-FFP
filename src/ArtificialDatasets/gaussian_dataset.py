import numpy as np

class GaussianDataset:
    def __init__(self, cluster_specs, num_samples, relative_probs=None):
        if relative_probs is None:
            self.relative_probs = [1/len(cluster_specs)]*len(cluster_specs)
        else:
            self.relative_probs = relative_probs
        self.num_datapoints = num_samples
        self.cluster_specs = cluster_specs
        cluster_indexes = np.random.choice(np.arange(len(cluster_specs)), num_samples, self.relative_probs)
        self.datapoints = []
        for n in range(len(self.cluster_specs)):
            num_samples = sum(cluster_indexes == n)
            samples = np.random.multivariate_normal(self.cluster_specs[n][0], self.cluster_specs[n][1], num_samples)
            indexes = np.array([[n]*num_samples])
            self.datapoints.append(np.concatenate([indexes.T, samples], axis=1))
        self.datapoints = np.concatenate(self.datapoints)
        np.random.shuffle(self.datapoints)
