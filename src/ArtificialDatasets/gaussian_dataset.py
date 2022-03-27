import numpy as np

class GaussianDataset:
    def __init__(self, class_specs, num_samples, relative_probs=None):
        if relative_probs is None:
            self.relative_probs = [1/len(class_specs)]*len(class_specs)
        else:
            self.relative_probs = relative_probs
        self.num_datapoints = num_samples
        self.class_specs = class_specs
        class_indexes = np.random.choice(np.arange(len(class_specs)), num_samples, self.relative_probs)
        self.datapoints = []
        for n in range(len(self.class_specs)):
            num_samples_for_class = sum(class_indexes == n)
            gaussian_indexes = np.random.choice(np.arange(len(self.class_specs[n])), num_samples_for_class)
            for i in range(len(self.class_specs[n])):
                num_samples_for_gaussian = sum(gaussian_indexes == i)
                gaussian_datapoints = np.random.multivariate_normal(self.class_specs[n][i][0], self.class_specs[n][i][1], num_samples_for_gaussian)
                indexes = np.array([[n]*num_samples_for_gaussian])
                self.datapoints.append(np.concatenate([indexes.T, gaussian_datapoints], axis=1))
        self.datapoints = np.concatenate(self.datapoints)
        np.random.shuffle(self.datapoints)
