import numpy as np
from sklearn.metrics import pairwise_distances

class TSNE:
    def __init__(self, n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

    def fit_transform(self, X):
        np.random.seed(self.random_state)
        similarities = self._compute_pairwise_similarities(X)
        Y = self._optimize(X, similarities)
        return Y

    def _compute_pairwise_similarities(self, X):
        n_samples = X.shape[0]
        distances = pairwise_distances(X, metric='euclidean')
        similarities = 1 / (1 + distances)
        np.fill_diagonal(similarities, 0.0)
        return similarities
    def _optimize(self, X, similarities):
        n_samples = X.shape[0]
        Y = np.random.normal(0, 0.0001, (n_samples, self.n_components))
        for _ in range(self.n_iter):
            distances = pairwise_distances(Y, metric='euclidean')
            inv_distances = 1.0 / (1.0 + distances)
            P = inv_distances / np.sum(inv_distances)
            Q = similarities / np.sum(similarities)
            gradients = 4 * (P - Q) * inv_distances
            Y -= self.learning_rate * np.dot(gradients, Y)
        return Y
