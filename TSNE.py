from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


class tSNE:
    def __init__(self, perplexity=10, T=1000, η=200, early_exaggeration=4, n_dimensions=2):
        self.perplexity = perplexity
        self.T = T
        self.η = η
        self.early_exaggeration = early_exaggeration
        self.n_dimensions = n_dimensions

    def fit_transform(self, X):
        n = len(X)
        p_ij = self._get_original_pairwise_affinities(X)
        p_ij_symmetric = self._get_symmetric_p_ij(p_ij)
        Y = np.zeros(shape=(self.T, n, self.n_dimensions))
        Y_minus1 = np.zeros(shape=(n, self.n_dimensions))
        Y[0] = Y_minus1
        Y1 = self._initialization(X)
        Y[1] = np.array(Y1)
        for t in range(1, self.T - 1):
            if t < 250:
                α = 0.5
                early_exaggeration = self.early_exaggeration
            else:
                α = 0.8
                early_exaggeration = 1
            q_ij = self._get_low_dimensional_affinities(Y[t])
            gradient = self._get_gradient(early_exaggeration * p_ij_symmetric, q_ij, Y[t])
            Y[t + 1] = Y[t] - self.η * gradient + α * (Y[t] - Y[t - 1])  
            if t % 50 == 0 or t == 1:
                cost = np.sum(p_ij_symmetric * np.log(p_ij_symmetric / q_ij))
                print(f"Iteration {t}: Value of Cost Function is {cost}")
        print(
            f"Completed Embedding: Final Value of Cost Function is {np.sum(p_ij_symmetric * np.log(p_ij_symmetric / q_ij))}"
        )
        solution = Y[-1]
        return solution, Y

    def _get_original_pairwise_affinities(self, X):
        n = len(X)
        print("Computing Pairwise Affinities....")
        p_ij = np.zeros(shape=(n, n))
        for i in range(0, n):
            diff = X[i] - X
            σ_i = self._grid_search(diff, i, self.perplexity)  
            norm = np.linalg.norm(diff, axis=1)
            p_ij[i, :] = np.exp(-norm ** 2 / (2 * σ_i ** 2))
            np.fill_diagonal(p_ij, 0)
            p_ij[i, :] = p_ij[i, :] / np.sum(p_ij[i, :])
        ε = np.nextafter(0, 1)
        p_ij = np.maximum(p_ij, ε)
        return p_ij

    def _grid_search(self, diff_i, i, perplexity):
        result = np.inf  
        norm = np.linalg.norm(diff_i, axis=1)
        std_norm = np.std(norm)  
        for σ_search in np.linspace(0.01 * std_norm, 5 * std_norm, 200):
            p = np.exp(-norm ** 2 / (2 * σ_search ** 2))
            p[i] = 0
            ε = np.nextafter(0, 1)
            p_new = np.maximum(p / np.sum(p), ε)            
            H = -np.sum(p_new * np.log2(p_new))
            if np.abs(np.log(perplexity) - H * np.log(2)) < np.abs(result):
                result = np.log(perplexity) - H * np.log(2)
                σ = σ_search
        return σ

    def _get_symmetric_p_ij(self, p_ij):
        n = len(p_ij)
        p_ij_symmetric = np.zeros(shape=(n, n))
        for i in range(0, n):
            for j in range(0, n):
                p_ij_symmetric[i, j] = (p_ij[i, j] + p_ij[j, i]) / (2 * n)
        ε = np.nextafter(0, 1)
        p_ij_symmetric = np.maximum(p_ij_symmetric, ε)
        return p_ij_symmetric

    def _initialization(self, X):
        return np.random.normal(loc=0, scale=1e-4, size=(len(X), self.n_dimensions))

    def _get_low_dimensional_affinities(self, Y):
        n = len(Y)
        q_ij = np.zeros(shape=(n, n))
        for i in range(0, n):
            diff = Y[i] - Y
            norm = np.linalg.norm(diff, axis=1)
            q_ij[i, :] = (1 + norm ** 2) ** (-1)
        np.fill_diagonal(q_ij, 0)
        q_ij = q_ij / q_ij.sum()
        ε = np.nextafter(0, 1)
        q_ij = np.maximum(q_ij, ε)

        return q_ij

    def _get_gradient(self, p_ij, q_ij, Y):
        n = len(p_ij)
        gradient = np.zeros(shape=(n, Y.shape[1]))
        for i in range(0, n):
            diff = Y[i] - Y
            A = np.array([(p_ij[i, :] - q_ij[i, :])])
            B = np.array([(1 + np.linalg.norm(diff, axis=1)) ** (-1)])
            C = diff
            gradient[i] = 4 * np.sum((A * B).T * C, axis=0)
        return gradient
