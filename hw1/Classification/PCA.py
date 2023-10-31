import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None

    def fit(self, X):
        cov_matrix = np.cov(X, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        eigenvector_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[eigenvector_indices]
        eigenvectors = eigenvectors[:, eigenvector_indices]
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        return X.dot(self.components)