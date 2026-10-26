"""Compact semantic projection and trace descriptors for tree rankers."""
import numpy as np

from features import descriptor_features


def fit_projection(arrays, dimensions=96, seed=7536):
    from sklearn.decomposition import PCA
    embeddings = arrays['embeddings'].reshape(-1, 1024).astype(np.float32)
    estimator = PCA(n_components=dimensions, svd_solver='randomized',
                    random_state=seed).fit(embeddings)
    return {'mean': estimator.mean_.astype(np.float32),
            'components': estimator.components_.astype(np.float32)}


def tree_features(bank, arrays, projection):
    embeddings = arrays['embeddings'].astype(np.float32)
    semantic = ((embeddings.reshape(-1, 1024) - projection['mean']) @
                projection['components'].T).reshape(len(bank), 4, -1)
    mean = semantic.mean(axis=1, keepdims=True)
    descriptors = descriptor_features(bank, arrays)
    return np.concatenate([semantic, semantic - mean, semantic * mean, descriptors],
                          axis=2).reshape(len(bank) * 4, -1).astype(np.float32)
