
import numpy as np

def partition_features(features, private_ratio=0.3):
    """
    Partitions features into public and private sets.

    Args:
        features (np.ndarray): The feature matrix for one party.
        private_ratio (float): The ratio of features to be designated as private.

    Returns:
        tuple: (public_feature_indices, private_feature_indices)
    """
    num_features = features.shape[1]
    
    # For simplicity, we'll use random partitioning here.
    # A real implementation could use feature importance or mutual information.
    num_private_features = int(num_features * private_ratio)

    indices = np.arange(num_features)
    np.random.shuffle(indices)

    private_indices = indices[:num_private_features]
    public_indices = indices[num_private_features:]

    return public_indices, private_indices
