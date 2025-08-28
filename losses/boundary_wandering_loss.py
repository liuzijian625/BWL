
import torch
import torch.nn.functional as F

def boundary_wandering_loss(embeddings, labels):
    """
    Calculates the Boundary-Wandering Loss.

    Args:
        embeddings (torch.Tensor): The embeddings from the shadow model (E_shadow).
        labels (torch.Tensor): The corresponding labels.

    Returns:
        torch.Tensor: The Boundary-Wandering Loss.
    """
    # Normalize embeddings to the unit hypersphere
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

    unique_labels = torch.unique(labels)
    total_loss = 0.0
    num_pairs = 0

    for label in unique_labels:
        same_class_embeddings = normalized_embeddings[labels == label]
        if len(same_class_embeddings) > 1:
            # Calculate pairwise cosine similarity
            cosine_similarity = torch.mm(same_class_embeddings, same_class_embeddings.t())
            # Sum of cosine similarities for unique pairs (upper triangle)
            loss = torch.sum(torch.triu(cosine_similarity, diagonal=1))
            total_loss += loss
            num_pairs += len(same_class_embeddings) * (len(same_class_embeddings) - 1) / 2

    if num_pairs == 0:
        return torch.tensor(0.0, device=embeddings.device)

    return total_loss / num_pairs
