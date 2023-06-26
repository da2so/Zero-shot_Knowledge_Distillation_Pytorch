import torch
import torch.nn.functional as F


def compute_class_similarity_matrix(weights: torch.Tensor) -> torch.Tensor:
    # normalize weights by one row at a time.
    norm_weights = F.normalize(weights, p=2, dim=1)
    # compute outer product
    outer_product = norm_weights @ norm_weights.T
    # Minmax normalization
    v_min = torch.min(outer_product, dim=1).values
    v_max = torch.max(outer_product, dim=1).values
    norm_outer_product = (outer_product - v_min) / (v_max - v_min)
    return norm_outer_product


class ClassSimilarityMatrix(torch.nn.Module):

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        # normalize weights by one row at a time.
        norm_weights = F.normalize(weights, p=2, dim=1)
        # compute outer product
        outer_product = norm_weights @ norm_weights.T
        # Minmax normalization
        v_min = torch.min(outer_product, dim=1).values
        v_max = torch.max(outer_product, dim=1).values
        norm_outer_product = (outer_product - v_min) / (v_max - v_min)
        return norm_outer_product