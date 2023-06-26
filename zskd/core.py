from typing import Iterator, Tuple

from tqdm import tqdm
import torch
import torch.nn.functional as F

from zskd.class_sim_matrix import compute_class_similarity_matrix
from zskd.hyperparams import ZeroShotKDHyperparams
from zskd.classifier_weights import extract_classifier_weights


class ZeroShotKDClassification:
    """Zero Shot Knowledge Distillation for a classification task
    input:
    teacher - the pre-trained model that we want to extract a dataset from
    hyperparams - ZeroShotKDHyperparams
    dimensions - shape of the generated images
    num_classes - Total number of classes
    """
    
    def __init__(
            self,
            teacher: torch.nn.Module,
            hyperparams: ZeroShotKDHyperparams,
            dimensions: Tuple[int, ...] = (1, 32, 32),
            num_classes: int = 10,
        ) -> None:

        self.dimensions = dimensions 
        self.num_classes = num_classes
        self.teacher = teacher
        self.hyperparams = hyperparams

    def synthesize(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """A Generator function that returns the optimized input value and the sampled dirichlet y values batchwise
        returns: 
        (1) x - Tensor of shape (BATCH_SIZE, *dimensions)
        (2) y - Tensor of shape (BATCH_SIZE, num_classes)
        """

        # Get Classifier Weights
        classifier_weights = extract_classifier_weights(self.teacher)
        class_similarity_matrix = (
            compute_class_similarity_matrix(classifier_weights)
                .clamp(min=1e-6, max=1.0)
        )

        # Generate Synthetic Images
        for label in range(self.num_classes):
            for beta in self.hyperparams.beta:
                concentration = beta * class_similarity_matrix[label]

                # Samples per label, batch and beta
                N = (
                    self.hyperparams.num_samples 
                    / len(self.hyperparams.beta) 
                    / self.hyperparams.batch_size 
                    / self.num_classes
                )
                assert N.is_integer()  # Divisibility Check

                N = int(N)

                for _ in range(N):
                   # sampling target label from Dirichlet distribution
                    dirichlet_distribution = (
                        torch
                            .distributions
                            .Dirichlet(concentration)
                    )

                    y: torch.Tensor = (
                        dirichlet_distribution
                            .rsample((self.hyperparams.batch_size,)) 
                    )

                    # optimization for images
                    inputs = (
                        torch
                            .randn(size=(self.hyperparams.batch_size, *self.dimensions))
                            .cuda()
                            .requires_grad_()
                    )
                    optimizer = torch.optim.Adam([inputs], self.hyperparams.learning_rate)
                    self.teacher.eval()
                    for iteration in (pbar := tqdm(range(self.hyperparams.iterations))):
                        output = F.softmax(self.teacher(inputs) / self.hyperparams.temperature, dim=1)
                        loss: torch.Tensor = F.binary_cross_entropy(output, y.detach())
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        pbar.set_description(f"Loss: {loss:.4f}")
                    
                    yield inputs, y.detach()