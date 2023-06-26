from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.utils as vutils


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


@dataclass
class ZeroShotKDHyperparams:
    learning_rate: float
    iterations: int
    batch_size: int
    temperature: float
    num_samples: int
    beta: List[float] = field(default_factory=list)


class ZeroShotKD:

    def __init__(
            self,
            teacher: torch.nn.Module,
            hyperparams: ZeroShotKDHyperparams,
            dimensions: Tuple[int, ...] = (1, 32, 32),
            num_classes: int = 10,
            normalize: bool = False
        ):
        self.dimensions = dimensions 
        self.num_classes = num_classes
        self.teacher = teacher
        self.hyperparams = hyperparams
        self.gen_num = 1
        self.normalize = normalize

    def synthesize(self, save_root: Path):

        save_root.parent.mkdir(parents=True, exist_ok=True)

        # lim_0, lim_1 = 2, 2
        file_num = np.zeros((self.num_classes), dtype=int)

        classifier_weights = self._extract_classifier_weights(self.teacher)
        cls_sim_norm = ClassSimilarityMatrix()(classifier_weights)
        cls_sim_norm = torch.clamp(cls_sim_norm, min=1e-6, max=1.0)
        
        self.teacher.eval()

        # generate synthesized images
        for k in range(self.num_classes):
            for beta in self.hyperparams.beta:
                concentration = beta * cls_sim_norm[k]
                N = (
                    self.hyperparams.num_samples 
                    // len(self.hyperparams.beta) 
                    // self.hyperparams.batch_size 
                    // self.num_classes
                )
                for n in range(N):

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

                    for iteration in (pbar := tqdm(range(self.hyperparams.iterations))):
                        optimizer.zero_grad()
                        output = F.softmax(self.teacher(inputs) / self.hyperparams.temperature, dim=1)

                        loss: torch.Tensor = F.binary_cross_entropy(output, y.detach())
                        loss.backward()
                        optimizer.step()
                        pbar.set_description(f"Loss: {loss:.4f}")

                    # save the synthesized images
                    labels = (
                        torch
                            .argmax(y, dim=1)
                            .detach()
                            .cpu()
                            .numpy()
                    )
                    self._save_synthesized_images(inputs, labels, file_num, save_root)

                    print(
                        'Generated {} synthesized images [{}/{}]'\
                        .format(
                            self.hyperparams.batch_size, 
                            self.hyperparams.batch_size*self.gen_num, 
                            self.hyperparams.num_samples
                        )
                    )

                    self.gen_num += 1
        return

    def _extract_classifier_weights(self, model: torch.nn.Module) -> torch.Tensor:
        """Assuming that the classifier is the last named parameter of a module, this function returns the weights of the classifier."""
        # Find last layer
        classifier = list(model.children())[-1]
        while 'Sequential' in str(classifier):
            classifier = list(classifier.children())[-1]

        # size(#class number, #weights in final-layer)
        return list(classifier.parameters())[0].cuda()
    
    def _save_synthesized_images(
            self, 
            inputs: torch.Tensor, 
            labels: torch.Tensor, 
            file_counts: np.ndarray,
            root_dir: Path
        ) -> None:

        for image, label in zip(inputs, labels):
            save_dir = root_dir / str(label)
            save_dir.mkdir(parents=True, exist_ok=True)

            vutils.save_image(
                tensor=image.detach().clone(), 
                fp=save_dir / f"{str(file_counts[label])}.jpg", 
                normalize=self.normalize
            )
            file_counts[label] += 1
        return
