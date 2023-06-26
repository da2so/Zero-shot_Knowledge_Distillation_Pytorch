from pathlib import Path
from PIL import Image
import numpy as np

import torch

from architectures.resnet import ResNet34, ResNet18
from architectures.lenet import LeNet5, LeNet5Half
from architectures.resmlp import ResMLP
from torchvision.utils import save_image


def save_synthesized_images_labelwise( 
        inputs: torch.Tensor, 
        labels: torch.Tensor, 
        file_counts: np.ndarray,
        root_dir: Path,
        normalize: bool=False
    ) -> None:

    for image, label in zip(inputs, labels):
        save_dir = root_dir / str(label)
        save_dir.mkdir(parents=True, exist_ok=True)

        save_image(
            tensor=image.detach().clone(), 
            fp=save_dir / f"{str(file_counts[label])}.jpg", 
            normalize=normalize
        )
        file_counts[label] += 1
