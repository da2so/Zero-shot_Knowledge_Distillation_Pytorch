from dataclasses import dataclass, field
from typing import List


@dataclass
class ZeroShotKDHyperparams:
    learning_rate: float
    iterations: int
    batch_size: int
    temperature: float
    num_samples: int
    beta: List[float] = field(default_factory=list)