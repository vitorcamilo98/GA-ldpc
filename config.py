from dataclasses import dataclass, field
import numpy as np

@dataclass
class Config:
    # dimensões
    n: int = 64
    k: int = 32
    DEGREE = 3

    @property
    def m(self):
        return self.n - self.k

    # canal
    SNR_dBs: np.ndarray = field(default_factory=lambda: np.arange(0, 5, 1))

    # decoder
    MAX_ITERS: int = 50
    ALPHA: float = 1.0

    # GA
    POP_SIZE: int = 30
    MAX_GEN: int = 30
    MUT_RATE: float = 0.05
    CROSS_RATE: float = 0.5

    # avaliação
    SNR_FITNESS: float = 1.0
    NUM_BLOCKS: int = 500
