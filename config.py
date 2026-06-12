from dataclasses import dataclass, field
import numpy as np

@dataclass
class Config:
    # dimensões
    Z: int = 8
    Mb: int = 4
    Nb: int = 8


    @property
    def m(self):
        return self.Mb - self.Z
    
    @property
    def n(self):
        return self.Nb - self.Z

    @property
    def k(self):
        return self.n - self.m

    @property
    def rate(self):
        return self.k / self.n

    # degree profile 
    dv: int = 3
    dc: int = 6

    # canal
    SNR_dBs: np.ndarray = field(default_factory=lambda: np.arange(0, 6.5, 0.5))

    # decoder
    MAX_ITERS: int = 50
    ALPHA: float = 1.0

    # GA
    POP_SIZE: int = 50
    MAX_GEN: int = 100
    MUT_RATE: float = 0.05
    CROSS_RATE: float = 0.7
    ELITE_FRAC: float = 0.10
    TOURNAMENT_K: int = 3

    #fitness weights
    W_BLER: float = 0.60
    W_GIRTH: float = 0.25
    W_CYCLES: float = 0.15
    GIRTH_MAX: int = 12

    # avaliação
    SNR_FITNESS: float = 2.0
    NUM_BLOCKS: int = 1000

    # reprodução
    SEED: int = 42