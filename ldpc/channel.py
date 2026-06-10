import numpy as np

def mapear_BPSK(bits):
    return 1 - 2 * bits

def awgn_noise(snr_db, size):
    snr = 10 ** (snr_db / 10)
    sigma = np.sqrt(1 / (2 * snr))
    return np.random.normal(0, sigma, size)

def calc_LLR(y, snr_db):
    snr = 10 ** (snr_db / 10)
    sigma2 = 1 / (2 * snr)
    return 2 * y / sigma2