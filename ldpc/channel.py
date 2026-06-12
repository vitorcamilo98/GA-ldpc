import numpy as np

def mapear_BPSK(bits):
    return 1 - 2 * bits

def awgn_noise(snr_db, size, rate=1.0):
    snr_lin = 10 ** (snr_db / 10)
    sigma = np.sqrt(1.0 / (2.0 * rate * snr_lin))
    return np.random.normal(0, sigma, size)

def calc_LLR(y, snr_db, rate=1.0):
    snr_lin = 10 ** (snr_db / 10)
    sigma2 = 1.0 / (2.0 * rate * snr_lin)
    return 2.0 * y / sigma2