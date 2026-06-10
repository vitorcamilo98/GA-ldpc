import numpy as np

def codificar(msg, G):
    return (msg @ G) % 2