import numpy as np
from functions.a_inicializar_particulas import Particula

def kernel_poly6(rs, h):
    """
    Kernel Poly6 para a função de suavização.
    
    r: Distância entre partículas.
    h: Raio de suavização.
    """
    diferente_0 = (0 <= rs) & (rs <= h)

    vetor_count = (315 / (64 * np.pi * h**9)) * (h**2 - rs**2)**3

    vetor_count[~diferente_0] = 0

    return vetor_count

def calcular_densidade(
    particula:Particula, 
    posicoes:np.ndarray,
    massas:np.ndarray,
    h:float
):
    """
    Calcula a densidade de uma partícula usando o kernel de suavização.
    
    particula: Partícula para a qual a densidade é calculada.
    particulas: Lista de todas as partículas.
    h: Raio de suavização.
    """

    dists = particula.posicao - posicoes
    rs = np.linalg.norm(dists, axis=1)
    
    kernels = kernel_poly6(rs, h)
    densidade = np.sum(massas * kernels)
    
    return densidade
