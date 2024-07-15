import numpy as np
from functions.a_inicializar_particulas import Particula

def kernel_spiky_gradient(rs, norm_r, mask, h):
    """
    Gradiente do kernel Spiky para a função de suavização.
    
    r: Vetor de distância entre partículas.
    h: Raio de suavização.
    """

    B = 45 / (np.pi * h**6)
    
    div_arrays = rs / norm_r[:, np.newaxis]
    multi = ((h - norm_r)**2)[:, np.newaxis] * div_arrays

    grads = -B * multi

    grads[mask] = np.zeros_like(rs[0])

    return grads

def kernel_viscosity_laplacian(norm_r, mask, h):
    """
    Laplaciano do kernel de viscosidade.
    
    r: Vetor de distância entre partículas.
    h: Raio de suavização.
    """

    B = 45 / (np.pi * h**5)
    viscosity = B * (h - norm_r)
    viscosity[mask] = 0
    return viscosity

def calcular_forca(
    index: int,
    particula:Particula,
    posicoes:np.ndarray,
    massas:np.ndarray,
    velocidades:np.ndarray,
    densidades:np.ndarray,
    pressoes:np.ndarray,
    h:float,
    mu:float
):
    
    """
    Calcula a força em uma partícula devido à pressão e viscosidade.
    
    particula: Partícula para a qual a força será calculada.
    particulas: Lista de todas as partículas.
    h: Raio de suavização.
    mu: Coeficiente de viscosidade.
    """

    if velocidades.shape[0] == 1:
        return np.zeros_like(particula.forca)

    velocidades = np.delete(velocidades, index, axis = 0)
    densidades = np.delete(densidades, index, axis = 0)
    posicoes = np.delete(posicoes, index, axis = 0)
    pressoes = np.delete(pressoes, index, axis = 0)
    massas = np.delete(massas, index, axis = 0)

    rs = particula.posicao - posicoes
    
    norm_r = np.linalg.norm(rs, axis=1)
    mask = ~((norm_r > 0) & (norm_r <= h))

    # Força pressão
    grads_w = kernel_spiky_gradient(rs, norm_r, mask, h)

    soma = (particula.pressao + pressoes)
    parte_1 = -massas * (soma / (2 * densidades))
    forca_pressao = np.sum(parte_1[:, np.newaxis] * grads_w, axis=0)

    # Força de viscosidade
    laplacians_w = kernel_viscosity_laplacian(norm_r, mask, h)
    
    subtracao = (velocidades - particula.velocidade)
    parte_1 = subtracao/densidades[:, np.newaxis]
    parte_2 = massas[:, np.newaxis] * parte_1
    forca_viscosidade = np.sum(mu * parte_2 * laplacians_w[:, np.newaxis], axis=0)
    
    particula.forca = forca_pressao + forca_viscosidade
    return particula.forca
