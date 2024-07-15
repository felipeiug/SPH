from functions.a_inicializar_particulas import Particula
import numpy as np

def integrar_gravidade(particulas:list[Particula], aceleracao_gravidade:np.ndarray, dt:float):
    """
    Integra as equações de movimento para atualizar as posições e velocidades das partículas com base na gravidade.
    
    particulas: Lista de todas as partículas.
    dt: Passo de tempo.
    """
    for particula in particulas:
        parcela2 = 0.5 * aceleracao_gravidade * dt

        # Antes da forca das paredes
        # Atualizar posição usando o método de Verlet
        nova_posicao = particula.posicao + particula.velocidade * dt + parcela2 * dt
        
        # Calcular nova velocidade
        nova_velocidade = particula.velocidade + parcela2
        
        # Atualizar posição e velocidade da partícula
        particula.posicao = nova_posicao
        particula.velocidade = nova_velocidade
