from functions.a_inicializar_particulas import Particula

def integrar_movimento(particulas:list[Particula], dt:float):
    """
    Integra as equações de movimento para atualizar as posições e velocidades das partículas.
    
    particulas: Lista de todas as partículas.
    dt: Passo de tempo.
    """
    for particula in particulas:
        
        aceleracao = (particula.forca / particula.massa)
        velocidade = 0.5 * aceleracao * dt

        # Antes da forca das paredes
        # Atualizar posição usando o método de Verlet
        nova_posicao = particula.posicao + particula.velocidade * dt + velocidade * dt
        
        # Calcular nova velocidade
        nova_velocidade = particula.velocidade + velocidade
        
        # Atualizar posição e velocidade da partícula
        particula.posicao = nova_posicao
        particula.velocidade = nova_velocidade
