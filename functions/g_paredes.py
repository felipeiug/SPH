import numpy as np

def calcular_forca_parede(particula, limites, normais, k_parede, damping):
    """
    Calcula a força de repulsão de todas as paredes em uma partícula.

    particula: Partícula para a qual a força de parede será calculada.
    limites: Lista de posições limites das paredes.
    normais: Lista de vetores normais das paredes.
    k_parede: Constante de rigidez das paredes.
    damping: Fator de amortecimento para a velocidade da partícula.
    """
    forca_total = np.zeros_like(particula.posicao)
    
    for limite, normal in zip(limites, normais):
        forca_total += forca_parede(particula.posicao, limite, normal, k_parede, damping)
    
    particula.forca += forca_total
    return particula.forca

# Parâmetros das paredes
k_parede = 1000  # Constante de rigidez das paredes
damping = 0.1  # Fator de amortecimento
limites = [np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]  # Posições das paredes
normais = [np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, -1])]  # Normais das paredes

# Exemplo de uso da função calcular_forca_parede
calcular_forca_parede(particulas[0], limites, normais, k_parede, damping)
print(f"Força da parede na primeira partícula: {particulas[0].forca}")

# Função de integração que inclui forças de parede
def integrar_movimento_com_parede(particulas, dt, limites, normais, k_parede, damping):
    """
    Integra as equações de movimento para atualizar as posições e velocidades das partículas,
    considerando forças de parede.

    particulas: Lista de todas as partículas.
    dt: Passo de tempo.
    limites: Lista de posições limites das paredes.
    normais: Lista de vetores normais das paredes.
    k_parede: Constante de rigidez das paredes.
    damping: Fator de amortecimento.
    """
    for particula in particulas:
        # Atualizar posição usando o método de Verlet
        nova_posicao = particula.posicao + particula.velocidade * dt + 0.5 * (particula.forca / particula.massa) * dt**2
        
        # Calcular nova velocidade
        nova_velocidade = particula.velocidade + 0.5 * (particula.forca / particula.massa) * dt
        
        # Atualizar posição e velocidade da partícula
        particula.posicao = nova_posicao
        particula.velocidade = nova_velocidade
        
        # Aplicar força de parede
        calcular_forca_parede(particula, limites, normais, k_parede, damping)

# Exemplo de uso da função de integração com paredes
integrar_movimento_com_parede(particulas, dt, limites, normais, k_parede, damping)

# Verificar a atualização
print(f"Nova posição da primeira partícula: {particulas[0].posicao}")
print(f"Nova velocidade da primeira partícula: {particulas[0].velocidade}")
