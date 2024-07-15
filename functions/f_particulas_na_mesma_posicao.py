import numpy as np

class Particula:
    def __init__(self, posicao, velocidade, massa):
        self.posicao = np.array(posicao)
        self.velocidade = np.array(velocidade)
        self.massa = massa
        self.densidade = 0.0
        self.pressao = 0.0
        self.forca = np.zeros_like(posicao)

def inicializar_particulas_sem_sobreposicao(num_particulas, volume, massa_total, distancia_minima):
    particulas = []
    
    # Dividir o volume uniformemente entre as partículas
    massa_particula = massa_total / num_particulas
    # Criar partículas em uma grade uniforme dentro do volume especificado
    nx, ny, nz = int(np.cbrt(num_particulas)), int(np.cbrt(num_particulas)), int(np.cbrt(num_particulas))
    espacamento = (volume[1] - volume[0]) / (nx - 1)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                posicao = volume[0] + np.array([i, j, k]) * espacamento
                
                # Verificar se a posição está livre
                sobreposto = False
                for p in particulas:
                    if np.linalg.norm(posicao - p.posicao) < distancia_minima:
                        sobreposto = True
                        break
                
                if not sobreposto:
                    velocidade = np.zeros(3)  # Inicialmente em repouso
                    particula = Particula(posicao, velocidade, massa_particula)
                    particulas.append(particula)
    
    return particulas

def calcular_forca_repulsao(particula, particulas, distancia_minima, k_repulsao):
    """
    Adiciona uma força de repulsão para evitar sobreposição de partículas.
    
    particula: Partícula para a qual a força de repulsão será calculada.
    particulas: Lista de todas as partículas.
    distancia_minima: Distância mínima permitida entre partículas.
    k_repulsao: Constante de força de repulsão.
    """
    forca_repulsao = np.zeros_like(particula.posicao)
    
    for p in particulas:
        if particula != p:
            r = particula.posicao - p.posicao
            distancia = np.linalg.norm(r)
            if distancia < distancia_minima:
                # Calcular força de repulsão
                forca_repulsao += k_repulsao * (distancia_minima - distancia) * (r / distancia)
    
    particula.forca += forca_repulsao
    return particula.forca

# Parâmetros
num_particulas = 1000
volume = np.array([[0, 0, 0], [1, 1, 1]])  # Volume cúbico de 1x1x1
massa_total = 1.0  # Massa total das partículas
distancia_minima = 0.05  # Distância mínima permitida entre partículas
k_repulsao = 1000  # Constante de força de repulsão

# Inicializar partículas sem sobreposição
particulas = inicializar_particulas_sem_sobreposicao(num_particulas, volume, massa_total, distancia_minima)

# Exemplo de uso da função calcular_forca_repulsao
calcular_forca_repulsao(particulas[0], particulas, distancia_minima, k_repulsao)
print(f"Força de repulsão na primeira partícula: {particulas[0].forca}")
