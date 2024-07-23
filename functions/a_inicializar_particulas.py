import numpy as np
import trimesh

class Particula:
    def __init__(self,
            posicao:list[float],
            velocidade:list[float],
            restituition_coeficient:float,
            friction_coeficient:float,
            massa:float, volume:float,
        ):
        self.posicao = np.array(posicao)
        self.velocidade = np.array(velocidade)
        self.massa = massa

        # Coeficientes de colizão
        self.restituition_coeficient = restituition_coeficient
        self.friction_coeficient = friction_coeficient

        #TODO: Verificar se densidade pode ter um valor inicial ou se deve ser 0
        self.densidade = massa/volume
        
        self.pressao = 0.0
        self.forca = np.zeros_like(posicao)

def inicializar_particulas(
        mesh:trimesh.Trimesh,
        num_particulas:int,
        massa_total:float = None,
        massa_particula:float = None
    )-> np.ndarray[Particula]:

    if massa_particula is None and massa_total is None:
        raise ValueError("Deve-se passar ao menos o valor da massa total ou da massa de cada partícula")
    
    # Dividir o volume uniformemente entre as partículas
    if massa_particula is None:
        massa_particula = massa_total / num_particulas

    #Dimensão da partícula (cubo)
    volume_total = mesh.volume
    volume_particula = volume_total/num_particulas
    dim_particula = np.cbrt(volume_particula)

    # Mínimos e máximos
    # Obtém os limites da malha
    min_bounds, max_bounds = mesh.bounds

    # Extrai os valores mínimos e máximos
    min_x, min_y, min_z = min_bounds - dim_particula
    max_x, max_y, max_z = max_bounds + dim_particula

    # Valores iniciais das posições das partículas, definido no centro geométrico delas (centro do cubo)
    x, y, z = min_x, min_y, min_z

    positions = np.array([[x, y, z]])
    while z < max_z:
        x += dim_particula

        if x > max_x:
            x = min_x
            y += dim_particula

        if y > max_y:
            y = min_y
            z += dim_particula

        positions = np.concatenate((positions, [[x, y, z]]), axis = 0)

    contained = mesh.contains(positions)

    positions = positions[contained]

    particulas:np.ndarray[Particula] = np.array([])
    for position in positions:
        velocidade = np.zeros(3)

        particula = Particula(position, velocidade, massa_particula, volume_particula)
        particulas = np.append(particulas, [particula])


    return particulas


if __name__ == "__main__":
    # Definições de parâmetros
    # Exemplo de pontos que formam um cubo
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1]
    ])

    # Definindo as faces do cubo
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [2, 3, 7],
        [2, 7, 6],
        [1, 2, 6],
        [1, 6, 5],
        [0, 3, 7],
        [0, 7, 4]
    ])

    # Cria a malha
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    num_particulas = 10000
    massa_total = 1.0  # Massa total das partículas [kg]

    # Inicializar partículas
    particulas = inicializar_particulas(mesh, num_particulas, massa_total)

    # Verificar a inicialização
    print(f"Inicializamos {len(particulas)} partículas.")
    print(f"Posição da primeira partícula: {particulas[0].posicao}")
    print(f"Massa da primeira partícula: {particulas[0].massa}")
