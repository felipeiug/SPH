import numpy as np
from trimesh import Trimesh
from functions.a_inicializar_particulas import Particula

def plan_by_points(points_plan:np.ndarray):
    if points_plan.shape[1] != points_plan.shape[2]:
        raise ValueError("O array deve conter representações dos planos, então as dimensões dos pontos e dos arrays devem ser iguais")
    
    # Calcular os vetores
    p0 = points_plan[:, 0, :]
    p1 = points_plan[:, 1, :]
    p2 = points_plan[:, 2, :]

    v1 = np.subtract(p1, p0)
    v2 = np.subtract(p2, p0)
    
    # Produto vetorial para encontrar o vetor normal
    normal = np.cross(v1, v2)
    a = normal[:, 0]
    b = normal[:, 1]
    c = normal[:, 2]
    
    # Calcular d usando o ponto p1
    d = - (a * p0[:, 0] + b * p0[:, 1] + c * p0[:, 2])
    
    return a, b, c, d

def points_in_faces(points:np.ndarray, faces:np.ndarray)->np.ndarray:
    if faces.shape[1] != faces.shape[2]:
        raise ValueError("O array deve conter representações dos planos, então as dimensões dos pontos e dos arrays devem ser iguais")
    
    # Obter os vértices dos triângulos
    A = faces[:, 0]
    B = faces[:, 1]
    C = faces[:, 2]
    
    # Vetores do triângulo
    v0 = C - A
    v1 = B - A
    v2 = points - A
    
    # Produtos escalares
    dot00 = np.sum(v0 * v0, axis=1)
    dot01 = np.sum(v0 * v1, axis=1)
    dot11 = np.sum(v1 * v1, axis=1)
    
    dot02 = np.sum(v2 * v0, axis=2)
    dot12 = np.sum(v2 * v1, axis=2)
    
    # Cálculo das coordenadas baricêntricas
    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    
    # Verifica se o ponto está dentro do triângulo
    contains = (u >= 0) & (v >= 0) & (u + v < 1)

    return contains

def distance_point_plan(points:np.ndarray, a:np.ndarray, b:np.ndarray, c:np.ndarray, d:np.ndarray):
    """Retorna para cada ponto, a distância relativa a cada plano
        P1: [distância_plan_1, distância_plan_2]
        P2: [distância_plan_1, distância_plan_2]
    """
    coef_a = a[:, np.newaxis]
    coef_b = b[:, np.newaxis]
    coef_c = c[:, np.newaxis]
    coef_d = d[:, np.newaxis]

    num = coef_a*points[:, 0] + coef_b*points[:, 1] + coef_c*points[:, 2] + coef_d
    den = np.sqrt(coef_a**2 + coef_b**2 + coef_c**2)
    distancias = num/den

    return distancias.T


def new_positions_and_velocities(
        p0:np.ndarray, p1:np.ndarray,
        velocity:np.ndarray,
        e_particles:np.ndarray,
        mi_particles:np.ndarray,
        points_plan:np.ndarray,
        e_plan:np.ndarray,
        mi_plan:np.ndarray,
    )->np.ndarray:
    if p0.shape != p1.shape:
        raise ValueError("P0 e P1 dever ter exatamente a mesma quantidade de pontos")
    if points_plan.shape[1] != points_plan.shape[2]:
        raise ValueError("O array deve conter representações dos planos, então as dimensões dos pontos e dos arrays devem ser iguais")

    a, b, c, d = plan_by_points(points_plan)
    
    # Calcular os denominadores
    coef_a = a[:, np.newaxis]
    coef_b = b[:, np.newaxis]
    coef_c = c[:, np.newaxis]
    coef_d = d[:, np.newaxis]

    D0 = (coef_a*p0[:, 0] + coef_b*p0[:, 1] + coef_c*p0[:, 2] + coef_d)
    D1 = (coef_a*p1[:, 0] + coef_b*p1[:, 1] + coef_c*p1[:, 2] + coef_d)

    mask = (D0 * D1) < 0

    point_mask:np.ndarray = mask.any(axis=0) # Pontos que cruzam qualquer plano

    if not point_mask.any():
        return p1, velocity

    # Planos que são cruzados por algum ponto
    plans_mask = mask.any(axis=1)
    plans = points_plan[plans_mask]
    

    # Ponto de interseção
    t = -D0 / (D1 - D0)
    p_interseccao = np.array([p0[:, 0] + t * (p1[:, 0] - p0[:, 0]), p0[:, 1] + t * (p1[:, 1] - p0[:, 1]), p0[:, 2] + t * (p1[:, 2] - p0[:, 2])]).T

    # Verificar se o ponto de interseção está dentro do triângulo
    mask_point_faces = points_in_faces(p_interseccao, plans)
    
    # Normais dos planos
    plane_normal = np.array([a, b, c]).T
    plane_normal = plane_normal / np.linalg.norm(plane_normal, axis=1)[:, np.newaxis]

    # Velocidades finais
    normal_velocity = np.sum(
        np.linalg.norm(velocity, axis=1)[:, np.newaxis]*plane_normal,
        axis=2
    )*plane_normal
    tangent_velocity = velocity - normal_velocity

    effective_e = np.sqrt(e_particles * e_plan)
    effective_friction = np.sqrt(mi_particles * mi_plan)
    
    valocity_final = (normal_velocity * effective_e) - (effective_friction * tangent_velocity)

    total_force = total_force * mask_point_faces[:, :, np.newaxis]
    total_force = np.nan_to_num(total_force, nan=0.0)
    
    return np.sum(total_force[:], axis=1)

def integrar_movimento_com_parede(posicao_t_1:np.ndarray, particulas:list[Particula], mesh:Trimesh, e:float, mi:float,  dt:float):
    faces = np.array(mesh.vertices[mesh.faces])

    for n, particula in enumerate(particulas):

        final_position, final_velocity = new_positions_and_velocities(
            p0 = np.array([posicao_t_1[n]]),
            p1 = np.array([particula.posicao]),
            velocity = np.array([particula.velocidade]),
            e_particles = np.array([particula.restituition_coeficient]),
            mi_particles = np.array([particula.friction_coeficient]),
            points_plan = faces,
            e_plan=np.full_like(e, faces.shape[0]),
            mi_plan=np.full_like(mi, faces.shape[0]),
        )
        
        # Atualizar posição e velocidade da partícula
        particula.posicao = final_position[0]
        particula.velocidade = final_velocity[0]


if __name__ == "__main__":
    pass
