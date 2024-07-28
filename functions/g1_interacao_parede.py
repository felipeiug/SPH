import numpy as np
from trimesh import Trimesh
from functions.a_inicializar_particulas import Particula
from config.config import Configs

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
    
    dot02 = np.sum(v2 * v0, axis=1)
    dot12 = np.sum(v2 * v1, axis=1)
    
    # Cálculo das coordenadas baricêntricas
    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    
    # Verifica se o ponto está dentro do triângulo
    contains = (u >= 0) & (v >= 0) & (u + v < 1)

    return contains

def point_in_face(point:np.ndarray, face:np.ndarray)->np.ndarray:
    
    # Obter os vértices dos triângulos
    A = face[0]
    B = face[1]
    C = face[2]
    
    # Vetores do triângulo
    v0 = C - A
    v1 = B - A
    v2 = point - A
    
    # Produtos escalares
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot11 = np.dot(v1, v1)
    
    dot02 = np.dot(v2, v0)
    dot12 = np.dot(v2, v1)
    
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
        p0:np.ndarray,
        particle: Particula,
        points_plan:np.ndarray,
        e_plan:np.ndarray,
        mi_plan:np.ndarray,
        dt:float,
    )->np.ndarray:
    p_inicial = p0
    p1 = particle.posicao
    velocity = particle.velocidade
    e_particle = particle.restituition_coeficient
    mi_particle = particle.friction_coeficient

    if p0.shape != particle.posicao.shape:
        raise ValueError("P0 e P1 dever ter exatamente a mesma quantidade de pontos")
    if points_plan.shape[1] != points_plan.shape[2]:
        raise ValueError("O array deve conter representações dos planos, então as dimensões dos pontos e dos arrays devem ser iguais")

    a, b, c, d = plan_by_points(points_plan)

    distance_p0_p1 = np.linalg.norm(p1-p0)
    
    while True:
        # Verificar se ainda cruza a parede
        D0 = (a*p_inicial[0] + b*p_inicial[1] + c*p_inicial[2] + d)
        D1 = (a*p1[0] + b*p1[1] + c*p1[2] + d)

        m1 = np.where((D0>=0), 1, -1)
        m2 = np.where((D1>=0), 1, -1)
        mask = (m1*m2) < 0

        if mask.any():
            p0 = p_inicial

        else:
            # Calcular os novos denominadores
            D0 = (a*p0[0] + b*p0[1] + c*p0[2] + d)
            m1 = np.where((D0>=0), 1, -1)
            mask = (m1*m2) < 0

        if distance_p0_p1 < 0 and not mask.any():
            return p1, velocity
        
        # Ponto de interseção
        t = -D0 / (D1 - D0)
        p_interseccao = np.array([
            p0[0] + t * (p1[0] - p0[0]),
            p0[1] + t * (p1[1] - p0[1]),
            p0[2] + t * (p1[2] - p0[2])
        ]).T

        # Verificar se o ponto de interseção está dentro do triângulo
        mask_point_faces = points_in_faces(p_interseccao, points_plan)

        mask = mask & mask_point_faces

        if not mask.any():
            return p1, velocity

        # Planos que são cruzados por algum ponto
        ordem = np.argsort(np.abs(D0))
        indices = np.arange(D0.size)[ordem]

        for indice in indices:
            if not mask[indice]:
                continue

            p_intersec = p_interseccao[indice]
            
            # Normais dos planos
            plane_normal = np.array([a[indice], b[indice], c[indice]])
            plane_normal = plane_normal / np.linalg.norm(plane_normal)

            # Velocidades finais
            vn = np.dot(velocity, plane_normal)
            nn = np.dot(plane_normal, plane_normal)
            vnn=(vn/nn)*plane_normal

            normal_velocity = velocity - 2*vnn
            tangent_velocity = velocity - vnn

            effective_e = np.sqrt(e_particle * e_plan[indice])
            effective_friction = np.sqrt(mi_particle * mi_plan[indice])
            
            friction_velocity = (effective_friction*tangent_velocity)
            e_velocity = (effective_e*normal_velocity)
            velocity_final = e_velocity - friction_velocity

            mask_velocity_0 = (np.abs(velocity_final) < Configs.velocity_like_0)
            if mask_velocity_0.any():
                multi = np.where(mask_velocity_0, 0, 1)
                velocity_final = multi * velocity_final
            
            # Caso a velocidade tenha zerado
            norm_velocity_final = np.linalg.norm(velocity_final)
            if norm_velocity_final == 0:
                p0 = p_intersec
                p1 = p_intersec
                velocity = velocity_final
                continue
            
            # Distância percorrida até a colizão
            distance = np.linalg.norm(p_intersec - p0)

            # Distancia restante
            distance_p0_p1 = (distance_p0_p1 - distance)*(effective_e)

            dir_velocidade = velocity_final / norm_velocity_final

            p0 = p_intersec
            p1 = p_intersec + dir_velocidade*distance_p0_p1

            velocity = velocity_final
    
    
def integrar_movimento_com_parede(posicao_t_1:np.ndarray, particulas:list[Particula], mesh:Trimesh, e:float, mi:float,  dt:float):
    faces = np.array(mesh.vertices[mesh.faces])

    for n, particula in enumerate(particulas):

        final_position, final_velocity = new_positions_and_velocities(
            p0 = posicao_t_1[n],
            particle = particula,
            points_plan = faces,
            e_plan=np.full(faces.shape[0], e),
            mi_plan=np.full(faces.shape[0], mi),
            dt=dt,
        )
        
        # Atualizar posição e velocidade da partícula
        particula.posicao = final_position
        particula.velocidade = final_velocity


if __name__ == "__main__":
    pass
