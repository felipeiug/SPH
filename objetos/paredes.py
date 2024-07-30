import numpy as np
from trimesh import Trimesh

from functions.a_inicializar_particulas import Particula
from functions.g1_interacao_parede import plan_by_points, points_in_faces, point_in_face, distance_point_plan

class Parede(Trimesh):
    def __init__(self, k_parede:float, damping:float, *args, **kwargs):
        """
        Parede qualquer

        posicao: Posição da partícula.
        limite: Limite da parede.
        normal: Normal da parede (direção da força de repulsão).
        k_parede: Constante de rigidez da parede.
        damping: Fator de amortecimento para a velocidade da partícula.
        """

        super().__init__(*args, **kwargs)

        self.k_parede = k_parede
        self.damping = damping

        self.faces = np.array(self.vertices[self.faces])
        self.a, self.b, self.c, self.d = plan_by_points(self.faces)


    def forca_parede_particula(self, p0:np.ndarray, particula: Particula):
        p1 = particula.posicao

        D0 = (self.a*p0[0] + self.b*p0[1] + self.c*p0[2] + self.d)
        D1 = (self.a*p1[0] + self.b*p1[1] + self.c*p1[2] + self.d)

        m1 = np.where((D0>=0), 1, -1)
        m2 = np.where((D1>=0), 1, -1)
        mask = (m1*m2) < 0

        if not mask.any():
            return np.zeros_like(particula.forca)
        
        t = -D0 / (D1 - D0)
        p_interseccao = np.array([
            p0[0] + t * (p1[0] - p0[0]),
            p0[1] + t * (p1[1] - p0[1]),
            p0[2] + t * (p1[2] - p0[2])
        ]).T

        mask_point_faces = points_in_faces(p_interseccao, self.faces)

        mask = mask & mask_point_faces

        if not mask.any():
            return np.zeros_like(particula.forca)
        
        indices = np.argsort(np.abs(D0))

        for indice in indices:
            if not mask[indice]:
                continue

            p_intersec = p_interseccao[indice]
            
            # Normais dos planos
            plane_normal = np.array([self.a[indice], self.b[indice], self.c[indice]])
            plane_normal = plane_normal / np.linalg.norm(plane_normal)

            distancia = p1 - p_intersec
            dist_proj_normal = distancia.dot(plane_normal)

            if dist_proj_normal < 0:
                forca_repulsiva = -self.k_parede * dist_proj_normal * plane_normal
                forca_amortecimento = -self.damping * particula.velocidade.dot(plane_normal) * plane_normal
                return -(forca_repulsiva + forca_amortecimento)
                
        return np.zeros_like(p1)








