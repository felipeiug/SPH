from functions.a_inicializar_particulas import Particula

def calcular_pressao(particula:Particula, k:float, densidade_referencia:float):
    """
    Calcula a pressão de uma partícula usando uma equação de estado.

    particula: A partícula para a qual a pressão será calculada.
    k: Constante de rigidez do fluido.
    densidade_referencia: Densidade de referência (densidade no estado de repouso).
    """
    particula.pressao = k * (particula.densidade - densidade_referencia)
    return particula.pressao
