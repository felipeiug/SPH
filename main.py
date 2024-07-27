import numpy as np
import trimesh
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from threading import Thread
import string

from functions.a_inicializar_particulas import inicializar_particulas, Particula
from functions.b_calcular_densidade import calcular_densidade
from functions.c_calcular_pressao import calcular_pressao
from functions.d_calcular_forca import calcular_forca
from functions.e_integrar_movimento import integrar_movimento
from functions.g1_interacao_parede import integrar_movimento_com_parede
from functions.h_integralizar_gravidade import integrar_gravidade

from layout.screen_position import start_screen_position, on_close_window_save_screen_position

# from frames.create_gif import *

# Definições de parâmetros
num_particulas = 100
massa_total = 1.0  # Massa total das partículas [kg]
raio_suavizacao = 0.1 # Raio de suavização para o kernel_poly6 para cálculo de densidade
k = 1000  # Constante de rigidez do fluido
mu = 1.002E-3 #coeficiente de viscosidade dinâmica (𝜇) [kg/(m·s)] ou 0.1??
dt = 0.00005 # Tempo entre cada passo em segundos
tempo_max = 15 # tempo do fim da simulação em segundos
num_passos = int(tempo_max/dt)  # Número de passos de tempo
e_parede  = 0.5 # Coeficiente de restituição da parede
mi_parede = 0.002 # Coeficiente de fricção da parede
aceleracao_gravidade = np.array([0, 0, -9.81])

# Exemplo de pontos que formam um cubo
vertices = np.array([
    [-10, -10, 0],
    [10, -10, 2],
    [10, 10, 0],
    [-10, 10, 0],
    # [0, 0, 1],
    # [1, 0, 1],
    # [1, 1, 1],
    # [0, 1, 1]
])

# Definindo as faces do cubo
faces = np.array([
    [0, 1, 2],
    [0, 2, 3],
    # [4, 5, 6],
    # [4, 6, 7],
    # [0, 1, 5],
    # [0, 5, 4],
    # [2, 3, 7],
    # [2, 7, 6],
    # [1, 2, 6],
    # [1, 6, 5],
    # [0, 3, 7],
    # [0, 7, 4]
])

# Cria a malha
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

densidade_referencia = massa_total/mesh.volume  # Densidade de referência (por exemplo, densidade da água em kg/m^3)

# Inicializar partículas
particulas:list[Particula] = np.array([])  #inicializar_particulas(mesh, num_particulas, massa_total)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

global stop_simulation
stop_simulation = False

def loop_simulacao(
    particulas:list[Particula],
    raio_suavizacao:float,
    k:float,
    densidade_referencia:float,
    mu:float,
    dt:float,
    num_passos:float,
    e_parede:float,
    mi_parede:float,
    aceleracao_gravidade:list[float],
):
    global stop_simulation
    
    for passo in range(num_passos):
        if stop_simulation:
            break

        # print(f"Tempo em {passo * dt}s")

        # Posções em t(n-1)
        posicoes = np.array([part.posicao for part in particulas])

        # Atualizar densidade de cada partícula
        massas = np.array([part.massa for part in particulas])
        for n, particula in enumerate(particulas):
            particula.densidade = calcular_densidade(particula, posicoes, massas, raio_suavizacao)
        
        # Atualizar pressão de cada partícula
        for n, particula in enumerate(particulas):
            particula.pressao = calcular_pressao(particula, k, densidade_referencia)
        
        # Novos arrays
        velocidades = np.array([part.velocidade for part in particulas])
        densidades = np.array([part.densidade for part in particulas])
        pressoes = np.array([part.pressao for part in particulas])

        # Atualizar forças em cada partícula
        for n, particula in enumerate(particulas):
            particula.forca = calcular_forca(
                n,
                particula,
                posicoes,
                massas,
                velocidades,
                densidades,
                pressoes,
                raio_suavizacao,
                mu
            )

        # Integrar movimento de cada partícula
        integrar_movimento(particulas, dt)
        integrar_gravidade(particulas, aceleracao_gravidade, dt)
        integrar_movimento_com_parede(
            posicoes,
            particulas,
            mesh,
            e_parede,
            mi_parede,
            dt,
        )

        if passo%np.ceil(np.power(10, (-np.log10(dt)-1))) != 0:
            continue

        # Posição atual
        # print(particulas[0].posicao[2])

        ax.clear()
        xs = [i.posicao[0] for i in particulas]
        ys = [i.posicao[1] for i in particulas]
        zs = [i.posicao[2] for i in particulas]

        ax.set_xlim(mesh.vertices[:, 0].min()-1,mesh.vertices[:, 0].max()+1)
        ax.set_ylim(mesh.vertices[:, 1].min()-1, mesh.vertices[:, 1].max()+1)
        ax.set_zlim(-7, 12)

        str_title = f"Tempo em {(passo * dt):.4f}s"
        str_title += f"\nPosição ({particulas[0].posicao[0]:.2f},{particulas[0].posicao[1]:.2f},{particulas[0].posicao[2]:.2f})"
        str_title += f"\nVelocidade  ({particulas[0].velocidade[0]:.2f},{particulas[0].velocidade[1]:.2f},{particulas[0].velocidade[2]:.2f})"
        ax.set_title(str_title)
        ax.grid(False)

        # Plotando a malha
        ax.plot_trisurf(
            mesh.vertices[:, 0],
            mesh.vertices[:,1],
            mesh.vertices[:,2],
            triangles=mesh.faces,
            ec='k',
            color='black', edgecolor='black',
            alpha=0.2,
            zorder=1,
            # facecolors=plt.cm.viridis(colors)
        )

        # Plotando os pontos
        ax.scatter(
            xs,
            ys,
            zs,
            s=50,
            cmap='viridis',
            zorder=10
            # s=tamanhos,
            # c=cores,
            # alpha=0.5
        )

        canvas.draw()

        plt.savefig(f"frames/frame_{passo}.png")


root = Tk()

root.after(0, lambda: root.wm_state('zoomed'))

# Start position
start_position = start_screen_position()
if start_position:
    x, y = start_position
    sig_x, sig_y = "+", "+"
    if x < 0:
        sig_x = "-"
    if y< 0:
        sig_y = "-"

    x = abs(x)
    y = abs(y)

    root.geometry(f"{sig_x}{x}{sig_y}{y}")

# Ao fechar a janela
root.protocol("WM_DELETE_WINDOW", lambda x=None: on_close_window_save_screen_position(root))

root.title("Gráfico com Tkinter e Matplotlib")

# Criação do frame para o gráfico
frame_grafico = Frame(root)
frame_grafico.pack(padx=10, pady=10)

# Criação da figura do Matplotlib
ax.set_xlim(mesh.vertices[:, 0].min()-0.1,mesh.vertices[:, 0].max()+0.1)
ax.set_ylim(mesh.vertices[:, 1].min()-0.1, mesh.vertices[:, 1].max()+0.1)
ax.set_zlim(0, 12)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.grid(False)

# Plotando a malha
ax.plot_trisurf(
    mesh.vertices[:, 0],
    mesh.vertices[:,1],
    mesh.vertices[:,2],
    triangles=mesh.faces,
    ec='k',
    color='black', edgecolor='black',
    alpha=0.2,
    zorder=1,
    # facecolors=plt.cm.viridis(colors)
)

# Criação do canvas para o gráfico com Tkinter
canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
canvas.draw()
canvas.get_tk_widget().pack()

# Executar loop de simulação
global t
t:Thread = None

def startSimulation():
    global stop_simulation
    global t
    stop_simulation = False

    t = Thread(
        target=loop_simulacao,
        args=(
            np.array([Particula(
                posicao=[2, -2, 3],
                velocidade=[0, 0.01, 3.],
                restituition_coeficient=0.9,
                friction_coeficient=0.9,
                massa=1,
                volume=1
            )]),
            raio_suavizacao,
            k,
            densidade_referencia,
            mu,
            dt,
            num_passos,
            e_parede,
            mi_parede,
            aceleracao_gravidade,
        )
    )
    t.start()

def stopSimulation():
    global t
    global stop_simulation

    stop_simulation = True

    t.join()

# Botão para atualizar o gráfico
frame_btns = Frame(root)
frame_btns.pack()
Button(frame_btns, text="Atualizar Gráfico", command=lambda x=None: startSimulation()).pack(pady=10)
Button(frame_btns, text="Stop", command=lambda x=None: stopSimulation()).pack(pady=10)

# loop_simulacao(particulas, raio_suavizacao, k, densidade_referencia, mu, dt, num_passos)

# Loop principal do Tkinter
root.mainloop()

# Verificar o estado final
print(f"Posição final da primeira partícula: {particulas[0].posicao}")
print(f"Velocidade final da primeira partícula: {particulas[0].velocidade}")
