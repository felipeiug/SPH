from tkinter import Tk
from tkinter import messagebox
import os

#Local para salvar os dados da tela inicial
app_data_local = os.getenv('LOCALAPPDATA')
os.makedirs(f"{app_data_local}/SPH/", exist_ok=True)
start_position_file = f"{app_data_local}/SPH/screen_position.txt"

def start_screen_position():
    try:
        with open(start_position_file, mode="r", encoding="UTF-8") as arquivo:
            posicao = arquivo.read().split(",")
            return int(posicao[0]), int(posicao[1])
    except FileNotFoundError:
        return None

def on_close_window_save_screen_position(root: Tk):
    with open(start_position_file, mode="w+", encoding="UTF-8") as arquivo:
        arquivo.write(f"{root.winfo_x()},{root.winfo_y()}")

    res = messagebox.askyesno(
        "Cuidado!",
        f"Deseja mesmo sair do APP?\nAo sair do APP tenha certeza que já terminou seu estudo.",
        parent=root
    )

    if res:
        root.destroy()
        exit()