import imageio
import os

tempo_total = 15 #segundos
image_list:list[str] = os.listdir("frames/")
gif_name = 'output.gif'

image_list = [i for i in image_list if i.endswith(".png")]
def create_gif(image_list:list[str], gif_name):
    frames = []
    # Carrega cada imagem na lista de imagens
    image_number = 0
    while len(image_list) > 0:
        image_name = f"frame_{image_number}.png"
        if image_name in image_list:
            image_list.remove(image_name)
            frames.append(imageio.imread(f"frames/{image_name}"))
            os.remove(f"frames/{image_name}")
        image_number += 1
    
    # Salva como GIF usando imageio
    imageio.mimsave(gif_name, frames, duration=tempo_total/len(frames))  # Ajuste a duração conforme necessário

create_gif(image_list, gif_name)

print("GIF criado")
