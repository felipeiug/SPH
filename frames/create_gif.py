import imageio
import os

image_list:list[str] = os.listdir("frames/")
gif_name = 'output.gif'

def create_gif(image_list:list[str], gif_name):
    frames = []
    # Carrega cada imagem na lista de imagens
    for image_number in range(len(image_list)):
        for image_name in image_list:
            if not image_name.endswith(".png") or ("_" + str(image_number) + ".") not in image_name:
                continue

            frames.append(imageio.imread(f"frames/{image_name}"))
    
    # Salva como GIF usando imageio
    imageio.mimsave(gif_name, frames, duration=0.005)  # Ajuste a duração conforme necessário

create_gif(image_list, gif_name)

print("GIF criado")
