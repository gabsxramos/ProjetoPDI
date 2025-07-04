import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.util
import skimage.color  # <-- Importação necessária para a conversão de cor


def plot_class_montages(base_path, class_names, images_per_class=4, grid_shape=(1, 4)):
    """
    Carrega imagens de diretórios de classes e exibe uma montagem para cada classe.
    Esta versão suporta imagens coloridas (RGB), com transparência (RGBA) e
    em tons de cinza (grayscale).

    Args:
        base_path (str): O caminho para a pasta principal que contém as pastas das classes.
        class_names (list): Uma lista com os nomes das pastas das classes a serem plotadas.
        images_per_class (int): O número de imagens a serem incluídas na montagem.
        grid_shape (tuple): A forma da grade para a montagem (linhas, colunas).
    """
    num_classes = len(class_names)
    if num_classes == 0:
        print("A lista de classes está vazia. Nenhuma imagem para mostrar.")
        return

    fig, axes = plt.subplots(num_classes, 1, figsize=(40.96, 20.48))
    print(f"\nGerando montagens para {num_classes} classes de '{base_path}'...")

    if num_classes == 1:
        axes = [axes]

    for i, class_name in enumerate(class_names):
        class_path = os.path.join(base_path, class_name)
        ax = axes[i]

        if not os.path.isdir(class_path):
            print(f"⚠️ Aviso: Diretório não encontrado para a classe '{class_name}'. Pulando.")
            ax.set_title(f"Classe '{class_name}' não encontrada", color='red')
            ax.set_axis_off()
            continue

        image_files = sorted([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[
                      :images_per_class]

        class_images = []
        for file_name in image_files:
            image_path = os.path.join(class_path, file_name)
            try:
                class_images.append(skimage.io.imread(image_path))
            except Exception as e:
                print(f"Erro ao ler a imagem {image_path}: {e}")

        if not class_images:
            print(f"⚠️ Aviso: Nenhuma imagem encontrada ou lida na pasta da classe '{class_name}'.")
            ax.set_title(f"Nenhuma imagem para '{class_name}'", color='orange')
            ax.set_axis_off()
            continue

        standardized_images = []
        for img in class_images:
            if img.ndim == 2:
                standardized_images.append(skimage.color.gray2rgb(img))
            elif img.ndim == 3 and img.shape[2] == 4:
                standardized_images.append(skimage.color.rgba2rgb(img))
            elif img.ndim == 3 and img.shape[2] == 1:
                squeezed_img = np.squeeze(img, axis=2)
                standardized_images.append(skimage.color.gray2rgb(squeezed_img))
            else:
                standardized_images.append(img)

        final_images = [skimage.util.img_as_ubyte(img) for img in standardized_images]

        montage = skimage.util.montage(final_images, grid_shape=grid_shape, channel_axis=-1)

        ax.imshow(montage)
        ax.set_title(class_name.replace('_', ' ').title(), fontsize=20)
        ax.set_axis_off()

    plt.tight_layout(pad=2.0)
    plt.show()
