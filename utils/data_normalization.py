# /utils/data_normalization.py

import numpy as np
from pathlib import Path
import skimage as ski
from matplotlib import pyplot as plt
from typing import Dict, List, Any


def load_image_data(data_path: str) -> Dict[str, List[np.ndarray]]:
    """
    Carrega imagens de um diretório estruturado por classes.

    A estrutura esperada é:
    - data_path/
      - classe1/
        - img1.png
        - img2.png
      - classe2/
        - ...

    Args:
        data_path (str): O caminho para o diretório principal das imagens (ex: './processed_images').

    Returns:
        Dict[str, List[np.ndarray]]: Um dicionário onde as chaves são os nomes das classes (pastas)
                                     e os valores são listas de imagens (arrays NumPy).
    """
    path = Path(data_path)
    img_data = {}
    print(f"Carregando imagens de '{path.resolve()}'...")
    for item in path.iterdir():
        if item.is_dir():
            img_data[item.name] = []
            for img_file in item.iterdir():
                img = ski.io.imread(img_file)
                img_data[item.name].append(img)
    print(f"Dados carregados para {len(img_data)} classes.")
    return img_data


def calculate_mean_prototypes(img_data: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
    """
    Calcula o protótipo médio (imagem média) para cada canal de cor (R, G, B) de cada classe.

    Args:
        img_data (Dict[str, List[np.ndarray]]): Dicionário com os dados das imagens.

    Returns:
        Dict[str, List[np.ndarray]]: Um dicionário onde as chaves são os nomes das classes e os valores
                                     são listas contendo as matrizes dos canais médios [R, G, B].
    """
    media_classes_canais = {}
    for classe, imagens in img_data.items():
        if not imagens:
            continue

        img_media_canais = [np.zeros_like(imagens[0][:, :, 0], dtype=np.float64) for _ in range(3)]

        for img in imagens:
            for i in range(3):
                img_media_canais[i] += img[:, :, i]

        for i in range(len(img_media_canais)):
            img_media_canais[i] /= len(imagens)

        media_classes_canais[classe] = img_media_canais

    return media_classes_canais


def plot_mean_prototypes(mean_prototypes: Dict[str, List[np.ndarray]]):
    """Plota os protótipos médios de cada classe."""
    num_classes = len(mean_prototypes)
    fig, ax = plt.subplots(num_classes, 3, figsize=(40.96, 20.48))
    ax = ax.ravel()
    rgb_labels = ['R', 'G', 'B']

    i = 0
    for classe, canais in mean_prototypes.items():
        for j in range(3):
            ax[i].imshow(canais[j], cmap='gray')
            ax[i].set_title(f'{classe} - Canal {rgb_labels[j]}')
            ax[i].set_axis_off()
            i += 1

    plt.tight_layout()
    plt.show()


def calculate_mean_histograms(img_data: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Calcula o histograma médio para cada canal de cor de cada classe.

    Args:
        img_data (Dict[str, List[np.ndarray]]): Dicionário com os dados das imagens.

    Returns:
        Dict[str, np.ndarray]: Um dicionário onde as chaves são os nomes das classes e os valores são
                               arrays (256, 3) contendo os histogramas médios para R, G, B.
    """
    hist_medio_classes = {}
    for classe, imagens in img_data.items():
        if not imagens:
            continue

        hist_medio_canais = np.zeros((256, 3), dtype=np.float64)
        for img in imagens:
            for i in range(3):  # Para cada canal R, G, B
                hist, _ = ski.exposure.histogram(img[:, :, i], nbins=256, source_range='dtype')
                hist_medio_canais[:, i] += hist

        hist_medio_canais /= len(imagens)
        hist_medio_classes[classe] = hist_medio_canais

    return hist_medio_classes


def plot_mean_histograms(mean_histograms: Dict[str, np.ndarray]):
    """Plota os histogramas médios de cada classe."""
    num_classes = len(mean_histograms)
    fig, ax = plt.subplots(int(np.ceil(num_classes / 2)), 2, figsize=(40.96, 20.48))
    ax = ax.ravel()

    cores = ['red', 'green', 'blue']
    labels = ['Red', 'Green', 'Blue']

    for i, (classe, hist) in enumerate(mean_histograms.items()):
        for j in range(3):
            ax[i].plot(hist[:, j], color=cores[j], label=labels[j])
        ax[i].set_title(f'Histograma Médio - {classe}')
        ax[i].set_xlabel('Intensidade de Pixel')
        ax[i].set_ylabel('Frequência Média')
        ax[i].legend()
        ax[i].grid(True, linestyle='--', alpha=0.6)

    if len(mean_histograms) < len(ax):
        for i in range(len(mean_histograms), len(ax)):
            ax[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def calculate_histogram_variance(mean_histograms: Dict[str, np.ndarray], img_shape: tuple) -> Dict[
    str, Dict[str, float]]:
    """
    Calcula a variância do histograma médio normalizado de cada classe.

    Args:
        mean_histograms (Dict[str, np.ndarray]): Dicionário com os histogramas médios.
        img_shape (tuple): A forma (altura, largura) de uma imagem de exemplo para normalização.

    Returns:
        Dict[str, Dict[str, float]]: Dicionário com a variância para cada canal de cada classe.
    """
    num_pixels = img_shape[0] * img_shape[1]
    hist_var_classes = {}

    for classe, hist_medio in mean_histograms.items():
        hist_medio_norm = hist_medio.astype(np.float64) / num_pixels

        hist_var_classes[classe] = {
            'R': hist_medio_norm[:, 0].var(),
            'G': hist_medio_norm[:, 1].var(),
            'B': hist_medio_norm[:, 2].var()
        }
    return hist_var_classes


def equalize_histogram_dataset(img_data: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
    """
    Aplica equalização de histograma a um dataset de imagens.
    A equalização é feita no canal V do espaço de cor HSV.

    Args:
        img_data (Dict[str, List[np.ndarray]]): Dicionário com os dados das imagens originais.

    Returns:
        Dict[str, List[np.ndarray]]: Dicionário com as imagens normalizadas.
    """
    norm_imgs = {}
    for classe, imagens in img_data.items():
        norm_imgs[classe] = []
        for img in imagens:
            hsv_img = ski.color.rgb2hsv(img)
            hsv_img[:, :, 2] = ski.exposure.equalize_hist(hsv_img[:, :, 2])
            norm_img = (ski.color.hsv2rgb(hsv_img) * 255).astype(np.uint8)
            norm_imgs[classe].append(norm_img)
    return norm_imgs


def plot_equalization_comparison(img_data: Dict[str, List[np.ndarray]], norm_imgs: Dict[str, List[np.ndarray]]):
    """
    Plota uma comparação lado a lado de uma imagem original e sua versão equalizada para cada classe.
    """
    num_classes = len(img_data)
    fig, ax = plt.subplots(num_classes, 2, figsize=(40.96, 20.48))

    for i, classe in enumerate(img_data.keys()):
        ax[i, 0].imshow(img_data[classe][0])
        ax[i, 0].set_title(f'{classe} - Original')
        ax[i, 0].set_axis_off()

        ax[i, 1].imshow(norm_imgs[classe][0])
        ax[i, 1].set_title(f'{classe} - Equalizada')
        ax[i, 1].set_axis_off()

    plt.tight_layout()
    plt.show()