# /utils/augmented_dataset.py

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Callable

# --- Funções de Aumento de Dados (Lógica de Transformação) ---

def aplicar_log(imagem: np.ndarray) -> np.ndarray:
    """
    Aplica uma transformação logarítmica robusta na imagem.

    A função trata casos de imagens completamente pretas para evitar erros
    e normaliza o resultado para o intervalo de 8-bits (0-255).

    Args:
        imagem (np.ndarray): A imagem de entrada (array NumPy).

    Returns:
        np.ndarray: A imagem transformada.
    """
    img_float = imagem.astype(np.float64)
    max_val = np.max(img_float)

    # Evita divisão por zero se a imagem for toda preta
    if max_val == 0:
        return imagem

    c = 255 / np.log(1 + max_val)
    log_imagem = c * (np.log(img_float + 1))

    return np.array(log_imagem, dtype=np.uint8)


def aplicar_exponencial(imagem: np.ndarray, gamma: float = 0.8) -> np.ndarray:
    """
    Aplica uma transformação exponencial (correção de gamma) na imagem.

    Gamma < 1 clareia a imagem, enquanto gamma > 1 a escurece.

    Args:
        imagem (np.ndarray): A imagem de entrada.
        gamma (float): O valor de gamma para a correção. O padrão é 0.8.

    Returns:
        np.ndarray: A imagem transformada.
    """
    gamma_corrigida = np.power(imagem / 255.0, gamma)
    return np.array(gamma_corrigida * 255, dtype=np.uint8)


def aplicar_filtro_media(imagem: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Aplica um filtro de suavização da média na imagem.

    Args:
        imagem (np.ndarray): A imagem de entrada.
        kernel_size (int): O tamanho da janela do kernel (ex: 5 para uma janela 5x5).

    Returns:
        np.ndarray: A imagem suavizada.
    """
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    media_imagem = cv2.filter2D(src=imagem, ddepth=-1, kernel=kernel)
    return media_imagem


# --- Função Principal (Orquestrador) ---

def process_and_augment_dataset(
    source_dir: str,
    output_dir: str,
    class_map: Dict[int, str],
    augmentations: Dict[str, Callable[[np.ndarray], np.ndarray]]
):
    """
    Processa um dataset, aplicando uma série de aumentos e salvando os resultados.

    Args:
        source_dir (str): Caminho para o diretório do dataset original.
        output_dir (str): Caminho para o diretório onde as imagens aumentadas serão salvas.
        class_map (Dict[int, str]): Dicionário mapeando IDs de classe para nomes de pastas.
        augmentations (Dict[str, Callable]): Dicionário onde a chave é o sufixo do arquivo
                                             (ex: '_log') e o valor é a função de aumento
                                             a ser aplicada.
    """
    print("Iniciando o processo de aumento de dados...")
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Diretório de saída: '{output_path.resolve()}'")

    for class_name in class_map.values():
        dir_classe_original = source_path / class_name
        dir_classe_aumentada = output_path / class_name

        dir_classe_aumentada.mkdir(exist_ok=True)

        print(f"\nProcessando classe: '{class_name}'...")

        if not dir_classe_original.exists():
            print(f"  Aviso: Diretório de origem '{dir_classe_original}' não encontrado. Pulando.")
            continue

        arquivos_imagem = list(dir_classe_original.glob('*.png'))
        print(f"  Encontradas {len(arquivos_imagem)} imagens.")

        for caminho_imagem in arquivos_imagem:
            imagem = cv2.imread(str(caminho_imagem))

            if imagem is None:
                print(f"   Aviso: Não foi possível ler a imagem {caminho_imagem.name}")
                continue

            nome_base, extensao = caminho_imagem.stem, caminho_imagem.suffix

            for suffix, func in augmentations.items():
                img_aumentada = func(imagem)
                novo_nome = f'{nome_base}_{suffix}{extensao}'
                caminho_salvar = dir_classe_aumentada / novo_nome
                cv2.imwrite(str(caminho_salvar), img_aumentada)

    print("\n--- Processo de aumento de dados concluído! ---")