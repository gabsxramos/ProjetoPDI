import cv2
import numpy as np
import os
import glob

pasta_cvat_masks = 'SegmentationClass'

pasta_preto_e_branco = 'masks_ground_truth_binario'

os.makedirs(pasta_preto_e_branco, exist_ok=True)

lista_mascaras = glob.glob(os.path.join(pasta_cvat_masks, '*.png'))

if not lista_mascaras:
    print(f"ERRO: Nenhuma máscara encontrada na pasta '{pasta_cvat_masks}'.")
    print("Verifique se você exportou corretamente do CVAT e se o nome da pasta está certo.")
else:
    print(f"Encontradas {len(lista_mascaras)} máscaras para converter em binário.")

for caminho_mascara in lista_mascaras:
    nome_base = os.path.basename(caminho_mascara)

    mascara_cinza = cv2.imread(caminho_mascara, cv2.IMREAD_GRAYSCALE)

    if mascara_cinza is not None:
        _, mascara_binaria = cv2.threshold(mascara_cinza, 0, 255, cv2.THRESH_BINARY)

        caminho_saida = os.path.join(pasta_preto_e_branco, nome_base)
        cv2.imwrite(caminho_saida, mascara_binaria)
    else:
        print(f"Erro ao ler a máscara: {nome_base}")

print("\nConversão para binário concluída!")
print(f"Suas máscaras finais foram salvas em '{pasta_preto_e_branco}'.")