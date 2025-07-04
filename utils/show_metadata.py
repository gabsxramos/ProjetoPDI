# /Trabalho PDI/utils/show_metadata.py

import json
import os

def show_json_metadata(file_path):
    """
    Lê um arquivo JSON e exibe seu conteúdo de forma elegante e formatada,
    com tratamento especial para listas de dicionários e listas simples.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            dados = json.load(f)

        file_name = os.path.basename(file_path)
        print("╔" + "═" * 55 + "╗")
        print(f"║ 📊 METADADOS DO PROJETO ({file_name})".ljust(55) + "║")
        print("╚" + "═" * 55 + "╝")

        for chave, valor in dados.items():
            titulo_secao = chave.replace('_', ' ').title()
            print(f"\n🔹 {titulo_secao}:")

            if isinstance(valor, list):
                if chave == 'classes' and all(isinstance(item, dict) for item in valor):
                    print("    ID | Nome da Classe")
                    print("    ---|---------------")
                    for item in valor:
                        class_id = item.get('id', '?')
                        class_name = item.get('nome', 'N/A')
                        print(f"    {class_id:<2} | {class_name}")
                else:
                    for item in valor:
                        print(f"    - {item}")

            elif isinstance(valor, dict):
                for sub_chave, sub_valor in valor.items():
                    print(f"    - {str(sub_chave).capitalize()}: {sub_valor}")

            else:
                print(f"    {valor}")

        print("\n" + "─" * 57)

    except FileNotFoundError:
        print(f"ERRO: O arquivo '{file_path}' não foi encontrado.")
    except json.JSONDecodeError:
        print(f"ERRO: O arquivo '{file_path}' não é um JSON válido.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")