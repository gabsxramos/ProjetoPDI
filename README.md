# 🍎 Classificador de Frutas

## 📝 Descrição do Projeto

Este projeto consiste em um classificador de imagens de frutas. O processo documentado no notebook `fruit_dataset.ipynb` abrange desde a criação da base de dados, com fotos tiradas de 10 tipos de frutas, até o pré-processamento dessas imagens para serem utilizadas em modelos de aprendizado de máquina.

As fotos foram tiradas com fundo branco e preto em diferentes posições para aumentar a variabilidade do dataset.

## 👥 Autores

- Eduardo Bif Pitol  
- Eduardo Bombonatto Lorenzetti  
- Gabriela Strieder Ramos  
- Jose Vitor Montanger Ribeiro da Silva  

## 📊 Informações do Dataset

- **Classes**: 10  
- **Nomes das Classes**:  
  `acerola`, `lemon`, `cherry_tomato`, `khaki`, `banana`, `lime`, `orange_lemon`, `avocado`, `tangerine`, `pear`  
- **Resolução Original**: 3024 x 4032 pixels  
- **Resolução Redimensionada**: 336 x 448 pixels  
- **Câmera Utilizada**: iPhone 12 e iPhone 13 (Modo Retrato)  
- **Condição de Iluminação**: Luz Natural  

## 📁 Estrutura de Diretórios Sugerida

Para que o notebook funcione corretamente, a seguinte estrutura de pastas é recomendada:

trabalho pdi/
├── fruit_dataset.ipynb
├── metadata.json
├── fruits_heic/ <-- Pasta com as imagens originais em formato .HEIC
├── fruits_png/ <-- Pasta para as imagens convertidas para .PNG
├── fruits_classes/ <-- Pasta final com as imagens redimensionadas e separadas por classe
│ ├── acerola/
│ ├── banana/
│ ├── ... (outras classes)
└── utils/
└── show_images.py


## ⚙️ Dependências

As seguintes bibliotecas Python são necessárias para executar o notebook:

- Pillow  
- pillow-heif  
- scikit-image  
- matplotlib  

Você pode instalá-las usando o comando:

```bash
pip install Pillow pillow-heif scikit-image matplotlib
