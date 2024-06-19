import os
import random
import matplotlib.pyplot as plt
from PIL import Image

from google.colab import drive
drive.mount('/content/drive')

# Defina o caminho do dataset no Google Drive
dataset_path = '/content/drive/MyDrive/Faculdade/TCC/src/APTOS/train'

# Dicionário para mapear os nomes das classes para os respectivos caminhos das pastas
# Exemplo: classes = {'Classe 1': 'folder1', 'Classe 2': 'folder2', ...}
classes = {
    'Sem RD': os.path.join(dataset_path, '0'),
    'RD leve não proliferativa': os.path.join(dataset_path, '1'),
    'RD moderada não proliferativa': os.path.join(dataset_path, '2'),
    'RD severa não proliferativa': os.path.join(dataset_path, '3'),
    'RD proliferativa': os.path.join(dataset_path, '4')
}

# Função para mostrar uma imagem
def imshow(img_path):
    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')

# Procurando uma imagem aleatória de cada classe
images_per_class = {}
for class_name, class_path in classes.items():
    images = os.listdir(class_path)
    random_image = random.choice(images)
    images_per_class[class_name] = os.path.join(class_path, random_image)

plt.figure(figsize=(15, 3))  # Ajuste do tamanho da figura e da largura da linha
for i, (class_name, img_path) in enumerate(images_per_class.items()):
    plt.subplot(1, 5, i+1)  # 1 linha, 5 colunas
    plt.title(class_name, fontsize=10)  # Tamanho da fonte do título ajustado
    imshow(img_path)

plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Ajuste do espaço horizontal e vertical
plt.show()
