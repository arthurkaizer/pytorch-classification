import os
import random
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image

def plot_original_vs_processed_images(original_root, processed_root, classes):
    # Ordenar a lista de classes
    classes.sort()
    # Criar uma figura com subplots para cada classe
    num_classes = len(classes)
    fig, axes = plt.subplots(num_classes, 2, figsize=(6, num_classes*3))

    # Iterar sobre cada classe
    for i, class_name in enumerate(classes):
        # Obter o caminho das imagens originais e processadas para a classe atual
        original_class_dir = os.path.join(original_root, class_name)
        processed_class_dir = os.path.join(processed_root, class_name)
        # Obter uma lista de todas as imagens na pasta da classe
        images = [f for f in os.listdir(original_class_dir) if f.endswith('.jpg') or f.endswith('.png')]
        # Selecionar uma imagem aleatória dessa classe
        random_image = random.choice(images)
        # Carregar a imagem original
        original_image_path = os.path.join(original_class_dir, random_image)
        original_image = Image.open(original_image_path)
        # Carregar a imagem processada com o mesmo nome
        processed_image_path = os.path.join(processed_class_dir, random_image)
        processed_image = Image.open(processed_image_path)

        # Plotar as imagens original e processada
        axes[i, 0].imshow(original_image)
        axes[i, 0].set_title('Original - ' + class_name)
        axes[i, 0].axis('off')
        axes[i, 1].imshow(processed_image)
        axes[i, 1].set_title('Processada - ' + class_name)
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

# Diretório raiz das imagens originais
original_root = '/content/drive/MyDrive/Faculdade/TCC/src/APTOS/train'
# Diretório raiz das imagens processadas
processed_root = '/content/drive/MyDrive/Faculdade/TCC/src/APTOS_processed_256/train'
# Lista de classes
classes = os.listdir(original_root)

# Plotar imagens originais vs processadas de cada classe
plot_original_vs_processed_images(original_root, processed_root, classes)
