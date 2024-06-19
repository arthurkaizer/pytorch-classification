# CROP AND APPLY CLAHE
import os
import shutil
import cv2
import numpy as np
import tqdm

# Defina os diretórios de entrada e saída
input_dir = "/content/drive/MyDrive/Faculdade/TCC/src/APTOS"
output_dir = "/content/drive/MyDrive/Faculdade/TCC/src/APTOS_processed_256"
SIZE = 256  # Tamanho das imagens de saída

# Funções de pré-processamento de imagens
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # A imagem é muito escura para cortar algo
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def circle_crop(img):
    height, width, _ = img.shape
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)

    return img

# Função para criar a estrutura de diretório de saída
def create_output_structure():
    for split in ['train', 'test', 'val']:
        for class_folder in range(5):
            output_split_dir = os.path.join(output_dir, split, str(class_folder))
            os.makedirs(output_split_dir, exist_ok=True)

def apply_clahe(img):
    # Converta a imagem para escala de cinza
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Crie um objeto CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Aplique o CLAHE à imagem em escala de cinza
    clahe_img = clahe.apply(gray_img)
    # Empilhe novamente os canais para obter uma imagem RGB
    clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
    return clahe_img

# Função para processar as imagens
def process_image(src, dst):
    img = cv2.imread(src)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = circle_crop(img)
    img = cv2.resize(img, (SIZE, SIZE))
    img = apply_clahe(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(dst, img)

# Movimentar os arquivos de entrada para os diretórios de saída e processar as imagens
def move_and_process_files():
    create_output_structure()
    for split in ['train', 'test', 'val']:
        split_dir = os.path.join(input_dir, split)
        for class_folder in os.listdir(split_dir):
            class_folder_path = os.path.join(split_dir, class_folder)
            output_class_folder = os.path.join(output_dir, split, class_folder)
            os.makedirs(output_class_folder, exist_ok=True)
            print(f"Processando {len(os.listdir(class_folder_path))} imagens em {class_folder_path}")
            for i,file_name in enumerate(os.listdir(class_folder_path)):
                src = os.path.join(class_folder_path, file_name)
                dst = os.path.join(output_class_folder, file_name)

                # Processar a imagem antes de mover
                process_image(src, dst)

                print(f"Progresso: {i+1}/{len(os.listdir(class_folder_path))} imagens processadas.", end='\r')


move_and_process_files()
