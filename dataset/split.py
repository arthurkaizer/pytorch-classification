import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(image_folder_path, csv_path, train_size=0.7, val_size=0.15):
    # Lê o CSV com os nomes das imagens e suas classes
    df = pd.read_csv(csv_path)

    # Divide os dados em conjunto de treino e teste temporário (incluindo validação)
    train_temp, test = train_test_split(df, test_size=(1 - train_size), stratify=df['Retinopathy grade'])

    # Divide os dados temporários de treino em treino efetivo e validação
    train, val = train_test_split(train_temp, test_size=val_size / (1 - train_size), stratify=train_temp['Retinopathy grade'])

    # Função para copiar as imagens para as respectivas pastas
    def copy_images(data, subset_name):
        for _, row in data.iterrows():
            file_path = os.path.join(image_folder_path, row['Image name'] + '.png') # Supõe que todas as imagens são .jpg
            dest_path = os.path.join(image_folder_path, subset_name, str(row['Retinopathy grade']))

            # Cria a pasta de destino se não existir
            os.makedirs(dest_path, exist_ok=True)
            shutil.copy(file_path, dest_path)

    # Copia as imagens para as respectivas pastas
    for subset, data in zip(['train', 'val', 'test'], [train, val, test]):
        copy_images(data, subset)


image_folder_path = '/content/drive/MyDrive/Faculdade/TCC/src/IDRID/data' # Atualize isso para o seu caminho
csv_path = '/content/drive/MyDrive/Faculdade/TCC/src/IDRID/data.csv' # Atualize isso para o seu caminho

# Chama a função para dividir o dataset e organizar as imagens
split_dataset(image_folder_path, csv_path)
