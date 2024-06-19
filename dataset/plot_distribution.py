# Importa bibliotecas
import os
import matplotlib.pyplot as plt

# Define paths para pastas
train_path = "/content/drive/MyDrive/Faculdade/TCC/src/APTOS_processed_256/train"
val_path = "/content/drive/MyDrive/Faculdade/TCC/src/APTOS_processed_256/val"
test_path = "/content/drive/MyDrive/Faculdade/TCC/src/APTOS_processed_256/test"

# Define a função para contar imagens em cada classe
def count_images_per_class(folder_path):
  """
  Conta o número de imagens em cada classe em uma pasta.

  Args:
    folder_path: Caminho para a pasta que contém as imagens.

  Returns:
    Dicionário com o número de imagens em cada classe.
  """
  class_counts = {}
  for class_dir in os.listdir(folder_path):
    if os.path.isdir(os.path.join(folder_path, class_dir)):
      class_counts[class_dir] = len(os.listdir(os.path.join(folder_path, class_dir)))
  return class_counts

# Conta imagens em cada classe em train, val e test
train_counts = count_images_per_class(train_path)
val_counts = count_images_per_class(val_path)
test_counts = count_images_per_class(test_path)

# Define a função para plotar a distribuição de classes
def plot_class_distribution(counts, title):
  """
  Plota a distribuição de classes em um conjunto de dados.

  Args:
    counts: Dicionário com o número de imagens em cada classe.
    title: Título do gráfico.
  """
  # Prepara os dados para o gráfico
  labels = list(counts.keys())
  counts = list(counts.values())

  # Cria o gráfico de barras
  plt.figure(figsize=(10, 6))
  plt.bar(labels, counts, color='skyblue')
  plt.xlabel('Classe')
  plt.ylabel('Número de Imagens')
  plt.title(title)
  plt.xticks(rotation=45, ha='right')
  plt.tight_layout()
  plt.show()

# Plota a distribuição de classes em train, val e test
print("train",train_counts)
print("val",val_counts)
print("test",test_counts)
plot_class_distribution(train_counts, "Distribuição de Classes no Treino")
plot_class_distribution(val_counts, "Distribuição de Classes na Validação")
plot_class_distribution(test_counts, "Distribuição de Classes no Teste")