import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(config1_file, config2_file):
    # Leitura dos arquivos CSV
    config1_df = pd.read_csv(config1_file)
    config2_df = pd.read_csv(config2_file)

    # Iteração sobre as métricas
    for metric in config1_df.columns[1:]:  # Ignora a primeira coluna (época)
        plt.figure(figsize=(10, 6))
        plt.plot(config1_df['epoch'], config1_df[metric], label='ADAMW')
        plt.plot(config2_df['epoch'], config2_df[metric], label='SGD')

        plt.xlabel('Épocas')
        plt.ylabel(metric)
        plt.title('Comparação da Métrica {} entre os otimizadores ADAMW e SGD'.format(metric))
        plt.legend()
        plt.grid(True)
        plt.show()

# Exemplo de uso:
config1_file = '/content/drive/MyDrive/Faculdade/TCC/src/metrics/APTOS_processed_256_ADAMW_100_epochs.csv'
config2_file = '/content/drive/MyDrive/Faculdade/TCC/src/metrics/APTOS_processed_256_SGD_100_epochs.csv'
plot_metrics(config1_file, config2_file)
