import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes, normalize=True, title='ADAMW Optimizer', cmap=plt.cm.Blues):
    """
    Esta função plota e imprime a matriz de confusão.
    A normalização pode ser aplicada definindo `normalize=True`.
    """
    cm = np.array(cm)  # Converter para array numpy

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de Confusão Normalizada")
    else:
        print('Matriz de Confusão, sem Normalização')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = np.max(cm) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')


# Sua matriz de confusão (exemplo)
confusion_matrix = [[197, 1, 1, 0, 0], [3, 13, 13, 0, 1], [0, 4, 80, 3, 0], [0, 0, 12, 3, 2], [0, 2, 17, 3, 11]]
confusion_matrix = [[198, 0, 1, 0, 0],[10, 2, 18, 0, 0],[5, 0, 82, 0, 0],[1, 0, 16, 0, 0],[5, 0, 28, 0, 0]]
# Classes associadas
classes = ["No DR","Mild NPDR","Moderate NPDR","Severe NPDR","PDR"]

# Plot
plot_confusion_matrix(confusion_matrix, classes)

# Mostrar o gráfico
plt.show()
