import numpy as np
import matplotlib.pyplot as plt

# Define the confusion matrices
confusion_matrices = [
    np.array([[197, 1, 1, 0, 0], [3, 13, 13, 0, 1], [0, 4, 80, 3, 0], [0, 0, 12, 3, 2], [0, 2, 17, 3, 11]]),
    np.array([[198, 0, 1, 0, 0], [10, 2, 18, 0, 0], [5, 0, 82, 0, 0], [1, 0, 16, 0, 0], [5, 0, 28, 0, 0]])
]

classes = ["No DR","Mild NPDR","Moderate NPDR","Severe NPDR","PDR"]

def calculate_sensitivity_specificity(conf_matrix):
    sensitivities = []
    specificities = []
    num_classes = conf_matrix.shape[0]

    for i in range(num_classes):
        TP = conf_matrix[i, i]
        FN = np.sum(conf_matrix[i, :]) - TP
        FP = np.sum(conf_matrix[:, i]) - TP
        TN = np.sum(conf_matrix) - (TP + FN + FP)

        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

        sensitivities.append(sensitivity)
        specificities.append(specificity)

    return sensitivities, specificities

# Calculate for both matrices
results = [calculate_sensitivity_specificity(cm) for cm in confusion_matrices]

# Convert results to arrays for easy indexing
sens1, spec1 = results[0]
sens2, spec2 = results[1]

# Plotting bar chart for Sensitivity
x = np.arange(len(sens1))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 7))
bar1 = ax.bar(x - width/2, sens1, width, label='Sensibilidade ADAMW')
bar2 = ax.bar(x + width/2, sens2, width, label='Sensibilidade SGD')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Classe')
ax.set_ylabel('Valor')
ax.set_title('Sensibilidade por Classe')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()

# Add a legend to differentiate the bars
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

# Show the plot
plt.tight_layout()
plt.show()

# Plotting bar chart for Specificity
fig, ax = plt.subplots(figsize=(12, 7))
bar1 = ax.bar(x - width/2, spec1, width, label='Especificidade ADAMW')
bar2 = ax.bar(x + width/2, spec2, width, label='Especificidade SGD')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Classe')
ax.set_ylabel('Valor')
ax.set_title('Especificidade por Classe')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()

# Add a legend to differentiate the bars
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

# Show the plot
plt.tight_layout()
plt.show()
