import pandas as pd
import seaborn as sns

# Create a confusion matrix dataframe from the provided error matrix
confusion_matrix = pd.DataFrame([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 95, 1, 92, 178, 3, 0, 0, 5, 0],
    [0, 0, 55, 0, 21, 1, 0, 0, 0, 0],
    [0, 0, 0, 618, 9, 21, 0, 0, 0, 0],
    [0, 9, 0, 3, 441, 2, 21, 0, 0, 0],
    [0, 2, 0, 1, 48, 592, 0, 0, 0, 0],
    [0, 0, 0, 1, 80, 9, 425, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 59, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 134],
    [0, 0, 0, 0, 0, 0, 0, 0, 122, 0]
], columns=[
    "Unclassified", "Consumption pot", "Peas", "Seed potatoes",
    "Sugarbeet", "Winterwheat", "Seed onions", "Water",
    "Deciduous fores", "Pine forest"
], index=[
    "Unclassified", "Consumption pot", "Peas", "Seed potatoes",
    "Sugarbeet", "Winterwheat", "Seed onions", "Water",
    "Deciduous fores", "Pine forest"
])

# Normalize the confusion matrix for better visualization
confusion_matrix_normalized = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0).fillna(0)

# Plot the normalized confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(confusion_matrix_normalized, annot=True, fmt=".2f", cmap="Blues", cbar=True, xticklabels=True, yticklabels=True)
plt.title("Normalized Confusion Matrix")
plt.xlabel("Reference Data (Ground Truth)")
plt.ylabel("Classified Data")
plt.tight_layout()
plt.show()

# Plot the raw confusion matrix for comparison
plt.figure(figsize=(12, 10))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Reds", cbar=True, xticklabels=True, yticklabels=True)
plt.title("Raw Confusion Matrix")
plt.xlabel("Reference Data (Ground Truth)")
plt.ylabel("Classified Data")
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the matrix and accuracy totals
classes = [
    "Unclassified", "Consumption pot", "Peas", "Seed potatoes", 
    "Sugarbeet", "Winterwheat", "Seed onions", "Water", 
    "Deciduous fores", "Pine forest"
]

reference_totals = [0, 130, 56, 715, 777, 628, 446, 59, 127, 134]
classified_totals = [0, 374, 77, 672, 476, 643, 515, 59, 134, 122]
number_correct = [0, 95, 55, 618, 441, 592, 425, 59, 0, 0]
producer_accuracy = [None, 73.08, 98.21, 86.43, 56.76, 94.27, 95.29, 100.0, 0.0, 0.0]
user_accuracy = [None, 25.4, 71.43, 91.96, 92.65, 92.07, 82.52, 100.0, 0.0, 0.0]

# Bar chart for classified and reference totals
x = np.arange(len(classes))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))

# Plotting the data
bar1 = ax.bar(x - width/2, reference_totals, width, label='Reference Totals')
bar2 = ax.bar(x + width/2, classified_totals, width, label='Classified Totals')

# Adding labels and title
ax.set_xlabel('Classes')
ax.set_ylabel('Counts')
ax.set_title('Reference vs Classified Totals by Class')
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45, ha='right')
ax.legend()

# Annotate bars with their values
for bars in [bar1, bar2]:
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 10, int(yval), ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Line chart for accuracies
fig, ax = plt.subplots(figsize=(12, 6))

# Convert accuracies to arrays for filtering valid values
valid_classes = [c for c, p in zip(classes, producer_accuracy) if p is not None]
valid_producer_accuracy = [p for p in producer_accuracy if p is not None]
valid_user_accuracy = [u for u in user_accuracy if u is not None]

x = np.arange(len(valid_classes))

ax.plot(x, valid_producer_accuracy, marker='o', label='Producer Accuracy (%)')
ax.plot(x, valid_user_accuracy, marker='s', label='User Accuracy (%)')

# Adding labels and title
ax.set_xlabel('Classes')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Producer vs User Accuracy by Class')
ax.set_xticks(x)
ax.set_xticklabels(valid_classes, rotation=45, ha='right')
ax.legend()

# Annotate points with their values
for i, (p, u) in enumerate(zip(valid_producer_accuracy, valid_user_accuracy)):
    ax.text(i, p + 1, f"{p:.1f}%", ha='center', va='bottom')
    ax.text(i, u - 5, f"{u:.1f}%", ha='center', va='top')

plt.tight_layout()
plt.show()

