# Updated data for the new error matrix and accuracy totals

# Data for the confusion matrix
updated_confusion_matrix = pd.DataFrame([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 95, 1, 92, 178, 3, 0, 0, 5, 0],
    [0, 0, 55, 0, 21, 1, 0, 0, 0, 0],
    [0, 0, 0, 618, 9, 21, 0, 0, 0, 0],
    [0, 9, 0, 3, 441, 2, 21, 0, 0, 0],
    [0, 2, 0, 1, 48, 592, 0, 0, 0, 0],
    [0, 0, 0, 1, 80, 9, 425, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 59, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 122, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 134]
], columns=[
    "Unclassified", "Consump-potatoe", "Peas", "Seed-potatoes",
    "Sugarbeet", "Winterwheat", "Seed onions", "Water",
    "Pine forest", "Deciduous fores"
], index=[
    "Unclassified", "Consump-potatoe", "Peas", "Seed-potatoes",
    "Sugarbeet", "Winterwheat", "Seed onions", "Water",
    "Pine forest", "Deciduous fores"
])

# Plot the normalized confusion matrix
updated_confusion_matrix_normalized = updated_confusion_matrix.div(updated_confusion_matrix.sum(axis=1), axis=0).fillna(0)

plt.figure(figsize=(12, 10))
sns.heatmap(updated_confusion_matrix_normalized, annot=True, fmt=".2f", cmap="Blues", cbar=True, xticklabels=True, yticklabels=True)
plt.title("Normalized Confusion Matrix (Updated Data)")
plt.xlabel("Reference Data (Ground Truth)")
plt.ylabel("Classified Data")
plt.tight_layout()
plt.show()

# Plot the raw confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(updated_confusion_matrix, annot=True, fmt="d", cmap="Reds", cbar=True, xticklabels=True, yticklabels=True)
plt.title("Raw Confusion Matrix (Updated Data)")
plt.xlabel("Reference Data (Ground Truth)")
plt.ylabel("Classified Data")
plt.tight_layout()
plt.show()

# Kappa statistics by class
kappa_classes = [
    "Unclassified", "Consump-potatoes", "Peas", "Seed-potatoes",
    "Sugarbeet", "Winterwheat", "Seed onions", "Water",
    "Pine forest", "Deciduous forest"
]
kappa_values = [0.0, 0.2210, 0.7090, 0.8953, 0.9016, 0.9003, 0.7956, 1.0, 1.0, 1.0]

# Plot Kappa statistics
plt.figure(figsize=(12, 6))
plt.bar(kappa_classes, kappa_values, color="skyblue")
plt.title("Conditional Kappa Statistics by Class")
plt.xlabel("Classes")
plt.ylabel("Kappa Value")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1.1)

# Annotate each bar with its value
for i, v in enumerate(kappa_values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom")

plt.tight_layout()
plt.show()

# Overall classification accuracy
overall_accuracy = 82.71
plt.figure(figsize=(6, 6))
plt.pie(
    [overall_accuracy, 100 - overall_accuracy],
    labels=["Correctly Classified", "Misclassified"],
    autopct="%.1f%%",
    colors=["green", "red"],
    startangle=90
)
plt.title("Overall Classification Accuracy")
plt.tight_layout()
plt.show()
