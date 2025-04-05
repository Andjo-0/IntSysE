#ChatGPT
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the wine quality dataset (using white wine as an example)
data = pd.read_csv('../data/winequality-white.csv', sep=';')

# Display basic dataset info
print("Dataset shape:", data.shape)
print(data.head())


# Convert wine quality to 3 classes: Low, Medium, High
def quality_to_label(q):
    if q <= 4:
        return 0  # Low quality
    elif q <= 6:
        return 1  # Medium quality
    else:
        return 2  # High quality


data['quality_label'] = data['quality'].apply(quality_to_label)

# Separate features and the new 3-class target variable
X = data.drop(['quality', 'quality_label'], axis=1)
y = data['quality_label']

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Define 3 sets of hyperparameters: n_neighbors, leaf_size, and weights
param_sets = [
    {'n_neighbors': 3, 'leaf_size': 20, 'weights': 'uniform'},
    {'n_neighbors': 5, 'leaf_size': 30, 'weights': 'uniform'},
    {'n_neighbors': 7, 'leaf_size': 40, 'weights': 'uniform'}
]

# Define class names for the classification report and confusion matrix
class_names = ['Low', 'Medium', 'High']

# Run tests with different hyperparameters
for i, params in enumerate(param_sets):
    print(f"\n--- KNN Test {i + 1} with parameters: {params} ---")

    # Initialize the KNN classifier with the current hyperparameters
    knn = KNeighborsClassifier(**params)

    # Train the classifier
    knn.fit(X_train, y_train)

    # Make predictions on the training and test sets
    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)

    # Evaluate the classifier
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print("Classification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, target_names=class_names))

    # Display the training confusion matrix
    cm_train = confusion_matrix(y_train, y_pred_train)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"KNN Training Confusion Matrix for Test {i + 1}")
    plt.show()

    # Display the test confusion matrix
    cm_test = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"KNN Test Confusion Matrix for Test {i + 1}")
    plt.show()

    # Basic over/underfitting check based on accuracy gap
    if train_accuracy - test_accuracy > 0.15:
        print("=> The model might be overfitting.")
    elif train_accuracy < 0.80 and test_accuracy < 0.80:
        print("=> The model might be underfitting.")
    else:
        print("=> The model's performance appears balanced.")
