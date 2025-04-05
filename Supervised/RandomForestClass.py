#ChatGPT
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the wine quality dataset (using white wine as an example)
data = pd.read_csv('../data/winequality-white.csv', sep=';')


# Create a 3-class quality label
def quality_to_label(q):
    if q <= 4:
        return 0  # Low
    elif q <= 6:
        return 1  # Medium
    else:
        return 2  # High


data['quality_label'] = data['quality'].apply(quality_to_label)

# Display basic info
print("Dataset shape:", data.shape)
print(data.head())

# Separate features and target variable for classification
X = data.drop(['quality', 'quality_label'], axis=1)
y = data['quality_label']

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Define 3 sets of hyperparameters
param_sets = [
    {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 2},
    {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 5},
    {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 10}
]

# Labels for confusion matrix and classification report
class_names = ['Low', 'Medium', 'High']

# Run tests with different hyperparameters
for i, params in enumerate(param_sets):
    print(f"\n--- Random Forest Test {i + 1} with parameters: {params} ---")
    rf_classifier = RandomForestClassifier(**params, random_state=12)

    # Train the classifier
    rf_classifier.fit(X_train, y_train)

    # Make predictions on training and test sets
    y_pred_train = rf_classifier.predict(X_train)
    y_pred_test = rf_classifier.predict(X_test)

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
    plt.title(f"Random Forest Training Confusion Matrix for Test {i + 1}")
    plt.show()

    # Display the test confusion matrix
    cm_test = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Random Forest Test Confusion Matrix for Test {i + 1}")
    plt.show()

    # Plot feature importances for each test
    importances = rf_classifier.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = X.columns

    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances for Test {i + 1}")
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), feature_names[indices], rotation=45)
    plt.tight_layout()
    plt.show()

    # Basic over/underfitting check based on accuracy gap
    if train_accuracy - test_accuracy > 0.15:
        print("=> The model might be overfitting.")
    elif train_accuracy < 0.80 and test_accuracy < 0.80:
        print("=> The model might be underfitting.")
    else:
        print("=> The model's performance appears balanced.")
