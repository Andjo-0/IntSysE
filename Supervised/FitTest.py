import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the wine quality dataset (assuming winequality-red.csv is in your working directory)
data = pd.read_csv('../data/winequality-white.csv', sep=';')

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

# Initialize Random Forest and KNN classifiers
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=12)
knn_classifier = KNeighborsClassifier(n_neighbors=5, weights='uniform')

# Train the classifiers
rf_classifier.fit(X_train, y_train)
knn_classifier.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_classifier.predict(X_test)
y_pred_knn = knn_classifier.predict(X_test)

# Evaluate the models on both training and test sets
y_pred_rf_train = rf_classifier.predict(X_train)
y_pred_knn_train = knn_classifier.predict(X_train)

# Confusion Matrix for Random Forest
cm_rf_test = confusion_matrix(y_test, y_pred_rf)
cm_rf_train = confusion_matrix(y_train, y_pred_rf_train)

# Confusion Matrix for KNN
cm_knn_test = confusion_matrix(y_test, y_pred_knn)
cm_knn_train = confusion_matrix(y_train, y_pred_knn_train)

# Print accuracy for training and test sets
print(f"Random Forest Training Accuracy: {accuracy_score(y_train, y_pred_rf_train):.2f}")
print(f"Random Forest Test Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
print(f"KNN Training Accuracy: {accuracy_score(y_train, y_pred_knn_train):.2f}")
print(f"KNN Test Accuracy: {accuracy_score(y_test, y_pred_knn):.2f}")

# Plot Confusion Matrices for both models

# Random Forest Confusion Matrices
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(cm_rf_train, annot=True, fmt="d", cmap="Blues", xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.title('Random Forest Training Confusion Matrix')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.subplot(1, 2, 2)
sns.heatmap(cm_rf_test, annot=True, fmt="d", cmap="Blues", xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.title('Random Forest Test Confusion Matrix')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# KNN Confusion Matrices
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(cm_knn_train, annot=True, fmt="d", cmap="Blues", xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.title('KNN Training Confusion Matrix')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.subplot(1, 2, 2)
sns.heatmap(cm_knn_test, annot=True, fmt="d", cmap="Blues", xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.title('KNN Test Confusion Matrix')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
