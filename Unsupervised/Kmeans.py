import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load the dataset
df = pd.read_csv('../data/winequality-white.csv', sep=';')

# Use all features except 'quality' for clustering
features = df.drop('quality', axis=1)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# List of k values to experiment with
k_values = [2, 4, 6, 8, 10]

# Initialize lists to store silhouette scores
silhouette_scores = []

# Perform K-means clustering for different k values
for k in k_values:
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    # Calculate the Silhouette Score
    silhouette_avg = silhouette_score(X_scaled, kmeans_labels)
    silhouette_scores.append(silhouette_avg)

    # Reduce dimensions with PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plot the K-means clustering results
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', s=50)
    plt.title(f"K-means Clustering on Wine Quality Dataset (k={k})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label='Cluster Label')
    plt.show()

    print(f"Silhouette Score for k={k}: {silhouette_avg:.3f}")

# Plot silhouette scores for different k values
plt.figure(figsize=(8, 6))
plt.plot(k_values, silhouette_scores, marker='o')
plt.title("Silhouette Score for Different k Values in K-means")
plt.xlabel("k (Number of Clusters)")
plt.ylabel("Silhouette Score")
plt.show()
