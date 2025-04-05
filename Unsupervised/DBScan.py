import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('../data/winequality-white.csv', sep=';')

# Use all features except 'quality' for clustering
features = df.drop('quality', axis=1)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# List of DBSCAN parameter combinations (eps, min_samples)
dbscan_params = [
    {'eps': 0.3, 'min_samples': 5},
    {'eps': 0.5, 'min_samples': 10},
    {'eps': 0.7, 'min_samples': 15}
]

# Perform DBSCAN clustering with different hyperparameters
for i, params in enumerate(dbscan_params):
    print(f"\n--- DBSCAN Experiment {i + 1} with eps={params['eps']} and min_samples={params['min_samples']} ---")

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
    dbscan_labels = dbscan.fit_predict(X_scaled)

    # Reduce dimensions with PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plot the DBSCAN clustering results
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis', s=50)
    plt.title(f"DBSCAN Clustering (eps={params['eps']}, min_samples={params['min_samples']})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label='Cluster Label (with -1 as noise)')
    plt.show()

    # Analyze DBSCAN behavior
    unique_labels = set(dbscan_labels)
    print(f"Number of clusters (including noise): {len(unique_labels)}")
    print(f"Cluster labels: {unique_labels}")

    # DBSCAN often labels noise as -1. If noise is present, consider its effect
    if -1 in unique_labels:
        print("=> The algorithm identified noise points (-1).")
    else:
        print("=> No noise points detected.")

