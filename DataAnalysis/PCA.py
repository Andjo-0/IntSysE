import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


def pca_correlation_circle_without_quality(file_path):
    try:
        # Load the dataset
        df = pd.read_csv(file_path, delimiter=';')  # Specify semicolon as the delimiter

        # Separate features and target (quality)
        features = df.drop(columns=['quality'])
        target = df['quality']

        # Standardize the features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Apply PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features_scaled)

        # Define distinct colors for each feature (e.g., using a color palette)
        colors = plt.cm.get_cmap('tab20', len(features.columns))

        # Correlation circle plot (biplot)
        plt.figure(figsize=(10, 7))

        # Plot vectors (features) with different colors
        for i, feature in enumerate(features.columns):
            # Plot each vector (feature)
            plt.quiver(0, 0, pca.components_[0, i], pca.components_[1, i],
                       angles='xy', scale_units='xy', scale=1,
                       color=colors(i), label=feature)  # Use the distinct color for each feature

            # Add label for each vector
            plt.text(pca.components_[0, i] * 1.1, pca.components_[1, i] * 1.1, feature,
                     color=colors(i), fontsize=12)

        # Plot settings
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.axhline(0, color='gray', linewidth=1)
        plt.axvline(0, color='gray', linewidth=1)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA Correlation Circle (Feature Vectors)')

        # Custom legend for feature vectors
        plt.legend(loc='best')

        plt.grid(True)
        plt.show()

        # Scatter plot for PCA results, colored by wine quality
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7, c=target, cmap='viridis')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA of Wine Quality Dataset (Colored by Quality)')

        # Add a colorbar for quality
        cbar = plt.colorbar(scatter)
        cbar.set_label('Wine Quality')

        # Show grid and the plot
        plt.grid(True)
        plt.show()

    except Exception as e:
        print("Error reading CSV file:", e)


# Example usage
if __name__ == "__main__":
    file_path = '../data/winequality-white.csv'
    pca_correlation_circle_without_quality(file_path)
