import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_boxplot(file_path, x_col, y_col):
    try:
        df = pd.read_csv(file_path, delimiter=';')  # Specify semicolon as the delimiter

        # Plot Boxplot
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=df[x_col], y=df[y_col])
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'{y_col} vs {x_col}')
        plt.grid(True)
        plt.show()
    except Exception as e:
        print("Error reading CSV file:", e)


# Example usage
if __name__ == "__main__":
    file_path = '../data/winequality-white.csv'
    x_col = 'quality'  # Change this as needed
    y_col = 'residual sugar'  # Change this as needed
    plot_boxplot(file_path, x_col, y_col)
