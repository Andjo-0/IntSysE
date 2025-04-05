import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

def plot_linear_regression(file_path, x_col, y_col):
    try:
        df = pd.read_csv(file_path, delimiter=';')  # Specify semicolon as the delimiter

        # Prepare data for linear regression
        X = df[[x_col]]  # Independent variable
        y = df[y_col]    # Dependent variable

        # Fit the model
        model = LinearRegression()
        model.fit(X, y)

        # Predict values
        y_pred = model.predict(X)

        # Plot Scatter Plot and Regression Line
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x=df[x_col], y=df[y_col], color='blue', label='Data Points')
        plt.plot(df[x_col], y_pred, color='red', label='Regression Line', linewidth=2)

        # Plot settings
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'Linear Regression: {y_col} vs {x_col}')
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print("Error reading CSV file:", e)

# Example usage
if __name__ == "__main__":
    file_path = '../data/winequality-red.csv'  # Adjust the path to your CSV file
    x_col = 'fixed acidity'  # Independent variable (e.g., 'quality')
    y_col = 'alcohol'  # Dependent variable (e.g., 'residual sugar')
    plot_linear_regression(file_path, x_col, y_col)
