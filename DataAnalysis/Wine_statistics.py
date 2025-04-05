import pandas as pd


def generate_statistics(file_path, output_file):
    try:
        # Load the dataset
        df = pd.read_csv(file_path, delimiter=';')  # Assuming ';' as the delimiter

        # Separate features and the target variable
        features = df.drop(columns=['quality'])

        # Generate descriptive statistics for each feature
        statistics = features.describe().T  # Transpose for better readability

        # Mode calculation
        statistics['mode'] = features.mode().iloc[0]

        # Calculate standard deviation (variance is the square of the standard deviation)
        statistics['variance'] = features.var()

        # Calculate Q1 and Q3 (25th and 75th percentiles)
        statistics['Q1'] = features.quantile(0.25)
        statistics['Q3'] = features.quantile(0.75)

        # Count the number of duplicate rows
        duplicate_count = df.duplicated().sum()

        # Add duplicate count as a new row to the statistics dataframe
        duplicate_row = pd.DataFrame({'count': [duplicate_count]}, index=['duplicate_count'])
        statistics = pd.concat([statistics, duplicate_row.T])

        # Save the statistics as a CSV file
        statistics.to_csv(output_file)

        print(f"Statistics saved successfully to {output_file}")

    except Exception as e:
        print("Error reading CSV file:", e)


# Example usage
if __name__ == "__main__":
    # Specify the path to your wine quality dataset
    input_file = '../data/winequality-red.csv'  # Change the path if needed
    output_file = 'red_wine_quality_statistics.csv'  # Specify the output file name
    generate_statistics(input_file, output_file)
