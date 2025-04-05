import pandas as pd


def count_quality(file_path):
    df = pd.read_csv(file_path, delimiter=';')

    quality_counts = df['quality'].value_counts()
    print(quality_counts)


if __name__ == "__main__":
    file_path = '../data/winequality-white.csv'  # Adjust the path to your CSV file
    count_quality(file_path)




