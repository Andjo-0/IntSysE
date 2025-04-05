import pandas as pd

# Load both datasets
red_wine = pd.read_csv('../data/winequality-red.csv', sep=';')
white_wine = pd.read_csv('../data/winequality-white.csv', sep=';')

# Combine both datasets (excluding the 'quality' column for feature range analysis)
combined_wine = pd.concat([red_wine, white_wine], ignore_index=True)

# Calculate min and max values for each feature
feature_ranges = pd.DataFrame({
    'Feature': combined_wine.columns,
    'Min Value': combined_wine.min(),
    'Max Value': combined_wine.max()
})

# Print the results
print(feature_ranges)

# Save to CSV (optional)
feature_ranges.to_csv('wine_feature_ranges.csv', index=False)
