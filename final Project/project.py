import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# --- Task 1: Load and Explore the Dataset ---

print("--- Task 1: Load and Explore the Dataset ---")

try:
    # Load the Iris dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target_names[iris.target]

    print("\nDataset loaded successfully!")

    # Display the first few rows of the dataset
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    # Explore the structure of the dataset
    print("\nDataset Information:")
    df.info()

    # Check for missing values
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # --- Cleaning the dataset (demonstrating with a synthetic missing value) ---
    # The Iris dataset is naturally clean, so we'll introduce a missing value
    # for demonstration purposes and then handle it.
    df_cleaned = df.copy()
    # Introduce a NaN value at a specific location for demonstration
    df_cleaned.loc[5, 'sepal length (cm)'] = np.nan
    df_cleaned.loc[10, 'petal width (cm)'] = np.nan

    print("\nMissing values after introducing synthetic NaNs:")
    print(df_cleaned.isnull().sum())

    # Clean the dataset by dropping missing values
    # For this demonstration, we'll drop rows with any missing values.
    # Alternatively, you could use df_cleaned.fillna(df_cleaned.mean()) for numerical columns.
    initial_rows = df_cleaned.shape[0]
    df_cleaned.dropna(inplace=True)
    rows_after_cleaning = df_cleaned.shape[0]

    print(f"\nCleaned dataset by dropping {initial_rows - rows_after_cleaning} rows with missing values.")
    print("\nMissing values after cleaning:")
    print(df_cleaned.isnull().sum())

except FileNotFoundError:
    print("Error: Dataset file not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred during data loading or initial exploration: {e}")

# --- Task 2: Basic Data Analysis ---

print("\n--- Task 2: Basic Data Analysis ---")

try:
    # Compute the basic statistics of the numerical columns
    print("\nBasic statistics of numerical columns:")
    print(df_cleaned.describe())

    # Perform groupings on 'species' and compute the mean of 'sepal length (cm)' for each group
    print("\nMean 'sepal length (cm)' grouped by 'species':")
    species_sepal_length_mean = df_cleaned.groupby('species')['sepal length (cm)'].mean()
    print(species_sepal_length_mean)

    print("\nMean 'petal length (cm)' grouped by 'species':")
    species_petal_length_mean = df_cleaned.groupby('species')['petal length (cm)'].mean()
    print(species_petal_length_mean)

    # Identify any patterns or interesting findings
    print("\n--- Patterns and Interesting Findings ---")
    print("From the basic statistics and grouped analysis, we can observe:")
    print("1. 'setosa' species generally has the smallest sepal and petal lengths/widths.")
    print("2. 'virginica' species generally has the largest sepal and petal lengths/widths.")
    print("3. There's a clear differentiation in petal dimensions between species, which is often used for classification.")
    print("4. The standard deviation indicates more variability in 'sepal width (cm)' for 'setosa' compared to its other features.")

except Exception as e:
    print(f"An error occurred during data analysis: {e}")

# --- Task 3: Data Visualization ---

print("\n--- Task 3: Data Visualization ---")

try:
    # Set a style for the plots
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 12))

    # 1. Line chart (adapted for non-time series data)
    # Showing sepal length progression across samples, ordered by index.
    # This simulates a 'trend' as data is processed sequentially.
    plt.subplot(2, 2, 1)
    df_cleaned['sample_index'] = df_cleaned.index # Create an index column for plotting
    sns.lineplot(x='sample_index', y='sepal length (cm)', data=df_cleaned)
    plt.title('Sepal Length Progression Across Samples')
    plt.xlabel('Sample Index')
    plt.ylabel('Sepal Length (cm)')
    plt.grid(True)

    # 2. Bar chart: Average petal length per species
    plt.subplot(2, 2, 2)
    sns.barplot(x='species', y='petal length (cm)', data=df_cleaned, ci=None)
    plt.title('Average Petal Length per Species')
    plt.xlabel('Species')
    plt.ylabel('Average Petal Length (cm)')
    plt.grid(axis='y')

    # 3. Histogram of a numerical column: Sepal Length Distribution
    plt.subplot(2, 2, 3)
    sns.histplot(df_cleaned['sepal length (cm)'], kde=True, bins=10)
    plt.title('Distribution of Sepal Length (cm)')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Frequency')
    plt.grid(axis='y')

    # 4. Scatter plot: Relationship between sepal length and petal length
    plt.subplot(2, 2, 4)
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df_cleaned, s=100, alpha=0.7)
    plt.title('Sepal Length vs. Petal Length by Species')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend(title='Species')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred during data visualization: {e}")