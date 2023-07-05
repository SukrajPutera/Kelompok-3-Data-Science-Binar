import pandas as pd

# Load the TSV file
df = pd.read_csv('train_preprocess.tsv.txt', sep='\t')

# Explore the dataset
print(df.head())  # Display the first few rows of the DataFrame
print(df.info())  # Display information about the DataFrame, including column names and data types
print(df.describe())  # Generate descriptive statistics of the DataFrame

# Perform initial data cleaning
columns_to_drop = ['irrelevant_column1', 'irrelevant_column2']
existing_columns = df.columns.tolist()  # Get the list of existing columns in the DataFrame
columns_to_drop = [col for col in columns_to_drop if col in existing_columns]  # Check if the columns exist in the DataFrame
df = df.drop(columns=columns_to_drop)  # Drop the specified columns from the DataFrame

# Handle missing values
print(df.isnull().sum())  # Check the number of missing values in each column
df = df.dropna()  # Drop rows with missing values or use fillna() to impute missing values

# Validate the cleansed data
print(df.info())      # Verify the information of the DataFrame after cleansing
print(df.describe())  # Validate the descriptive statistics of the DataFrame after cleansing
