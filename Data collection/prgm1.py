# In this type we are downloading the datasets from various sources like Kaggle, Huggingface etc.

import pandas as pd
import os

# Creating a local directory for the data
data_dir = 'house_pricing_data'
os.makedirs(data_dir, exist_ok=True)

try:
    # Loading the data from the local CSV file
    df = pd.read_csv(f'{data_dir}/housing_data.csv')
    print("Data loaded successfully! First 5 rows:")
    print(df.head())
except FileNotFoundError:
    print("Error: 'housing_data.csv' not found. Please download it and place it in the 'house_pricing_data' directory.")