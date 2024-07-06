"""Script to combine all CSV files into one"""
import pandas as pd
import os
import csv

# Get all CSV files in the data directory
input_files = ["./data/" + file for file in os.listdir("./data/") if file.endswith('.csv')]

# Write to a unified "train.csv" file
output_file = "./data/train.csv"

# Delete the existing train.csv file if it exists
if os.path.exists(output_file):
    os.remove(output_file)
    print(f"Deleted existing {output_file}")

try:
    # Read and concatenate all CSV files
    dataframes = [pd.read_csv(file) for file in input_files]
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    # Fill NaN values with 0
    concatenated_df.fillna(0, inplace=True)

    # Randomly shuffle the data
    concatenated_df = concatenated_df.sample(frac=1).reset_index(drop=True)

    # Write the concatenated DataFrame to a new CSV file
    concatenated_df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
    print(f"Successfully concatenated {len(input_files)} files into {output_file}")
except Exception as e:
    print(f"An error occurred: {e}")
