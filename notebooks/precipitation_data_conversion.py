import netCDF4 as nc
import pandas as pd
import numpy as np
import os

# Function to reshape arrays to match the base length
def reshape_array(data, base_length):
    flat_data = data.flatten()
    current_length = len(flat_data)

    if current_length < base_length:
        # Pad the array with NaN values if it is too short
        padded_data = np.pad(flat_data, (0, base_length - current_length), constant_values=np.nan)
        return padded_data
    elif current_length > base_length:
        # Truncate the array if it is too long
        truncated_data = flat_data[:base_length]
        return truncated_data
    else:
        # If the length matches, return the flattened data as-is
        return flat_data

# Function to read NetCDF4 file and extract data with reshaping
def nc4_to_dataframe(file_path, base_length=None):
    ds = nc.Dataset(file_path)
    data_dict = {}

    # Find the base length if not already set
    if base_length is None:
        # Set the base length to the first variable's flattened size
        for var_name in ds.variables:
            var_data = np.array(ds.variables[var_name][:]).flatten()
            base_length = len(var_data)
            break  # Use the first variable's length as the base

    # Process each variable
    for var_name in ds.variables:
        var_data = np.array(ds.variables[var_name][:])
        reshaped_data = reshape_array(var_data, base_length)
        data_dict[var_name] = reshaped_data

    # Create a DataFrame from the reshaped data
    df = pd.DataFrame(data_dict)
    return df, base_length

# Path to the folder where your nc4 files are located
folder_path = r'C:\Users\ASUS\DataspellProjects\jeepney_rerouting\precipitation_data\\'

# Generate the list of file paths for each day in the range
from datetime import datetime, timedelta

# Start and end dates for the range
start_date = datetime(2023, 10, 31)
end_date = datetime(2024, 4, 30)

# Generate the list of file paths for each day in the range
file_paths = []
current_date = start_date
while current_date <= end_date:
    file_name = current_date.strftime('%Y-%m-%d.nc4')
    file_paths.append(os.path.join(folder_path, file_name))
    current_date += timedelta(days=1)

# Initialize base_length
base_length = None

# Convert each file and concatenate into one DataFrame
dataframes = []
for file in file_paths:
    df, base_length = nc4_to_dataframe(file, base_length)
    dataframes.append(df)

combined_df = pd.concat(dataframes)

# Save the combined DataFrame to a CSV file
output_csv_path = os.path.join(folder_path, 'combined_data.csv')
combined_df.to_csv(output_csv_path, index=False)

print(f"Conversion complete. CSV saved as '{output_csv_path}'.")
