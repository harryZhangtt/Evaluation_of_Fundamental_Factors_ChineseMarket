import os
import pandas as pd

# Define the base directories and subfolders
base_dir_1 = '/Users/zw/Desktop/DataBase-1'
base_dir_2 = '/Users/zw/Desktop/DataBase'
output_base_dir = '/Users/zw/Desktop/FinanceReport/FinanceReport/MergedCSVs'
subfolder = 'Merged_CSVs'

# Ensure the output directory exists
os.makedirs(output_base_dir, exist_ok=True)


# Function to extract the date from the filename
def extract_date(filename):
    date_str = filename.split('_')[1].split('.')[0]
    return date_str


# Function to get sorted files from a base directory
def get_sorted_files(base_dir, subfolder):
    subfolder_path = os.path.join(base_dir, subfolder)
    if os.path.exists(subfolder_path):
        # Get a list of files and sort them by date
        sorted_files = sorted([f for f in os.listdir(subfolder_path) if f.endswith('.csv')], key=extract_date)
        return sorted_files
    return []


# Get sorted files from both directories
sorted_files_dir_1 = get_sorted_files(base_dir_1, subfolder)
sorted_files_dir_2 = get_sorted_files(base_dir_2, subfolder)

# Find the intersection of dates
dates_dir_1 = {extract_date(f) for f in sorted_files_dir_1}
dates_dir_2 = {extract_date(f) for f in sorted_files_dir_2}
common_dates = sorted(dates_dir_1.intersection(dates_dir_2))


# Function to get file path by date
def get_file_by_date(base_dir, subfolder, date_str):
    subfolder_path = os.path.join(base_dir, subfolder)
    for file_name in os.listdir(subfolder_path):
        if extract_date(file_name) == date_str:
            return os.path.join(subfolder_path, file_name)
    return None


# Process and merge files by common dates
for date_str in common_dates:
    print(date_str)
    file_1 = get_file_by_date(base_dir_1, subfolder, date_str)
    file_2 = get_file_by_date(base_dir_2, subfolder, date_str)

    if file_1 and file_2:
        df1 = pd.read_csv(file_1)
        df2 = pd.read_csv(file_2)
        print(df1.columns)
        print(df2.columns)

        # Merge the two dataframes horizontally
        combined_df = pd.concat([df1, df2], axis=1)
        # Desired column names
        desired_columns = ['MvLiqFree_1', 'MvTotal_1', 'MvLiq_1', 'Amount_1', 'CP_1','CPnr_1', 'HP_1', 'LP_1', 'OP_1', 'sk_1',
                           'sk_2', 'Volume_1']

        if list(combined_df.columns) != desired_columns:
            # If not, rename the columns
            combined_df.columns = desired_columns

        combined_df=combined_df[['sk_1','CP_1','CPnr_1','MvLiqFree_1', 'MvTotal_1', 'MvLiq_1', 'Amount_1',  'HP_1', 'LP_1', 'OP_1' ,
                           'sk_2', 'Volume_1']]


        # Save the combined dataframe as a new CSV file
        output_file = os.path.join(output_base_dir, f"merged_{date_str}.csv")
        combined_df.to_csv(output_file, index=False)
        print(f"Saved combined file for date {date_str} to {output_file}")
    else:
        print(f"Missing files for date {date_str} in one or both directories")

print("All files processed and saved.")
