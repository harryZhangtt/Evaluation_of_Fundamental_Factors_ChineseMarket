import os
import pandas as pd

# Define the base directory
base_dir = '/Users/zw/Desktop/FinanceReport/FinanceReport'
day_base_dir= '/Users/zw/Desktop/DataBase'
day_liq_free_base_dir= '/Users/zw/Desktop/DataBase-1'


# Define the subfolders
subfolders = ['FSk','NetProfitTtm', 'NetProfit', 'OperRevTtm', 'OperRev', 'NetCashOperTtm',
       'EBIT', 'Monetray', 'TotalAssets', 'TotalEquity', 'TotalLiab','GrossMarginMy', 'TotalProfitTtm', 'TotalCurAssets', 'TotalNonCurLiab',
                         'GOperR','GP','GOperCPS']
base_subfolders=['CP']
# Maximum number of files to process in each subfolder
  # Change this parameter as needed

# Function to extract the date from the filename
def extract_date(filename):
    date_str = filename.split('_')[1].split('.')[0]
    return date_str

# Dictionary to store lists of files by date
files_by_date = {}

# Populate the dictionary with filenames grouped by date
for subfolder in subfolders:
    subfolder_path = os.path.join(base_dir, subfolder)
    if os.path.exists(subfolder_path):
        # Get a list of files and sort them by date
        sorted_files = sorted([f for f in os.listdir(subfolder_path) if f.endswith('.txt')], key=extract_date)
        for file_name in sorted_files:
            date_str = extract_date(file_name)
            if date_str not in files_by_date:
                files_by_date[date_str] = {}
            files_by_date[date_str][subfolder] = os.path.join(subfolder_path, file_name)

# Sort the dates
sorted_dates = sorted(files_by_date.keys())

# Process and merge files by date
output_dir = os.path.join(base_dir, 'Merged_Financial')
os.makedirs(output_dir, exist_ok=True)

for date_str in sorted_dates:
    file_dict = files_by_date[date_str]
    # Read and merge files of the same date
    df_list = []
    column_names = []
    for subfolder in subfolders:
        if subfolder in file_dict:
            df = pd.read_csv(file_dict[subfolder], header=None, delimiter='\t')
            df_list.append(df)
            column_names.extend([f"{subfolder}_{i+1}" for i in range(df.shape[1])])
        else:
            print(f"Warning: Missing file for {subfolder} on {date_str}")
            continue

    # Concatenate dataframes horizontally
    if df_list:
        combined_df = pd.concat(df_list, axis=1)
        # Rename the column 'FSk' to 'SECU_CODE'
        combined_df.columns=['FSk','NetProfitTtm', 'NetProfit', 'OperRevTtm', 'OperRev', 'NetCashOperTtm',
       'EBIT', 'Monetray', 'TotalAssets', 'TotalEquity', 'TotalLiab','GrossMarginMy', 'TotalProfitTtm', 'TotalCurAssets', 'TotalNonCurLiab',
                         'GOperR','GP','GOperCPS']
        combined_df.rename(columns={'FSk': 'SECU_CODE'}, inplace=True)

        # Assign column names based on subfolder names
        combined_df = combined_df[
            ['SECU_CODE', 'NetProfitTtm', 'NetProfit', 'OperRevTtm', 'OperRev', 'NetCashOperTtm',
       'EBIT', 'Monetray', 'TotalAssets', 'TotalEquity', 'TotalLiab','GrossMarginMy', 'TotalProfitTtm', 'TotalCurAssets', 'TotalNonCurLiab',
                         'GOperR','GP','GOperCPS']]

        output_file = os.path.join(output_dir, f"merged_{date_str}.csv")
        combined_df.to_csv(output_file, index=False)
        print(f"Saved combined file for date {date_str} to {output_file}")

print("All files processed and saved.")
