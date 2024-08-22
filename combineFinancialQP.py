import gc

import numpy as np
import pandas as pd
import os
from datetime import datetime

from matplotlib import pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from statsmodels.robust.scale import mad
from tqdm import tqdm

class StockPitcher:
    def __init__(self, qp_folder_path, financial_data_folder, output_path,industry_file_path):
        self.industry_data = pd.read_excel(industry_file_path, engine='openpyxl')
        self.qp_folder_path = qp_folder_path
        self.financial_data_folder = financial_data_folder
        self.output_path = output_path
        self.data = pd.DataFrame()
        self.value_factors= []
        self.profit_factors=[]
        self.growth_operation_factors=[]
        self.reversion_factor=[]
        self.liquidity_and_other_factors=[]
        self.all_factors=[]

    def extract_date_from_filename(self, filename):
        date_str = filename.split('_')[-1].split('.')[0]
        return datetime.strptime(date_str, '%Y%m%d')

    def get_sorted_files(self, folder_path, prefix):
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and f.startswith(prefix)]
        files.sort(key=lambda x: self.extract_date_from_filename(x))
        return files

    def load_financial_data(self):
        financial_files = self.get_sorted_files(self.financial_data_folder, 'merged')
        merged_financial_df = pd.DataFrame()

        for file_name in financial_files:
            file_path = os.path.join(self.financial_data_folder, file_name)
            financial_df = pd.read_csv(file_path)
            date = self.extract_date_from_filename(file_name)
            financial_df['Date'] = date
            merged_financial_df = pd.concat([merged_financial_df, financial_df], ignore_index=True)

        return merged_financial_df

    def load_qp_data(self):
        start_date = pd.to_datetime('2015-03-31')
        end_date = pd.to_datetime('2024-06-30')
        qp_files = self.get_sorted_files(self.qp_folder_path, 'merged')[2912:]

        qp_merged_df = pd.DataFrame()

        for file_name in qp_files:
            date = self.extract_date_from_filename(file_name)
            if start_date <= date <= end_date:
                qp_file_path = os.path.join(self.qp_folder_path, file_name)
                qp_df = pd.read_csv(qp_file_path)
                qp_df['Date'] = date
                qp_merged_df = pd.concat([qp_merged_df, qp_df], ignore_index=True)

        qp_merged_df.rename(columns={'sk_1': 'SECU_CODE'}, inplace=True)
        qp_merged_df.rename(columns={'CP_1': 'ADJ_CLOSE_PRICE'}, inplace=True)
        qp_merged_df = qp_merged_df.sort_values(by=['Date', 'SECU_CODE'])

        return qp_merged_df

    # def merge_data(self, qp_merged_df, merged_financial_df):
    #     final_df = qp_merged_df.copy()
    #     unique_dates = final_df['Date'].unique()
    #
    #     for unique_date in unique_dates:
    #         print(unique_date)
    #         qp_rows = final_df[final_df['Date'] == unique_date]
    #         financial_rows = merged_financial_df[merged_financial_df['Date'] == unique_date]
    #
    #         if not financial_rows.empty:
    #             for index, qp_row in qp_rows.iterrows():
    #                 secu_code = qp_row['SECU_CODE']
    #                 matching_financial_rows = financial_rows[financial_rows['SECU_CODE'] == secu_code]
    #                 if not matching_financial_rows.empty:
    #                     for col in matching_financial_rows.columns:
    #                         if col not in final_df.columns:
    #                             final_df[col] = None
    #                         final_df.loc[index, col] = matching_financial_rows[col].values[0]
    #     ##only filter stocks that are on trade
    #     final_df= final_df[final_df['sk_2']==1]
    #     # Filter and print the row with Date=2003-04-16 and SECU_CODE=600624
    #     return final_df

    def merge_data(self, qp_merged_df, merged_financial_df):
        final_df = qp_merged_df.copy()
        unique_dates = final_df['Date'].unique()

        # Sort industry data by SECU_CODE
        self.industry_data.sort_values(by='SECU_CODE', inplace=True)

        all_combined = []

        for unique_date in unique_dates:
            print(unique_date)
            qp_rows = final_df[final_df['Date'] == unique_date].copy()
            financial_rows = merged_financial_df[merged_financial_df['Date'] == unique_date]

            if not financial_rows.empty:
                for index, qp_row in qp_rows.iterrows():
                    secu_code = qp_row['SECU_CODE']
                    matching_financial_rows = financial_rows[financial_rows['SECU_CODE'] == secu_code]
                    if not matching_financial_rows.empty:
                        for col in matching_financial_rows.columns:
                            if col not in qp_rows.columns:
                                qp_rows[col] = None
                            qp_rows.loc[index, col] = matching_financial_rows[col].values[0]

            # Merge industry information
            qp_rows.sort_values(by='SECU_CODE', inplace=True)
            industry_info = self.industry_data[['SECU_CODE', 'industry']].set_index('SECU_CODE')
            qp_rows = qp_rows.join(industry_info, on='SECU_CODE')

            all_combined.append(qp_rows)

        final_df = pd.concat(all_combined)

        # Only filter stocks that are not ST and with price greater than 1
        final_df = final_df[(final_df['sk_2'] == 1) & (final_df['ADJ_CLOSE_PRICE'] >= 1)]

        print(final_df.columns)

        return final_df


    def forward_fill_columns(self, df):
        columns_to_ffill = [
            'NetProfitTtm', 'NetProfit', 'OperRevTtm', 'OperRev', 'NetCashOperTtm',
       'EBIT', 'Monetray', 'TotalAssets', 'TotalEquity', 'TotalLiab','GrossMarginMy', 'TotalProfitTtm', 'TotalCurAssets', 'TotalNonCurLiab',
                         'GOperR','GP','GOperCPS'
        ]

        # Filter the columns to only those present in df and not completely NaN
        columns_to_ffill = [col for col in columns_to_ffill if col in df.columns and not df[col].isna().all()]

        # Forward fill the columns within each group
        df[columns_to_ffill] = df.groupby('SECU_CODE')[columns_to_ffill].ffill()

        return df

    def rearrange_columns(self, df):
        columns_order = ['Date','SECU_CODE', 'ADJ_CLOSE_PRICE','CPnr_1' ,'industry','MvLiqFree_1', 'MvTotal_1', 'MvLiq_1',
       'Amount_1', 'HP_1', 'LP_1', 'OP_1', 'sk_2', 'Volume_1',
        'NetProfitTtm', 'NetProfit', 'OperRevTtm', 'OperRev',
       'NetCashOperTtm', 'EBIT', 'Monetray', 'TotalAssets', 'TotalEquity',
       'TotalLiab', 'GrossMarginMy', 'TotalProfitTtm', 'TotalCurAssets',
       'TotalNonCurLiab', 'GOperR', 'GP', 'GOperCPS']
        columns_order = [col for col in columns_order if col in df.columns]
        return df[columns_order]

    def save_to_csv(self, df):
        df.to_csv(self.output_path, index=False)
        print("Data saved to", self.output_path)

    def calculate_value_factor(self):
        df=self.data.copy()
        df.sort_values(by=['Date','SECU_CODE'])
        df['BOOK_VALUE']= df['TotalAssets']-df['TotalLiab']
        df['Market_Cap']= df['MvTotal_1']
        df['BP_LF']=np.where(df['Market_Cap']!=0,df['BOOK_VALUE']/df['Market_Cap'],0)
        df['EP_TTM']= np.where(df['Market_Cap']!=0,df['TotalProfitTtm']/df['Market_Cap'],0)
        df['SP_TTM'] = np.where(df['Market_Cap'] != 0, df['OperRevTtm'] / df['Market_Cap'], 0)
        df['CFP_TTM'] = np.where(df['Market_Cap'] != 0, df['NetCashOperTtm'] / df['Market_Cap'], 0)
        df['EV']= df['Market_Cap']+df['TotalLiab']-df['TotalNonCurLiab']
        df['EBIT2EV']=np.where(df['EV'] != 0, df['EBIT'] / df['EV'], 0)
        df['SALES2EV']=np.where(df['EV'] != 0, df['OperRev'] / df['EV'], 0)
        self.data= df
        self.value_factors=['BP_LF','EP_TTM','SP_TTM','CFP_TTM','EBIT2EV','SALES2EV']
        BP_LF= df[['Date','SECU_CODE','BP_LF','EP_TTM','SP_TTM','CFP_TTM','EBIT2EV','SALES2EV']].copy()

        BP_LF.to_csv('/Users/zw/Desktop/BP_LF.csv')

    def calculate_profit_factor(self):
        df= self.data.copy()
        df.sort_values(by=['Date', 'SECU_CODE'])
        df['ROA']= np.where(df['TotalAssets']!=0,df['NetProfitTtm']/df['TotalAssets'],0)
        df['ROE']=np.where(df['TotalEquity']!=0,df['NetProfitTtm']/df['TotalEquity'],0)
        df['GrossMarginMy']=df['GrossMarginMy']
        self.profit_factors=['ROA','ROE','GrossMarginMy']
        self.data=df

    def calculate_Growth_Operation_factor(self):
        df=self.data.copy()
        df.sort_values(by=['Date', 'SECU_CODE'])
        df['AssetTurnover']=np.where(df['TotalAssets'] != 0, df['OperRev'] / df['TotalAssets'], 0)
        df['Debt2Asset']= np.where(df['TotalAssets'] != 0, df['TotalLiab'] / df['TotalAssets'], 0)
        self.growth_operation_factors = ['GOperR','GP','GOperCPS','AssetTurnover','Debt2Asset']
        self.data = df

    def calculate_reversion_factor(self):
        df = self.data.copy()
        df.sort_values(by=['Date', 'SECU_CODE'], inplace=True)
        df['ADJ_CLOSE_1MAgo']= df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].shift(21).fillna(0)
        df['Ret1M']= np.where(df['ADJ_CLOSE_1MAgo']!=0,(df['ADJ_CLOSE_PRICE']-df['ADJ_CLOSE_1MAgo'])/df['ADJ_CLOSE_1MAgo'],0)
        df['ADJ_CLOSE_6MAgo']= df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].shift(168).fillna(0)
        df['Momentumlast6M']= np.where(df['ADJ_CLOSE_6MAgo']!=0,(df['ADJ_CLOSE_PRICE'])/df['ADJ_CLOSE_6MAgo']-1,0)
        df['1MAvg_ADJ_CLOSE']= df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].rolling(window=21,min_periods=10).mean().reset_index(level=0,
                                                                                                             drop=True)
        df['Momentumave1M']=np.where(df['1MAvg_ADJ_CLOSE']!=0,(df['ADJ_CLOSE_PRICE']-df['1MAvg_ADJ_CLOSE'])/df['1MAvg_ADJ_CLOSE']-1,0)
        # Calculate daily return
        df['DailyReturn'] = df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].pct_change()
        df['PPReversal'] = -df['DailyReturn']
        # Calculate the 3-month lowest price
        df['Price_3M_Low'] = df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].rolling(window=63).min().reset_index(level=0,
                                                                                                             drop=True)

        # Calculate the CGO_3M
        df['CGO_3M'] = np.where(df['ADJ_CLOSE_PRICE']!=0,(df['ADJ_CLOSE_PRICE'] - df['Price_3M_Low']) / df['ADJ_CLOSE_PRICE'],0)
        # Drop the intermediate column
        df = df.drop(columns=['Price_3M_Low'])
        self.reversion_factor=['Ret1M','Momentumlast6M','Momentumave1M','PPReversal','CGO_3M']
        for factor in self.reversion_factor:
            print(df[factor])
        self.data=df


    def calculate_liquidity_other_factor(self):
        df = self.data.copy()
        df.sort_values(by=['Date', 'SECU_CODE'], inplace=True)

        # Calculate Turnover Ratio and TO_1M
        df['TurnoverRatio'] = np.where(df['MvLiqFree_1'] != 0, df['Volume_1'] / df['MvLiqFree_1'], 0)
        df['TO_1M'] = df.groupby('SECU_CODE')['TurnoverRatio'].rolling(window=30, min_periods=1).mean().reset_index(
            level=0, drop=True)



        # Calculate ILLIQ
        df['ILLIQ'] = df['DailyReturn'].abs() / df['Volume_1']

        # Calculate the ILLIQ for a fixed trading volume of 100 million
        fixed_volume = 100_000_000
        df['ILLIQ_fixed'] = df['ILLIQ'] * df['Volume_1'] / fixed_volume

        # Calculate AmountAvg_1M and AmountAvg_3M
        df['AmountAvg_1M'] = df.groupby('SECU_CODE')['Volume_1'].rolling(window=30, min_periods=30).mean().reset_index(
            level=0, drop=True)
        df['AmountAvg_3M'] = df.groupby('SECU_CODE')['Volume_1'].rolling(window=90, min_periods=30).mean().reset_index(
            level=0, drop=True)
        df['AmountAvg_1M_3M'] = df['AmountAvg_1M'] / df['AmountAvg_3M']

        # Calculate RealizedVol_3M
        df['RealizedVol_3M'] = df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].rolling(window=90,
                                                                                  min_periods=30).std().reset_index(
            level=0, drop=True)

        # Calculate Max_Ret
        df['Max_Ret'] = df.groupby('SECU_CODE')['DailyReturn'].rolling(window=30, min_periods=1).max().reset_index(
            level=0, drop=True)

        self.liquidity_and_other_factors = ['TO_1M', 'ILLIQ_fixed', 'AmountAvg_1M_3M', 'RealizedVol_3M', 'Max_Ret']
        self.data = df

    def normalize(self, series):
        mean = series.mean()
        std = series.std()
        return (series - mean) / std if std != 0 else series

    def _calculate_residuals(self, group, vector_column, response_column):
        X = pd.to_numeric(group[vector_column], errors='coerce').values.reshape(-1, 1)
        y = pd.to_numeric(group[response_column], errors='coerce').values.reshape(-1, 1)

        valid_indices = ~np.isnan(X).flatten() & ~np.isnan(y).flatten() & np.isfinite(X).flatten() & np.isfinite(
            y).flatten()
        X = X[valid_indices]
        y = y[valid_indices]

        if len(X) == 0 or len(y) == 0:
            return pd.Series([np.nan] * len(group), index=group.index)

        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        residuals = y - y_pred

        alpha = pd.Series(np.nan, index=group.index)
        alpha.iloc[valid_indices] = residuals.flatten()
        return alpha

    def raw_alpha_processing(self, df, value_factor):
        df['Market_Cap'] = pd.to_numeric(df['MvTotal_1'], errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)
        df['log_size'] = np.log(df['Market_Cap'].replace(0, np.nan))  # Replace 0 with NaN to avoid log(0)
        df['mad_log_size'] = df.groupby('Date')['log_size'].transform(lambda x: mad(x, center=np.median)).fillna(
            0).replace([np.inf, -np.inf], 0)

        industry_neutral_col = f'industry_neutral_{value_factor}'
        industry_size_neutral_col = f'industry_size_neutral_{value_factor}'
        normalized_col = f'normalized_{value_factor}'

        # Convert the value factor column to numeric
        df[value_factor] = pd.to_numeric(df[value_factor], errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)

        # Calculate industry neutral value factor
        tqdm.pandas(desc=f"Calculating industry neutral {value_factor}")
        df[industry_neutral_col] = df.groupby('industry')[value_factor].transform(lambda x: x - x.mean()).fillna(
            0).replace([np.inf, -np.inf], 0)

        # Calculate industry size neutral value factor
        tqdm.pandas(desc=f"Calculating industry size neutral {value_factor}")
        industry_size_neutral_df = df.groupby('Date', group_keys=False).progress_apply(
            lambda group: self._calculate_residuals(group, 'mad_log_size', industry_neutral_col)
        ).fillna(0).replace([np.inf, -np.inf], 0)
        df[industry_size_neutral_col] = industry_size_neutral_df

        # Normalize the value factor
        tqdm.pandas(desc="Normalizing the value factor")
        df[normalized_col] = df.groupby('Date')[industry_size_neutral_col].transform(self.normalize).fillna(
            0).infer_objects(copy=False)

        # Concatenate all columns at once to avoid fragmentation
        result_df = df.copy()  # Copy to avoid modifying the original DataFrame
        result_df = pd.concat([result_df, df[[industry_neutral_col, industry_size_neutral_col, normalized_col]]],
                              axis=1)
        result_df = result_df.copy()  # To defragment the DataFrame
        #save the individual factor as csv
        factor_csv= result_df[['Date','SECU_CODE',normalized_col,f'{value_factor}_filter']]
        factor_dir= os.path.join(output_path,'Factor')
        factor_path= os.path.join(factor_dir,f'{value_factor}.csv')
        factor_csv.to_csv(factor_path,index=False)

        gc.collect()
        return result_df

    def evaluate_factor_IC(self):
        df = self.data.copy()
        price_df= df[['Date','SECU_CODE','ADJ_CLOSE_PRICE','industry','MvTotal_1']].copy()
        price_dir = os.path.join(output_path, 'Factor')
        price_path= os.path.join(price_dir,'price.csv')
        price_df.to_csv(price_path,index=False)

        df.sort_values(by=['Date', 'SECU_CODE'], inplace=True)

        # Calculate stock returns
        df['stocks_return'] = np.log(df['ADJ_CLOSE_PRICE'] / df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].shift(1))


        factor_ics = {}


        # self.all_factors = self.value_factors + self.profit_factors + self.growth_operation_factors + self.reversion_factor + self.liquidity_and_other_factors
        self.all_factors = self.growth_operation_factors + self.reversion_factor + self.liquidity_and_other_factors+self.profit_factors+self.value_factors

        for value_factor in self.all_factors:
            # Calculate z-scores
            df[f'{value_factor}_zscore'] = (df[value_factor] - df.groupby('Date')[value_factor].transform('mean')) / \
                                           df.groupby('Date')[value_factor].transform('std')

            # Shift z-scores by one period
            df[f'{value_factor}_zscore'] = df.groupby('SECU_CODE')[f'{value_factor}_zscore'].shift(1)

            # Calculate daily IC
            daily_ic = df.groupby('Date', group_keys=False).apply(
                lambda x: x[f'{value_factor}_zscore'].corr(x['stocks_return']))

            # Store the mean IC for the value factor
            factor_ics[value_factor] = daily_ic.mean()

            '''determine the sorting order based on sign of IC'''

            df[f'{value_factor}_filter'] = daily_ic.mean()

            print(f'{value_factor} IC: {factor_ics[value_factor]}')
        self.data = df

    def factor_combination(self):
        """ currently, the even distribution has the best performance, while ic weighted does not"""
        df = self.data.copy()
        methods=['even','IC']
        df.sort_values(by=['Date', 'SECU_CODE'], inplace=True)
        print(df.columns)
        num = len(self.all_factors)
        self.all_factors= ['GrossMarginMy']


        def even_distribution(df):
            df['even_weight_alpha'] = 0.0  # Initialize with 0 for addition
            for factor in self.all_factors:
                self.raw_alpha_processing(df,factor)
                df[f'{factor}_filter'] = df[f'{factor}_filter'].fillna(0).astype(float)

                df['even_weight_alpha'] += np.where(df[f'{factor}_filter'] > 0, df[f'normalized_{factor}'],
                                                    -df[f'normalized_{factor}'])


            df['even_weight_alpha'] = df['even_weight_alpha'] / num
            # df['even_weight_alpha'] = df.groupby('Date')['even_weight_alpha'].transform(self.normalize).fillna(0).infer_objects(copy=False)
            return df

        def ic_weighted(df):
            sum=0
            df['IC_weighted_alpha']=0.0
            for factor in self.all_factors:
                if factor != 'even_weight_alpha':
                    df[f'{factor}_filter'] = df[f'{factor}_filter'].fillna(0).astype(float)
                    # if  (df[f'{factor}_filter'].abs()>0.01).any():
                    df['IC_weighted_alpha'] += df[f'{factor}_filter'] * df[f'normalized_{factor}']
                    sum+=df[f'{factor}_filter'].mean()

            df['IC_weighted_alpha']= df['IC_weighted_alpha']/abs(sum)
            # df['IC_weighted_alpha'] = df.groupby('Date')['IC_weighted_alpha'].transform(self.normalize).fillna(
            #     0).infer_objects(copy=False)
            return df

        for method in methods:
            if method == 'even':
                df = even_distribution(df)  # Assign the returned DataFrame
                df['even_weight_alpha_filter']=1
                self.all_factors.append('even_weight_alpha')
            elif method=='IC':
                df= ic_weighted(df)
                df['IC_weighted_alpha_filter']=1
                self.all_factors.append('IC_weighted_alpha')
        self.data = df

    def weight_assignment(self, df, positive_filter=True):
        df = df.sort_values(by='Date')

        def add_noise(series):
            return series + np.random.normal(0, 1e-6, len(series))


        """
        make sure in top 10 are all positive and bottom 10 are all negative"""

        def quantile_transform(group):
            group = group.dropna(subset=['vector'])  # Drop NaNs
            try:
                group['quantile'] = pd.qcut(group['vector'], q=10, labels=False, duplicates='drop')
            except ValueError:
                group['quantile'] = pd.qcut(add_noise(group['vector']), q=10, labels=False,
                                            duplicates='drop')
            return group

        df = df.groupby('Date').apply(lambda group: quantile_transform(group)).reset_index(drop=True)

        df['long_weight'] = 0.0
        df['short_weight'] = 0.0

        if positive_filter:
            top_10_mask = df['quantile'] == 9
            bottom_10_mask = df['quantile'] == 0
        else:
            top_10_mask = df['quantile'] == 0
            bottom_10_mask = df['quantile'] == 9

        # Assign weights based on normalized vectors

        df.loc[top_10_mask, 'long_weight'] = abs(
            df['vector'] / df[top_10_mask].groupby('Date')['vector'].transform('sum')).astype(float)
        df.loc[bottom_10_mask, 'short_weight'] = (
                    -1 * abs(df['vector'] / df[bottom_10_mask].groupby('Date')['vector'].transform('sum'))).astype(
                float)



        df['weight'] = 0.0
        df.loc[top_10_mask, 'weight'] = df['long_weight']
        df.loc[bottom_10_mask, 'weight'] = df['short_weight']

        return df

    def calculate_factor_exposure(self,df, factor_name):
        df = df.copy()
        df['factor_exposure'] = df[factor_name].astype(float)
        return df

    def calculate_factor_stability_coefficient(self,df):
        # Ensure the data is sorted by Date and SECU_CODE
        df = df.sort_values(by=['Date', 'SECU_CODE'])

        # Extract year and month from Date
        df['year_month'] = df['Date'].dt.to_period('M')

        # Initialize a list to store the stability coefficients
        stability_coefficients = []

        # Group by year_month
        grouped = df.groupby('year_month')

        # Get the unique periods
        periods = df['year_month'].unique()

        # Loop through consecutive periods to calculate the cross-sectional correlation
        for i in range(1, len(periods)):
            prev_period = periods[i - 1]
            current_period = periods[i]

            # Get the factor exposures for the previous and current periods
            prev_exposures = grouped.get_group(prev_period)[['SECU_CODE', 'factor_exposure']].set_index('SECU_CODE')
            current_exposures = grouped.get_group(current_period)[['SECU_CODE', 'factor_exposure']].set_index(
                'SECU_CODE')

            # Join the exposures on SECU_CODE
            merged_exposures = prev_exposures.join(current_exposures, lsuffix='_prev', rsuffix='_curr', how='inner')

            # Ensure both DataFrames have the same length and there are enough data points
            min_length = min(len(prev_exposures), len(current_exposures))
            if min_length > 1:
                prev_exposures = prev_exposures.iloc[:min_length]
                current_exposures = current_exposures.iloc[:min_length]

                # Calculate the cross-sectional correlation
                correlation = prev_exposures['factor_exposure'].corr(current_exposures['factor_exposure'])

                # Store the stability coefficient
                stability_coefficients.append({
                    'year_month': current_period,
                    'factor_stability_coefficient': correlation
                })

        # Convert to DataFrame
        stability_df = pd.DataFrame(stability_coefficients)
        # Initialize the factor_stability_coefficient column in df
        df['factor_stability_coefficient'] = np.nan

        # Assign stability coefficients to the original dataframe using .loc
        for index, row in stability_df.iterrows():
            df.loc[df['year_month'] == row['year_month'], 'factor_stability_coefficient'] = row[
                'factor_stability_coefficient']

        return df

    def construct_and_backtest_alpha(self):
        df = self.data.copy()
        df.sort_values(by=['Date', 'SECU_CODE'], inplace=True)
        pnl_dir = self.output_path
        initial_capital = 1e8  # Set an initial capital for the example
        self.all_factors=['even_weight_alpha','IC_weighted_alpha']

        for value_factor in self.all_factors:
            # Calculate factor exposures
            df['vector'] = df[f'{value_factor}'] # Shift by one to avoid future values
            df['vector'] = df.groupby('SECU_CODE')[value_factor].shift(1)

            positive_filter = (df[f'{value_factor}_filter'] > 0).any()
            df = self.weight_assignment(df, positive_filter)

            df['long_capital_allocation'] = initial_capital * df['long_weight']
            df['short_capital_allocation'] = initial_capital * df['short_weight']

            df['long_investments'] = ((df['long_capital_allocation'] / df['ADJ_CLOSE_PRICE']) // 100) * 100
            df['short_investments'] = ((df['short_capital_allocation'] / df['ADJ_CLOSE_PRICE']) // 100) * 100

            df['investment'] = 0
            df.loc[df['weight'] >= 0, 'investment'] = df['long_investments'].astype(float)
            df.loc[df['weight'] < 0, 'investment'] = df['short_investments'].astype(float)

            df['next_day_return'] = df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].diff()
            df['next_day_return'] = np.where(abs(df['next_day_return']) < abs(df['ADJ_CLOSE_PRICE']),
                                             df['next_day_return'],
                                             np.where(df['next_day_return'] < 0,
                                                      -0.2 * df['ADJ_CLOSE_PRICE'],
                                                      0.2 * df['ADJ_CLOSE_PRICE']))

            df['next_day_return'] = df['next_day_return'].fillna(0)

            df['previous_investment'] = df.groupby('SECU_CODE')['investment'].shift(1)
            df['investment_change'] = (df['investment'] - df['previous_investment']).fillna(0)
            df['abs_investment_change'] = abs(df['investment_change'])

            df['pnl'] = df['investment'] * df['next_day_return']
            df['pnl'] = df['pnl'].fillna(0)
            df['long_pnl'] = np.where(df['weight'] > 0, df['pnl'], 0)
            df['short_pnl'] = np.where(df['weight'] < 0, df['pnl'], 0)

            df['Date'] = pd.to_datetime(df['Date'])

            df['tvr_shares'] = df['abs_investment_change']
            df['tvr_values'] = df['abs_investment_change'] * df['ADJ_CLOSE_PRICE']
            df['tvr_shares'] = df['tvr_shares'].fillna(0)
            df['tvr_values'] = df['tvr_values'].fillna(0)
            # Calculate factor stability coefficient
            df = self.calculate_factor_exposure(df, value_factor)
            df = self.calculate_factor_stability_coefficient(df)
            df.rename(columns={'factor_stability_coefficient_y':'factor_stability_coefficient'},inplace=True)
            print(df['factor_stability_coefficient'])

            if positive_filter:
                long_quantile = 9
                short_quantile = 0
            else:
                long_quantile = 0
                short_quantile = 9

            aggregated = df.groupby('Date').agg(
                year_month=('year_month','last'),
                pnl=('pnl', 'sum'),
                long_pnl=('long_pnl', 'sum'),
                short_pnl=('short_pnl', 'sum'),
                long_size=('investment', lambda x: (
                        x[(df.loc[x.index, 'quantile'] == long_quantile)] * df.loc[x.index, 'ADJ_CLOSE_PRICE']).sum()),
                short_size=('investment', lambda x: (
                        -x[(df.loc[x.index, 'quantile'] == short_quantile)] * df.loc[
                    x.index, 'ADJ_CLOSE_PRICE']).sum()),
                total_size=('investment', lambda x: (
                                                            x[(df.loc[x.index, 'quantile'] == long_quantile)] * df.loc[
                                                        x.index, 'ADJ_CLOSE_PRICE']).sum() +
                                                    (-x[(df.loc[x.index, 'quantile'] == short_quantile)] * df.loc[
                                                        x.index, 'ADJ_CLOSE_PRICE']).sum()),
                tvrshares=('tvr_shares', 'sum'),
                tvrvalues=('tvr_values', 'sum'),
                long_count=('vector', lambda x: (
                    x[df.loc[x.index, 'quantile'] == long_quantile].dropna().ge(
                        x.shift(1)[df.loc[x.index, 'quantile'] == long_quantile].dropna())).sum()),
                short_count=('vector', lambda x: (
                    x[df.loc[x.index, 'quantile'] == short_quantile].dropna().lt(
                        x.shift(1)[df.loc[x.index, 'quantile'] == short_quantile].dropna())).sum()),
                factor_exposure=('factor_exposure', 'mean'),
                factor_stability_coefficient=('factor_stability_coefficient', 'last')
            ).reset_index()

            # Filter the aggregated data for the specific date
            filtered_data = df[df['Date'] == pd.to_datetime('2006-06-07')]

            # Save the filtered data to the same output directory
            filtered_output_file = os.path.join(pnl_dir, f'filter_{value_factor}_results.csv')
            filtered_data.to_csv(filtered_output_file, index=False)

            aggregated['cum_pnl'] = aggregated['pnl'].cumsum() / (2 * initial_capital)
            aggregated['cum_long_pnl'] = aggregated['long_pnl'].cumsum() / (2 * initial_capital)
            aggregated['cum_short_pnl'] = aggregated['short_pnl'].cumsum() / (2 * initial_capital)

            # Extract year from Date
            aggregated['year'] = aggregated['Date'].dt.year

            # Calculate annualized return for each year
            annual_returns = aggregated.groupby('year')['pnl'].sum().reset_index()
            annual_returns.columns = ['year', 'annualized_return']
            annual_returns['annualized_return'] = annual_returns['annualized_return'] / (2 * initial_capital)
            df['TOTALVALUE'] = df['MvTotal_1'] * df['ADJ_CLOSE_PRICE']

            # Merge annualized return back to aggregated DataFrame
            aggregated = pd.merge(aggregated, annual_returns, on='year', how='left')
            # Calculate Sharpe Ratio
            daily_returns = (aggregated['pnl'] / 2 * initial_capital).fillna(0)
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            aggregated['sharpe_ratio'] = sharpe_ratio


            df['stocks_return'] = df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].pct_change()
            df['stocks_return'] = df['stocks_return'].fillna(0)

            # Calculate Information Coefficient (IC)
            aggregated['IC'] = aggregated['Date'].apply(
                lambda day: df.loc[df['Date'] == day, 'vector'].corr(df.loc[df['Date'] == day, 'stocks_return'])
            )
            aggregated['cum_IC'] = aggregated['IC'].cumsum()

            aggregated['IC_MAvg']= aggregated.groupby('year_month')['IC'].transform('mean')

            # Calculate Sharpe Ratio
            daily_returns = (aggregated['pnl'] / 2 * initial_capital).fillna(0)
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            aggregated['sharpe_ratio'] = sharpe_ratio

            def calculate_max_drawdown(df):
                df = df.sort_values(by='Date')  # Ensure data is sorted by date
                max_drawdown = 0
                for i in range(1, len(df)):
                    drawdown = aggregated.loc[i, 'pnl'] / initial_capital
                    if drawdown < max_drawdown:
                        max_drawdown = drawdown
                    df.loc[i, 'mdd'] = max_drawdown

                return df

            aggregated = calculate_max_drawdown(aggregated)

            def plot_combined_graphs(aggregated, df, initial_principal, vector):
                # Ensure TRADINGDAY_x is treated as datetime
                aggregated['Date'] = pd.to_datetime(aggregated['Date'], format='%Y%m%d')
                df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
                df['pct_return'] = np.log(df['ADJ_CLOSE_PRICE'] / df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].shift(1))
                cumulative_avg_return = df.groupby('Date')['pct_return'].mean().cumsum()

                # Calculate TVR ratio
                aggregated['tvr_ratio'] = aggregated['tvrvalues'] / initial_principal

                # Calculate excess returns
                aggregated[f'{vector}_excess_pnl'] = aggregated['cum_pnl'] - cumulative_avg_return.reindex(
                    aggregated['Date']).values
                aggregated[f'{vector}_excess_long_pnl'] = aggregated[
                                                              'cum_long_pnl'] - cumulative_avg_return.reindex(
                    aggregated['Date']).values
                aggregated[f'{vector}_excess_short_pnl'] = aggregated[
                                                               'cum_short_pnl'] - cumulative_avg_return.reindex(
                    aggregated['Date']).values

                fig, axs = plt.subplots(3, 1, figsize=(10, 8))

                # Plot cumulative PnL
                axs[0].plot(aggregated['Date'], aggregated['cum_pnl'], label='Cumulative PnL')
                axs[0].plot(aggregated['Date'], aggregated['cum_long_pnl'], label='Cumulative Long PnL')
                axs[0].plot(aggregated['Date'], aggregated['cum_short_pnl'], label='Cumulative Short PnL')
                axs[0].plot(cumulative_avg_return.index, cumulative_avg_return.values,
                            label='Cumulative Average Return')
                axs[0].xaxis.set_major_locator(mdates.YearLocator())
                axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                axs[0].set_title('Cumulative PnL, Long PnL, Short PnL, and Cumulative Average Return', fontsize='small')
                axs[0].set_xlabel('Trading Day', fontsize='small')
                axs[0].set_ylabel('Cumulative Return', fontsize='small')
                axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                axs[0].grid(True)

                # Plot histogram of TVR ratio
                axs[1].hist(aggregated['tvr_ratio'], bins=30, color='blue', edgecolor='black', alpha=0.7)
                axs[1].set_title('Distribution of TVR Ratio', fontsize='small')
                axs[1].set_xlabel('TVR Ratio', fontsize='small')
                axs[1].set_ylabel('Frequency', fontsize='small')
                axs[1].grid(True)

                # Plot annualized return
                axs[2].plot(aggregated['Date'], aggregated['annualized_return'], label='Annualized Return')
                axs[2].xaxis.set_major_locator(mdates.YearLocator())
                axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                axs[2].set_title('Annualized Return Over Time', fontsize='small')
                axs[2].set_xlabel('Trading Day', fontsize='small')
                axs[2].set_ylabel('Annualized Return', fontsize='small')
                axs[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                axs[2].grid(True)
                # Save the first plot
                plt.tight_layout()
                plt.savefig(f'{output_path}/{value_factor}_pnl.png')
                plt.close(fig)

                fig, axs = plt.subplots(2, 1, figsize=(14, 8))
                # Plot excess returns
                axs[0].plot(aggregated['Date'], aggregated[f'{vector}_excess_pnl'], label='Excess PnL')
                axs[0].plot(aggregated['Date'], aggregated[f'{vector}_excess_long_pnl'],
                            label='Excess Long PnL')
                axs[0].plot(aggregated['Date'], aggregated[f'{vector}_excess_short_pnl'],
                            label='Excess Short PnL')
                axs[0].xaxis.set_major_locator(mdates.YearLocator())
                axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                axs[0].set_title('Excess Returns (Overall, Long, Short)', fontsize='small')
                axs[0].set_xlabel('Trading Day', fontsize='small')
                axs[0].set_ylabel('Excess Return', fontsize='small')
                axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                axs[0].grid(True)
                # Plot excess returns
                axs[0].plot(aggregated['Date'], aggregated[f'{vector}_excess_pnl'], label='Excess PnL')
                axs[0].plot(aggregated['Date'], aggregated[f'{vector}_excess_long_pnl'], label='Excess Long PnL')
                axs[0].plot(aggregated['Date'], aggregated[f'{vector}_excess_short_pnl'], label='Excess Short PnL')

                # Set major locator and formatter for x-axis
                axs[0].xaxis.set_major_locator(mdates.YearLocator())
                axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

                # Set title, labels, legend, and grid for the first plot
                axs[0].set_title('Excess Returns (Overall, Long, Short)', fontsize='small')
                axs[0].set_xlabel('Trading Day', fontsize='small')
                axs[0].set_ylabel('Excess Return', fontsize='small')
                axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                axs[0].grid(True)

                # Plot cumulative IC
                axs[1].plot(aggregated['Date'], aggregated['cum_IC'], label='Cumulative IC')

                # Set major locator and formatter for x-axis
                axs[1].xaxis.set_major_locator(mdates.YearLocator())
                axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

                # Set title, labels, legend, and grid for the second plot
                axs[1].set_title('Cumulative IC', fontsize='small')
                axs[1].set_xlabel('Trading Day', fontsize='small')
                axs[1].set_ylabel('Cumulative IC', fontsize='small')
                axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                axs[1].grid(True)

                # Save the second plot
                plt.tight_layout()
                plt.savefig(f'{output_path}/{value_factor}_excess.png')
                plt.close(fig)

            plot_combined_graphs(aggregated, df, initial_capital, value_factor)

            def grouping_analysis(df, output_path, value_factor):
                # Sort the dataframe by vector and trading day for proper quantile grouping
                df = df.sort_values(by=['Date', 'vector'])

                # Initialize a list to store average returns per day
                average_returns_per_day_list = []
                average_size_per_day_list = []

                # Loop over each trading day to calculate daily average returns for each vector group
                for trading_day in df['Date'].unique():
                    daily_df = df[df['Date'] == trading_day].copy()
                    daily_df['vector_group'] = pd.qcut(daily_df['vector'], q=50, labels=False, duplicates='drop')
                    daily_average_returns = daily_df.groupby('vector_group')['pct_return'].mean().reset_index()
                    daily_average_size = daily_df.groupby('vector_group')['TOTALVALUE'].mean().reset_index()

                    daily_average_returns['Date'] = trading_day
                    daily_average_size['Date'] = trading_day
                    average_returns_per_day_list.append(daily_average_returns)
                    average_size_per_day_list.append(daily_average_size)

                # Concatenate the daily average returns into a single dataframe
                average_returns_per_day = pd.concat(average_returns_per_day_list)
                average_size_per_day = pd.concat(average_size_per_day_list)
                average_returns_per_day.set_index('Date', inplace=True)
                average_size_per_day.set_index('Date', inplace=True)

                # Calculate the overall average return and size for each vector group across all days
                average_returns = average_returns_per_day.groupby('vector_group')['pct_return'].mean().reset_index()
                average_size = average_size_per_day.groupby('vector_group')['TOTALVALUE'].mean().reset_index()

                # Rename columns for clarity
                average_returns.columns = ['vector_group', 'average_return']
                average_returns = average_returns.sort_values(by='vector_group', ascending=True)
                average_size.columns = ['vector_group', 'average_size']
                average_size = average_size.sort_values(by='vector_group', ascending=True)


                # Calculate industry and size exposure
                df['vector_abs'] = df['vector'].abs()
                df['vector_group'] = pd.qcut(df['vector'], q=50, labels=False, duplicates='drop')

                # Calculate the sum of the absolute values of vector allocations within each industry group
                industry_exposure = df.groupby(['vector_group', 'industry'])['vector_abs'].sum().reset_index()
                total_vector_abs_per_group = df.groupby('vector_group')['vector_abs'].sum().reset_index()
                industry_exposure = industry_exposure.merge(total_vector_abs_per_group, on='vector_group',
                                                            suffixes=('', '_group_total'))

                # The percentage is the allocation of vector abs within vector group within industry
                industry_exposure['allocation_percentage'] = industry_exposure['vector_abs'] / industry_exposure[
                    'vector_abs_group_total']

                # Pivot the table to have 'vector_group' as index and 'industry' as columns
                pivot_df_industry_exposure = industry_exposure.pivot(index='vector_group', columns='industry',
                                                                     values='allocation_percentage').fillna(0)

                # Calculate size and industry exposure
                aggregated['size_exposure'] = average_size['average_size'].std()
                aggregated['industry_exposure'] = industry_exposure['allocation_percentage'].std()

                # Create subplots
                fig, axs = plt.subplots(3, 1, figsize=(14, 15))

                # Plotting average returns
                axs[0].bar(average_returns['vector_group'], average_returns['average_return'], color='b', alpha=0.6)
                axs[0].set_xlabel('Vector Group', fontsize='small')
                axs[0].set_ylabel('Average Return', fontsize='small')
                axs[0].set_title('Average Return by Vector Group', fontsize='small')
                axs[0].grid(True)

                # Plotting average sizes
                axs[1].bar(average_size['vector_group'], average_size['average_size'], color='r', alpha=0.6)
                axs[1].set_xlabel('Vector Group', fontsize='small')
                axs[1].set_ylabel('Average Size', fontsize='small')
                axs[1].set_title('Average Size by Vector Group', fontsize='small')
                axs[1].grid(True)

                # Plotting average industry exposure percentages
                pivot_df_industry_exposure.plot(kind='bar', stacked=True, ax=axs[2])
                axs[2].set_title('Average Industry Exposure as Percentage of Total Allocation by Vector Group',
                                 fontsize='small')
                axs[2].set_ylabel('Exposure Percentage', fontsize='small')
                axs[2].set_xlabel('Vector Group', fontsize='small')
                axs[2].legend(title='Industry', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

                # Save the plots
                plt.tight_layout()
                plt.savefig(f'{output_path}/{value_factor}_grouping.png')
                plt.close(fig)

            grouping_analysis(df,self.output_path,value_factor)

            output_file = os.path.join(pnl_dir, f'{value_factor}_results_2.0.csv')
            print('latest_date up to',aggregated['Date'].max())
            aggregated.to_csv(output_file, index=False)


            overall_pnl = df['pnl'].sum()

            print(f"{value_factor} PnL: {overall_pnl}")
            gc.collect()

    def run(self):
        merged_financial_df = self.load_financial_data()
        qp_merged_df = self.load_qp_data()
        self.data = self.merge_data(qp_merged_df, merged_financial_df)
        self.data = self.forward_fill_columns(self.data)
        self.data = self.rearrange_columns(self.data)
        self.calculate_value_factor()
        self.calculate_profit_factor()
        self.calculate_Growth_Operation_factor()
        self.calculate_reversion_factor()
        self.calculate_liquidity_other_factor()
        self.evaluate_factor_IC()
        self.factor_combination()
        # output_filename = 'overall_alpha.csv'
        # output_dir = os.path.join(output_path, output_filename)
        # self.data.to_csv(output_dir, index=False)
        # self.construct_and_backtest_alpha()



# Define the paths
industry_file_path= '/Users/zw/Desktop/IndustryCitic_with_industry.xlsx'
qp_folder_path = '/Users/zw/Desktop/FinanceReport/FinanceReport/MergedCSVs/'
financial_data_folder = '/Users/zw/Desktop/FinanceReport/FinanceReport/Merged_Financial/'
output_path = '/Users/zw/Desktop/FinanceReport/FinanceReport/PNL'

# Instantiate and run the StockPitcher
stock_pitcher = StockPitcher(qp_folder_path, financial_data_folder, output_path,industry_file_path)
stock_pitcher.run()
