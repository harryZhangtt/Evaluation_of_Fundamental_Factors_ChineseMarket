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
    def __init__(self, qp_folder_path, financial_data_folder, output_path,input_path,industry_file_path):
        self.industry_data = pd.read_excel(industry_file_path, engine='openpyxl')
        self.qp_folder_path = qp_folder_path
        self.financial_data_folder = financial_data_folder
        self.output_path = output_path
        self.input_path= input_path
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

    def merge_csv_files(self):
        # List to store individual DataFrames
        data_frames = []

        # Iterate over all files in the input directory
        for file_name in os.listdir(self.input_path):
            if file_name.endswith('.csv'):
                print(f"Processing file: {file_name}")
                factor_name = file_name.split('.')[0]
                if factor_name != 'price':
                    self.all_factors.append(factor_name)
                file_path = os.path.join(self.input_path, file_name)
                # Read each CSV file and drop unnamed index columns
                temp_df = pd.read_csv(file_path).drop(columns=['Unnamed: 0'], errors='ignore')
                data_frames.append(temp_df)

        # Check if data_frames list is not empty
        if data_frames:
            # Start with the first DataFrame
            self.data = data_frames[0]

            # Merge each subsequent DataFrame horizontally
            for temp_df in data_frames[1:]:
                # Drop duplicate columns except the merge keys
                temp_df = temp_df.loc[:, ~temp_df.columns.duplicated()]
                self.data = pd.merge(self.data, temp_df, how='outer', on=['Date', 'SECU_CODE'])

            copy = self.data.iloc[:10000]
            copy.to_csv('/Users/zw/Desktop/FinanceReport/FinanceReport/PNL/all_data.csv', index=False)
        else:
            print("No CSV files found or the directory is empty.")

    def factor_combination(self):
        """ currently, the even distribution has the best performance, while ic weighted does not"""
        df = self.data.copy()
        methods=['even','IC']
        df.sort_values(by=['Date', 'SECU_CODE'], inplace=True)

        num = len(self.all_factors)


        def even_distribution(df):
            df['even_weight_alpha'] = 0.0  # Initialize with 0 for addition
            for factor in self.all_factors:
                print(factor)
                df[f'{factor}_filter'] = df[f'{factor}_filter'].fillna(0).astype(float)

                df['even_weight_alpha'] += np.where(df[f'{factor}_filter'] > 0, df[f'normalized_{factor}'],
                                                    -df[f'normalized_{factor}'])
                print(df['even_weight_alpha'])


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
                print(df['even_weight_alpha'])
                self.all_factors.append('even_weight_alpha')
            elif method=='IC':
                df= ic_weighted(df)
                df['IC_weighted_alpha_filter']=1
                self.all_factors.append('IC_weighted_alpha')
                print(df['IC_weighted_alpha'])




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
        df['pct_return']= df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].pct_change()

        # Define conditions
        def is_zdt(stock_code, pct_return):
            stock_code = str(stock_code)  # Ensure the stock code is a string
            # Check if the stock is in 创业板 or 科创板
            if stock_code.startswith('300') or stock_code.startswith('688'):
                # 创业板 and 科创板 have 20% limit
                return pct_return >= 0.2 or pct_return <= -0.2
            else:
                # Other stocks have 10% limit
                return pct_return >= 0.1 or pct_return <= -0.1

        # Apply filter
        df = df[~df.apply(lambda row: is_zdt(row['SECU_CODE'], row['pct_return']), axis=1)]
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
            df['TOTALVALUE'] = df['MvTotal_1']

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
        self.merge_csv_files()
        self.factor_combination()
        self.construct_and_backtest_alpha()



# Define the paths
industry_file_path= '/Users/zw/Desktop/IndustryCitic_with_industry.xlsx'
qp_folder_path = '/Users/zw/Desktop/FinanceReport/FinanceReport/MergedCSVs/'
financial_data_folder = '/Users/zw/Desktop/FinanceReport/FinanceReport/Merged_Financial/'
input_path = '/Users/zw/Desktop/FinanceReport/FinanceReport/PNL/Factor'
output_path='/Users/zw/Desktop/FinanceReport/FinanceReport/PNL'

# Instantiate and run the StockPitcher
stock_pitcher = StockPitcher(qp_folder_path, financial_data_folder, output_path,input_path,industry_file_path)
stock_pitcher.run()
