def get_l36m_production(well_list_df, production_df):
    """
    Calculate production for last 36 months in 12-month intervals for each well.
   
    Args:
        well_list_df (pd.DataFrame): DataFrame containing well list with API and Well Name.
        production_df (pd.DataFrame): DataFrame containing production data.
       
    Returns:
        pd.DataFrame: Summary of production in 12-month intervals for each well.
    """
    # Convert DataFrames to pandas if they aren't already
    well_list_df = well_list_df.to_pandas() if not isinstance(well_list_df, pd.DataFrame) else well_list_df
    production_df = production_df.to_pandas() if not isinstance(production_df, pd.DataFrame) else production_df
   
    # Ensure ProducingMonth is datetime
    production_df['ProducingMonth'] = pd.to_datetime(production_df['ProducingMonth'])

    # Sort production data by API and ProducingMonth descending
    production_df = production_df.sort_values(['API_UWI', 'ProducingMonth'], ascending=[True, False])

    # Add a ranking column for each well's production month (reset rank per API)
    production_df['MonthRank'] = production_df.groupby('API_UWI').cumcount()

    # Use pivot table to sum production grouped by API and month rank buckets
    production_summary = production_df.pivot_table(
        index='API_UWI',
        columns=pd.cut(production_df['MonthRank'], bins=[-1, 11, 23, 35], labels=['1_12', '13_24', '25_36']),
        values=['LiquidsProd_BBL', 'GasProd_MCF'],
        aggfunc='sum',
        fill_value=0
    )

    # Flatten multi-level columns
    production_summary.columns = ['_'.join(filter(None, col)).strip() for col in production_summary.columns]

    # Reset index to make 'API_UWI' a column again
    production_summary = production_summary.reset_index()

    # Merge well names from well_list_df
    results = pd.merge(
        well_list_df[['API', 'Well Name']],
        production_summary,
        left_on='API',
        right_on='API_UWI',
        how='left'
    ).drop(columns=['API_UWI'])

    # Rename columns for clarity
    results.rename(columns={
        'LiquidsProd_BBL_1_12': 'L12M_Liquids_BBL',
        'GasProd_MCF_1_12': 'L12M_Gas_MCF',
        'LiquidsProd_BBL_13_24': 'L13_24M_Liquids_BBL',
        'GasProd_MCF_13_24': 'L13_24M_Gas_MCF',
        'LiquidsProd_BBL_25_36': 'L25_36M_Liquids_BBL',
        'GasProd_MCF_25_36': 'L25_36M_Gas_MCF'
    }, inplace=True)

    return results.sort_values('API').reset_index(drop=True)
    

import pandas as pd
from datetime import datetime

def PDP_CF(start_date, 
           num_months, 
           PDP_pivot, 
           NGL_Yield, 
           price_df=None, 
           oil_diff_pct=0, 
           oil_diff_usd=0, 
           gas_diff_pct=0, 
           gas_diff_usd=0, 
           ngl_diff_pct=0, 
           ngl_diff_usd=0, 
           gas_shrink=0, 
           oil_gpt=0, 
           gas_gpt=0, 
           ngl_gpt=0, 
           oil_tax=0, 
           gas_tax=0, 
           ngl_tax=0,
           ad_val=0):
    
    # Convert num_months to an integer
    num_months = int(num_months)
    
    # Create a date range with monthly frequency
    date_range = pd.date_range(start=start_date, periods=num_months, freq="M")
    
    # Convert the date range to a DataFrame
    df = pd.DataFrame({"Date": date_range})
    
    # Convert the 'Date' column to datetime.date objects
    df['Date'] = df['Date'].dt.date
    
    # Ensure the 'EODATE' column in the PDP_pivot is in datetime.date format
    PDP_pivot['EODATE'] = pd.to_datetime(PDP_pivot['EODATE']).dt.date
    
    # Merge the df with PDP_pivot on the 'Date' and 'EODATE' columns
    df = pd.merge(df, PDP_pivot[['EODATE', 'Count of Chosen ID', 'Sum of Blended_Oil', 'Sum of Blended_Gas', 'Sum of Net_Oil', 'Sum of Net_Gas']],
                  left_on='Date', right_on='EODATE', how='left')
    
    # Drop the 'EODATE' column after merge
    df.drop('EODATE', axis=1, inplace=True)
    
    # Divide the blended and net columns by 1000
    df[['Sum of Blended_Oil', 'Sum of Blended_Gas', 'Sum of Net_Oil', 'Sum of Net_Gas']] /= 1000
    
    # Calculate Net NGL by dividing Net Gas by 1000 and multiplying by NGL Yield
    df['Net NGL'] = (df['Sum of Net_Gas'] / 1000) * NGL_Yield
    
    # Add 'Full Oil Price' column based on the lookup from the price DataFrame
    if price_df is not None:
        df = pd.merge(df, price_df[['Month', 'Live Oil']], left_on='Date', right_on='Month', how='left')
        df.rename(columns={'Live Oil': 'Full Oil Price'}, inplace=True)
        df.drop('Month', axis=1, inplace=True)
    else:
        df['Full Oil Price'] = 0
    
    # Add 'Full Gas Price' column based on the lookup from the price DataFrame
    if price_df is not None:
        df = pd.merge(df, price_df[['Month', 'Live Gas']], left_on='Date', right_on='Month', how='left')
        df.rename(columns={'Live Gas': 'Full Gas Price'}, inplace=True)
        df.drop('Month', axis=1, inplace=True)
    else:
        df['Full Gas Price'] = 0
    
    # Add 'Real Oil' column
    df['Real Oil'] = (df['Full Oil Price'] * oil_diff_pct) + oil_diff_usd
    
    # Add 'Real Gas' column
    df['Real Gas'] = (df['Full Gas Price'] * gas_diff_pct) + gas_diff_usd
    
    # Add 'Real NGL' column
    df['Real NGL'] = (df['Full Oil Price'] * ngl_diff_pct) + ngl_diff_usd
    
    # Add 'Gross Revenue' column
    df['Gross Revenue'] = (df['Sum of Net_Oil'] * df['Real Oil']) + (df['Sum of Net_Gas'] * df['Real Gas'] * gas_shrink) + (df['Net NGL'] * df['Real NGL'])
    
    # Add 'GPT' column
    df['GPT'] = (df['Sum of Net_Oil'] * oil_gpt) + (df['Sum of Net_Gas'] * gas_gpt) + (df['Net NGL'] * ngl_gpt)
    
    # Add 'Tax' column
    df['Sev Tax'] = (df['Sum of Net_Oil'] * df['Real Oil'] * oil_tax) + (df['Sum of Net_Gas'] * df['Real Gas'] * gas_shrink * gas_tax) + (df['Net NGL'] * df['Real NGL'] * ngl_tax)
    
    # Add 'AdVal Tax' column
    df['AdVal Tax'] = df['Gross Revenue'] * ad_val
    
    # Add 'Net Cash Flow' column
    df['Net Cash Flow'] = df['Gross Revenue'] - df['GPT'] - df['Sev Tax'] - df['AdVal Tax']
    
    return df

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def ensure_pandas_df(df):
    """
    Ensures the input is a Pandas DataFrame. Converts it if it's not already Pandas.
    """
    return df.to_pandas() if not isinstance(df, pd.DataFrame) else df

def calculate_fitted_decline_rate(
    production_df, 
    api_uwi, 
    cutoff_date, 
    months, 
    b_bounds=(0.03, 0.20), 
    default_decline=0.06
):
    """
    Calculate the fitted decline rate \( b \) for a single well's gas production data.

    Parameters:
        production_df (pd.DataFrame): Input DataFrame containing:
            - API_UWI: Unique well identifier.
            - ProducingMonth: Production date (datetime or string).
            - TotalProdMonths: Time in months since production started.
            - GasProd_MCF: Observed gas production rates (MCF).
        api_uwi (str): The API_UWI of the well to process.
        cutoff_date (datetime or str): Ignore the well if its last production date is before this date.
        months (int): Number of months to consider for the decline calculation.
        b_bounds (tuple): Bounds for decline rate \( b \) as (min_b, max_b).
        default_decline (float): Default decline rate if fitting fails.

    Returns:
        float: The fitted decline rate \( b \), or the default value if fitting fails.
    """
    # Ensure months is an integer
    try:
        months = int(months)
    except ValueError:
        raise ValueError(f"Invalid value for months: {months}. Expected an integer.")

    # Ensure production_df is a Pandas DataFrame
    production_df = ensure_pandas_df(production_df)

    # Ensure ProducingMonth is a datetime object
    production_df['ProducingMonth'] = pd.to_datetime(production_df['ProducingMonth'], errors='coerce')

    # Filter for the specified well
    well_data = production_df[production_df['API_UWI'] == api_uwi]

    # Skip wells with no data or where last production date is before cutoff
    if well_data.empty or well_data['ProducingMonth'].max() < pd.to_datetime(cutoff_date):
        return default_decline

    # Remove rows with invalid ProducingMonth
    well_data = well_data.dropna(subset=['ProducingMonth'])

    # Sort data by TotalProdMonths
    well_data = well_data.sort_values('TotalProdMonths').reset_index(drop=True)

    # Ensure we don't attempt to fetch more rows than available
    if len(well_data) < months:
        months = len(well_data)

    # Select the last `months` of data
    well_data = well_data.tail(months)

    # Ensure there is sufficient data for curve fitting
    if len(well_data) < 3:  # Minimum 3 points for meaningful fit
        return default_decline

    # Extract TotalProdMonths and GasProd_MCF values
    t = well_data['TotalProdMonths'].values
    q = well_data['GasProd_MCF'].values

    # Define the exponential decline function
    def exponential_decline(t, qi, b):
        return qi * np.exp(-b * t)

    try:
        # Perform curve fitting with bounds
        popt, _ = curve_fit(
            exponential_decline,
            t,
            q,
            bounds=([0, b_bounds[0]], [np.inf, b_bounds[1]]),
            maxfev=10000
        )
        _, b = popt  # Extract decline rate
    except (RuntimeError, ValueError):
        b = default_decline

    return b

def WellPlotter2(welllist_df, production_df):
    """
    Aggregates summed production data for wells marked with 'Y' and generates forecasts.
    Includes smoothing between historical and forecast data.
    """
    # Convert and clean DataFrames
    welllist_df = welllist_df.to_pandas() if not isinstance(welllist_df, pd.DataFrame) else welllist_df
    production_df = production_df.to_pandas() if not isinstance(production_df, pd.DataFrame) else production_df
    
    # Basic data preparation
    production_df.columns = production_df.columns.str.strip()
    welllist_df.columns = welllist_df.columns.str.strip()
    
    # Ensure ProducingMonth is datetime
    production_df['ProducingMonth'] = pd.to_datetime(production_df['ProducingMonth'])
    
    # Filter wells marked with 'Y' and merge with production data
    wells_with_y = welllist_df[welllist_df['Plot'] == 'Y']
    
    # First filter historical data by date range
    production_df = production_df[production_df['ProducingMonth'] >= '2018-01-01']
    
    merged_df = production_df.merge(wells_with_y, left_on='API_UWI', right_on='API', how='inner')
    
    # Aggregate production data
    aggregated_data = merged_df.groupby('ProducingMonth', as_index=False).agg({
        'API_UWI': 'nunique',
        'LiquidsProd_BBL': 'sum',
        'GasProd_MCF': 'sum',
        'WaterProd_BBL': 'sum',
        'CumLiquids_BBL': 'sum',
        'CumGas_MCF': 'sum',
        'CumWater_BBL': 'sum'
    }).rename(columns={'API_UWI': 'WellCount'})
    
    # Generate forecasts efficiently
    all_forecasts = []
    
    for _, well in wells_with_y.iterrows():
        try:
            fcst_start = pd.to_datetime(well['Fcst Start Date']).replace(day=1)
            oil_months = min(int(float(well['Oil Yrs Remain']) * 12), 600)
            gas_months = min(int(float(well['Gas Yrs Remain']) * 12), 600)
            
            # Get well's historical production
            well_hist = production_df[production_df['API_UWI'] == well['API']]
            
            if oil_months > 0 or gas_months > 0:
                dates = pd.date_range(start=fcst_start, periods=max(oil_months, gas_months), freq='MS')
                
                if gas_months > 0:
                    # Get last 3 months of gas production for this well
                    last_gas_prod = well_hist[well_hist['ProducingMonth'] <= fcst_start]['GasProd_MCF'].tail(3)
                    if not last_gas_prod.empty:
                        # Calculate average of last 3 months
                        avg_last_gas = last_gas_prod.mean()
                        # Use the minimum of input Qi and average last 3 months
                        gas_qi = min(float(well['Fcst Qi Gas']), avg_last_gas * 1.1)  # Allow 10% increase max
                    else:
                        gas_qi = float(well['Fcst Qi Gas'])
                    
                    gas_b = float(well['Gas Decline'])
                    gas_decline = 1 - np.exp(-gas_b/12)  # Convert annual b to monthly decline
                    gas_declines = (1 - gas_decline) ** np.arange(gas_months)
                    gas_fcst = gas_qi * gas_declines
                    well_fcst = pd.DataFrame({
                        'ProducingMonth': dates[:gas_months],
                        'GasFcst_MCF': gas_fcst
                    })
                    all_forecasts.append(well_fcst)
                
                if oil_months > 0:
                    # Get last 3 months of oil production for this well
                    last_oil_prod = well_hist[well_hist['ProducingMonth'] <= fcst_start]['LiquidsProd_BBL'].tail(3)
                    if not last_oil_prod.empty:
                        # Calculate average of last 3 months
                        avg_last_oil = last_oil_prod.mean()
                        # Use the minimum of input Qi and average last 3 months
                        oil_qi = min(float(well['Fcst Qi Oil']), avg_last_oil * 1.1)  # Allow 10% increase max
                    else:
                        oil_qi = float(well['Fcst Qi Oil'])
                    
                    oil_b = float(well['Oil Decline'])
                    oil_decline = 1 - np.exp(-oil_b/12)  # Convert annual b to monthly decline
                    oil_declines = (1 - oil_decline) ** np.arange(oil_months)
                    oil_fcst = oil_qi * oil_declines
                    well_fcst = pd.DataFrame({
                        'ProducingMonth': dates[:oil_months],
                        'OilFcst_BBL': oil_fcst
                    })
                    all_forecasts.append(well_fcst)
                
        except Exception as e:
            print(f"Error processing well: {str(e)}")
            continue
    
    if all_forecasts:
        fcst_df = pd.concat(all_forecasts, ignore_index=True)
        fcst_df = fcst_df.groupby('ProducingMonth', as_index=False).sum()
        
        # Ensure all forecast columns exist before merge
        for col in ['OilFcst_BBL', 'GasFcst_MCF']:
            if col not in fcst_df.columns:
                fcst_df[col] = np.nan
                
        final_df = pd.merge(aggregated_data, fcst_df, on='ProducingMonth', how='outer')
    else:
        final_df = aggregated_data.copy()
        final_df['OilFcst_BBL'] = np.nan
        final_df['GasFcst_MCF'] = np.nan
    
    # Sort by date for cumulative calculations
    final_df = final_df.sort_values('ProducingMonth')
    
    # Ensure all required columns exist
    required_cols = {
        'OilFcst_BBL': np.nan,
        'GasFcst_MCF': np.nan,
        'GasFcstCum_MCF': final_df['CumGas_MCF'].copy() if 'CumGas_MCF' in final_df.columns else np.nan,
        'OilFcstCum_BBL': final_df['CumLiquids_BBL'].copy() if 'CumLiquids_BBL' in final_df.columns else np.nan
    }
    
    for col, default_value in required_cols.items():
        if col not in final_df.columns:
            final_df[col] = default_value
    
    # Find last valid historical cumulative values
    last_hist_gas_idx = final_df['CumGas_MCF'].last_valid_index()
    last_hist_oil_idx = final_df['CumLiquids_BBL'].last_valid_index()
    
    if last_hist_gas_idx is not None:
        last_cum_gas = final_df.loc[last_hist_gas_idx, 'CumGas_MCF']
        # Calculate forward cumulative for gas forecast
        mask = final_df.index > last_hist_gas_idx
        final_df.loc[mask, 'GasFcstCum_MCF'] = last_cum_gas + final_df.loc[mask, 'GasFcst_MCF'].fillna(0).cumsum()
    
    if last_hist_oil_idx is not None:
        last_cum_oil = final_df.loc[last_hist_oil_idx, 'CumLiquids_BBL']
        # Calculate forward cumulative for oil forecast
        mask = final_df.index > last_hist_oil_idx
        final_df.loc[mask, 'OilFcstCum_BBL'] = last_cum_oil + final_df.loc[mask, 'OilFcst_BBL'].fillna(0).cumsum()
    
    # Replace zeros with NaN for all numeric columns except WellCount
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('WellCount')
    final_df[numeric_cols] = final_df[numeric_cols].replace(0, np.nan)
    
    # Filter to end of 2030
    final_df = final_df[final_df['ProducingMonth'] <= '2030-12-31']
    
    # Add End of Month date
    final_df['EOM_Date'] = final_df['ProducingMonth'].dt.to_period('M').dt.to_timestamp('M')
    
    # Column ordering
    col_order = ['ProducingMonth', 'EOM_Date', 'WellCount', 
                 'LiquidsProd_BBL', 'GasProd_MCF', 'WaterProd_BBL',
                 'CumLiquids_BBL', 'CumGas_MCF', 'CumWater_BBL',
                 'OilFcst_BBL', 'GasFcst_MCF', 'OilFcstCum_BBL', 'GasFcstCum_MCF']
    
    # Create blended columns
    final_df['Oil_Blend'] = final_df['LiquidsProd_BBL'].fillna(final_df['OilFcst_BBL'])
    final_df['Gas_Blend'] = final_df['GasProd_MCF'].fillna(final_df['GasFcst_MCF'])
    
    # Update column order to include new blended columns
    col_order.extend(['Oil_Blend', 'Gas_Blend'])
    
    # Ensure all columns exist and add any additional columns at the end
    existing_cols = [col for col in col_order if col in final_df.columns]
    additional_cols = [col for col in final_df.columns if col not in col_order]
    final_df = final_df[existing_cols + additional_cols]
    
    return final_df

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def ensure_pandas_df(df):
    """
    Ensures the input is a Pandas DataFrame. Converts it if it's not already Pandas.
    """
    return df.to_pandas() if not isinstance(df, pd.DataFrame) else df

def calculate_effective_decline(rates, months=None):
    """Calculate effective decline rate using trend-adjusted analysis"""
    if len(rates) < 6:
        return None
        
    # Use moving average to smooth the data
    window = min(6, len(rates))
    smoothed = pd.Series(rates).rolling(window=window, center=True).mean().dropna()
    
    if len(smoothed) < 2:
        return None
    
    # Calculate total decline using smoothed data
    start_rate = smoothed.iloc[0]
    end_rate = smoothed.iloc[-1]
    
    if start_rate <= 0 or end_rate <= 0:
        return None
        
    total_decline = (start_rate - end_rate) / start_rate
    time_periods = len(smoothed) - 1
    
    # Convert directly to annual decline rate
    annual_decline = 1 - np.power(1 - total_decline, 12/time_periods)
    return annual_decline

def Decline_Fit(
    production_df, 
    api_uwi, 
    cutoff_date, 
    months=12, 
    b_bounds=(0.03, 0.25),  # These are now annual bounds
    default_decline=0.06    # This is now annual default
):
    """
    Calculate the fitted decline rate (b) for a single well's gas production data.
    Returns annual decline rate as a fraction (e.g., 0.15 for 15% annual decline)
    """
    try:
        months = int(months)
    except ValueError:
        raise ValueError(f"Invalid value for months: {months}. Expected an integer.")
        
    # Ensure we have a pandas DataFrame
    production_df = ensure_pandas_df(production_df)
    production_df = pd.DataFrame(production_df)
    
    production_df['ProducingMonth'] = pd.to_datetime(production_df['ProducingMonth'])
    
    # Get well data and sort by date
    well_data = production_df[production_df['API_UWI'] == api_uwi]
    well_data = well_data.sort_values('ProducingMonth').reset_index(drop=True)
    
    if well_data.empty or well_data['ProducingMonth'].max() < pd.to_datetime(cutoff_date):
        return default_decline
        
    # Clean data and remove zeros/nulls
    well_data = well_data[well_data['GasProd_MCF'] > 0]
    
    # Find significant production increases that might indicate workovers
    well_data['pct_change'] = well_data['GasProd_MCF'].pct_change()
    workover_threshold = 0.5  # 50% increase
    workover_idx = well_data[well_data['pct_change'] > workover_threshold].index
    
    # If we found workovers, use data after the last workover
    if len(workover_idx) > 0:
        last_workover = workover_idx[-1]
        well_data = well_data.iloc[last_workover:]
        well_data = well_data.reset_index(drop=True)
    
    # Ensure we have enough data points
    if len(well_data) < 6:  # Need at least 6 months for moving average
        return default_decline
        
    # Use last 'months' of data
    if len(well_data) > months:
        well_data = well_data.tail(months)
    
    rates = well_data['GasProd_MCF'].values
    
    # Calculate effective decline rate (already in annual terms)
    annual_decline = calculate_effective_decline(rates, months)
    
    if annual_decline is None:
        return default_decline
        
    # Check if rate is reasonable
    if annual_decline < 0.06:  # Less than 6% annual decline
        return default_decline
    elif annual_decline > 0.98:  # More than 98% annual decline
        return default_decline
        
    return annual_decline

import pandas as pd
import re

def is_pandas(obj):
    """
    Check if an object is a pandas DataFrame.
    
    Parameters:
    obj: Any Python object
    
    Returns:
    bool: True if object is a pandas DataFrame, False otherwise
    """
    return isinstance(obj, pd.DataFrame)

def clean_rrc_numbers(df):
    """
    Clean RRC numbers by removing 'RRC #' prefix and any extra spaces.
    Only processes Property and RRC columns.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing at minimum 'Property' and 'RRC' columns
    
    Returns:
    pandas.DataFrame: DataFrame with only Property, RRC, and Clean_RRC columns
    
    Raises:
    TypeError: If input is not a pandas DataFrame
    """
    # Validate input is a pandas DataFrame
    if not is_pandas(df):
        raise TypeError("Input must be a pandas DataFrame")
    
    # Only select the two columns we need, ignoring all others
    result_df = df[['Property', 'RRC']].copy()
    
    def clean_rrc(rrc_num):
        if pd.isna(rrc_num):
            return None
        # Remove 'RRC #' and any extra spaces, keep only digits
        cleaned = re.sub(r'[^0-9]', '', str(rrc_num))
        return cleaned if cleaned else None
    
    # Apply cleaning function to create new column
    result_df['Clean_RRC'] = result_df['RRC'].apply(clean_rrc)
    
    return result_df

import pandas as pd
import re

def clean_rrc_column(df: pd.DataFrame, rrc_column: str = 'RRC') -> pd.DataFrame:
    """
    Clean RRC column by removing 'RRC#' prefix and creating a new cleaned column.
    Also validates if input is a pandas DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing the RRC column
    rrc_column : str, optional
        Name of the RRC column (default is 'RRC')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional cleaned RRC column
        
    Raises:
    -------
    TypeError
        If input is not a pandas DataFrame
    ValueError
        If specified RRC column doesn't exist in DataFrame
    """
    
    # Check if input is pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    # Check if RRC column exists
    if rrc_column not in df.columns:
        raise ValueError(f"Column '{rrc_column}' not found in DataFrame")
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Function to clean RRC values
    def clean_rrc(value):
        if pd.isna(value):
            return value
        # Convert to string if not already
        value = str(value)
        # Remove 'RRC#' or any non-digit characters
        cleaned = re.sub(r'[^\d]', '', value)
        return cleaned
    
    # Add new cleaned column
    new_column_name = f'{rrc_column}_cleaned'
    df_copy[new_column_name] = df_copy[rrc_column].apply(clean_rrc)
    
    return df_copy

# Example usage:
# df = pd.read_csv('your_file.csv')
# cleaned_df = clean_rrc_column(df, rrc_column='RRC')

def clean_rrc(value):
    """
    Clean a single RRC value by removing non-numeric characters,
    ensuring the output matches the original number exactly.
    
    Args:
        value: The RRC value to clean (can be string, int, or float)
        
    Returns:
        str: String containing only the numeric characters from the input,
             matching the original number of digits
    """
    if not value:  # Handle None or empty values
        return ""
    
    # Convert to string and extract only digits, avoiding float conversion
    str_value = str(value)
    cleaned = ''.join(char for char in str_value if char.isdigit())
    
    # If the string contains "RRC", we know it's an input value and not a computed value
    if 'RRC' in str_value:
        # Extract just the numeric portion
        cleaned = ''.join(char for char in str_value if char.isdigit())
    elif cleaned.endswith('0'):
        # Remove the trailing zero that was added
        cleaned = cleaned[:-1]
        
    return cleaned

def OwnerPlot(welllist_df, production_df):
    """
    Aggregates summed production data for wells marked with 'Y' and generates forecasts.
    Includes smoothing between historical and forecast data.
    """
    # Convert and clean DataFrames
    welllist_df = welllist_df.to_pandas() if not isinstance(welllist_df, pd.DataFrame) else welllist_df
    production_df = production_df.to_pandas() if not isinstance(production_df, pd.DataFrame) else production_df
    
    # Basic data preparation
    production_df.columns = production_df.columns.str.strip()
    welllist_df.columns = welllist_df.columns.str.strip()
    
    # Ensure ProducingMonth is datetime
    production_df['ProducingMonth'] = pd.to_datetime(production_df['ProducingMonth'])
    
    # Filter wells marked with 'Y' and merge with production data
    wells_with_y = welllist_df[welllist_df['Owner Plot'] == 'Y']
    
    # First filter historical data by date range
    production_df = production_df[production_df['ProducingMonth'] >= '2018-01-01']
    
    merged_df = production_df.merge(wells_with_y, left_on='API_UWI', right_on='API', how='inner')
    
    # Rest of the function remains the same...
    aggregated_data = merged_df.groupby('ProducingMonth', as_index=False).agg({
        'API_UWI': 'nunique',
        'LiquidsProd_BBL': 'sum',
        'GasProd_MCF': 'sum',
        'WaterProd_BBL': 'sum',
        'CumLiquids_BBL': 'sum',
        'CumGas_MCF': 'sum',
        'CumWater_BBL': 'sum'
    }).rename(columns={'API_UWI': 'WellCount'})
    
    # Generate forecasts efficiently
    all_forecasts = []
    
    for _, well in wells_with_y.iterrows():
        try:
            fcst_start = pd.to_datetime(well['Fcst Start Date']).replace(day=1)
            oil_months = min(int(float(well['Oil Yrs Remain']) * 12), 600)
            gas_months = min(int(float(well['Gas Yrs Remain']) * 12), 600)
            
            # Get well's historical production
            well_hist = production_df[production_df['API_UWI'] == well['API']]
            
            if oil_months > 0 or gas_months > 0:
                dates = pd.date_range(start=fcst_start, periods=max(oil_months, gas_months), freq='MS')
                
                if gas_months > 0:
                    # Get last 3 months of gas production for this well
                    last_gas_prod = well_hist[well_hist['ProducingMonth'] <= fcst_start]['GasProd_MCF'].tail(3)
                    if not last_gas_prod.empty:
                        # Calculate average of last 3 months
                        avg_last_gas = last_gas_prod.mean()
                        # Use the minimum of input Qi and average last 3 months
                        gas_qi = min(float(well['Fcst Qi Gas']), avg_last_gas * 1.1)  # Allow 10% increase max
                    else:
                        gas_qi = float(well['Fcst Qi Gas'])
                    
                    gas_b = float(well['Gas Decline'])
                    gas_decline = 1 - np.exp(-gas_b/12)  # Convert annual b to monthly decline
                    gas_declines = (1 - gas_decline) ** np.arange(gas_months)
                    gas_fcst = gas_qi * gas_declines
                    well_fcst = pd.DataFrame({
                        'ProducingMonth': dates[:gas_months],
                        'GasFcst_MCF': gas_fcst
                    })
                    all_forecasts.append(well_fcst)
                
                if oil_months > 0:
                    # Get last 3 months of oil production for this well
                    last_oil_prod = well_hist[well_hist['ProducingMonth'] <= fcst_start]['LiquidsProd_BBL'].tail(3)
                    if not last_oil_prod.empty:
                        # Calculate average of last 3 months
                        avg_last_oil = last_oil_prod.mean()
                        # Use the minimum of input Qi and average last 3 months
                        oil_qi = min(float(well['Fcst Qi Oil']), avg_last_oil * 1.1)  # Allow 10% increase max
                    else:
                        oil_qi = float(well['Fcst Qi Oil'])
                    
                    oil_b = float(well['Oil Decline'])
                    oil_decline = 1 - np.exp(-oil_b/12)  # Convert annual b to monthly decline
                    oil_declines = (1 - oil_decline) ** np.arange(oil_months)
                    oil_fcst = oil_qi * oil_declines
                    well_fcst = pd.DataFrame({
                        'ProducingMonth': dates[:oil_months],
                        'OilFcst_BBL': oil_fcst
                    })
                    all_forecasts.append(well_fcst)
                
        except Exception as e:
            print(f"Error processing well: {str(e)}")
            continue
    
    if all_forecasts:
        fcst_df = pd.concat(all_forecasts, ignore_index=True)
        fcst_df = fcst_df.groupby('ProducingMonth', as_index=False).sum()
        
        # Ensure all forecast columns exist before merge
        for col in ['OilFcst_BBL', 'GasFcst_MCF']:
            if col not in fcst_df.columns:
                fcst_df[col] = np.nan
                
        final_df = pd.merge(aggregated_data, fcst_df, on='ProducingMonth', how='outer')
    else:
        final_df = aggregated_data.copy()
        final_df['OilFcst_BBL'] = np.nan
        final_df['GasFcst_MCF'] = np.nan
    
    # Sort by date for cumulative calculations
    final_df = final_df.sort_values('ProducingMonth')
    
    # Ensure all required columns exist
    required_cols = {
        'OilFcst_BBL': np.nan,
        'GasFcst_MCF': np.nan,
        'GasFcstCum_MCF': final_df['CumGas_MCF'].copy() if 'CumGas_MCF' in final_df.columns else np.nan,
        'OilFcstCum_BBL': final_df['CumLiquids_BBL'].copy() if 'CumLiquids_BBL' in final_df.columns else np.nan
    }
    
    for col, default_value in required_cols.items():
        if col not in final_df.columns:
            final_df[col] = default_value
    
    # Find last valid historical cumulative values
    last_hist_gas_idx = final_df['CumGas_MCF'].last_valid_index()
    last_hist_oil_idx = final_df['CumLiquids_BBL'].last_valid_index()
    
    if last_hist_gas_idx is not None:
        last_cum_gas = final_df.loc[last_hist_gas_idx, 'CumGas_MCF']
        # Calculate forward cumulative for gas forecast
        mask = final_df.index > last_hist_gas_idx
        final_df.loc[mask, 'GasFcstCum_MCF'] = last_cum_gas + final_df.loc[mask, 'GasFcst_MCF'].fillna(0).cumsum()
    
    if last_hist_oil_idx is not None:
        last_cum_oil = final_df.loc[last_hist_oil_idx, 'CumLiquids_BBL']
        # Calculate forward cumulative for oil forecast
        mask = final_df.index > last_hist_oil_idx
        final_df.loc[mask, 'OilFcstCum_BBL'] = last_cum_oil + final_df.loc[mask, 'OilFcst_BBL'].fillna(0).cumsum()
    
    # Replace zeros with NaN for all numeric columns except WellCount
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('WellCount')
    final_df[numeric_cols] = final_df[numeric_cols].replace(0, np.nan)
    
    # Filter to end of 2030
    final_df = final_df[final_df['ProducingMonth'] <= '2030-12-31']
    
    # Add End of Month date column
    final_df['EOM_Date'] = final_df['ProducingMonth'].dt.to_period('M').dt.to_timestamp('M')
    
    # Create blended columns that use actual values when available, forecast values when not
    final_df['Oil_Blend'] = final_df['LiquidsProd_BBL'].fillna(final_df['OilFcst_BBL'])
    final_df['Gas_Blend'] = final_df['GasProd_MCF'].fillna(final_df['GasFcst_MCF'])
    
    # Column ordering
    col_order = ['ProducingMonth', 'EOM_Date', 'WellCount', 
                 'LiquidsProd_BBL', 'GasProd_MCF', 'WaterProd_BBL',
                 'CumLiquids_BBL', 'CumGas_MCF', 'CumWater_BBL',
                 'OilFcst_BBL', 'GasFcst_MCF', 'OilFcstCum_BBL', 'GasFcstCum_MCF',
                 'Oil_Blend', 'Gas_Blend']
    
    # Ensure all columns exist and add any additional columns at the end
    existing_cols = [col for col in col_order if col in final_df.columns]
    additional_cols = [col for col in final_df.columns if col not in col_order]
    final_df = final_df[existing_cols + additional_cols]
    
    return final_df

import pandas as pd
from datetime import datetime

def is_valid_dataframe(df, required_columns, df_name):
    """
    Validates if input is a pandas DataFrame with required columns.
    
    Args:
        df: Object to validate
        required_columns: List of column names that must be present
        df_name: Name of DataFrame for error messages
    
    Returns:
        bool: True if valid, raises ValueError if invalid
    """
    # Check if input is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{df_name} must be a pandas DataFrame")
    
    # Check if required columns are present
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {df_name}: {missing_cols}")
    
    return True

def OWNER_CF(PDP_pivot, Comm_Assump_df, price_df=None):
    """
    Calculates cash flows for oil and gas production based on input assumptions.
    
    Args:
        PDP_pivot: DataFrame containing production data
        Comm_Assump_df: DataFrame containing commercial assumptions
        price_df: Optional DataFrame containing price forecasts
    
    Returns:
        DataFrame: Monthly cash flows with detailed revenue and tax calculations
    """
    # Define required columns for each DataFrame
    pdp_required_cols = ['EOM_Date', 'WellCount', 'Oil_Blend', 
                        'Gas_Blend', 'Net_Oil', 'Net_Gas']
    
    comm_required_cols = ['Eff_Date', 'Fcst Months', 'Yield (bbl/MMcf)', 'Oil Basis (%)',
                         'Oil Basis ($/bbl)', 'Gas Basis (%)', 'Gas Basis ($/mcf)', 
                         'NGL Basis (%)', 'NGL Basis ($/bbl)', 'Shrink (% Remaining)',
                         'Oil GPT ($/bbl)', 'Gas GPT ($/mcf)', 'NGL GPT ($/mcf)', 'Sev Tax Oil (%)',
                         'Sev Tax Gas (%)', 'Sev Tax NGL (%)', 'Ad Val Tax']
    
    # Validate input DataFrames
    is_valid_dataframe(PDP_pivot, pdp_required_cols, "PDP_pivot")
    is_valid_dataframe(Comm_Assump_df, comm_required_cols, "Comm_Assump_df")
    
    if price_df is not None:
        price_required_cols = ['Month', 'Live Oil', 'Live Gas']
        is_valid_dataframe(price_df, price_required_cols, "price_df")
    
    # Extract parameters from Comm_Assump_df
    start_date = Comm_Assump_df['Eff_Date'].iloc[0]
    num_months = int(Comm_Assump_df['Fcst Months'].iloc[0])
    NGL_Yield = Comm_Assump_df['Yield (bbl/MMcf)'].iloc[0]
    oil_diff_pct = Comm_Assump_df['Oil Basis (%)'].iloc[0]
    oil_diff_usd = Comm_Assump_df['Oil Basis ($/bbl)'].iloc[0]
    gas_diff_pct = Comm_Assump_df['Gas Basis (%)'].iloc[0]
    gas_diff_usd = Comm_Assump_df['Gas Basis ($/mcf)'].iloc[0]
    ngl_diff_pct = Comm_Assump_df['NGL Basis (%)'].iloc[0]
    ngl_diff_usd = Comm_Assump_df['NGL Basis ($/bbl)'].iloc[0]
    gas_shrink = Comm_Assump_df['Shrink (% Remaining)'].iloc[0]
    oil_gpt = Comm_Assump_df['Oil GPT ($/bbl)'].iloc[0]
    gas_gpt = Comm_Assump_df['Gas GPT ($/mcf)'].iloc[0]
    ngl_gpt = Comm_Assump_df['NGL GPT ($/mcf)'].iloc[0]
    oil_tax = Comm_Assump_df['Sev Tax Oil (%)'].iloc[0]
    gas_tax = Comm_Assump_df['Sev Tax Gas (%)'].iloc[0]
    ngl_tax = Comm_Assump_df['Sev Tax NGL (%)'].iloc[0]
    ad_val = Comm_Assump_df['Ad Val Tax'].iloc[0]
    
    # Create date range with monthly frequency
    date_range = pd.date_range(start=start_date, periods=num_months, freq="M")
    
    # Convert to DataFrame
    df = pd.DataFrame({"Date": date_range})
    df['Date'] = df['Date'].dt.date
    
    # Ensure EODATE is in datetime.date format
    PDP_pivot['EODATE'] = pd.to_datetime(PDP_pivot['EODATE']).dt.date
    
    # Merge with PDP_pivot
    df = pd.merge(df, PDP_pivot[pdp_required_cols],
                 left_on='Date', right_on='EODATE', how='left')
    df.drop('EODATE', axis=1, inplace=True)
    
    # Scale production values
    df[['Sum of Blended_Oil', 'Sum of Blended_Gas', 'Sum of Net_Oil', 'Sum of Net_Gas']] /= 1000
    
    # Calculate Net NGL
    df['Net NGL'] = (df['Sum of Net_Gas'] / 1000) * NGL_Yield
    
    # Add price information if available
    if price_df is not None:
        # Add Oil Price
        df = pd.merge(df, price_df[['Month', 'Live Oil']], 
                     left_on='Date', right_on='Month', how='left')
        df.rename(columns={'Live Oil': 'Full Oil Price'}, inplace=True)
        df.drop('Month', axis=1, inplace=True)
        
        # Add Gas Price
        df = pd.merge(df, price_df[['Month', 'Live Gas']], 
                     left_on='Date', right_on='Month', how='left')
        df.rename(columns={'Live Gas': 'Full Gas Price'}, inplace=True)
        df.drop('Month', axis=1, inplace=True)
    else:
        df['Full Oil Price'] = 0
        df['Full Gas Price'] = 0
    
    # Calculate real prices with differentials
    df['Real Oil'] = (df['Full Oil Price'] * oil_diff_pct) + oil_diff_usd
    df['Real Gas'] = (df['Full Gas Price'] * gas_diff_pct) + gas_diff_usd
    df['Real NGL'] = (df['Full Oil Price'] * ngl_diff_pct) + ngl_diff_usd
    
    # Calculate revenue and taxes
    df['Gross Revenue'] = (
        (df['Sum of Net_Oil'] * df['Real Oil']) + 
        (df['Sum of Net_Gas'] * df['Real Gas'] * gas_shrink) + 
        (df['Net NGL'] * df['Real NGL'])
    )
    
    df['GPT'] = (
        (df['Sum of Net_Oil'] * oil_gpt) + 
        (df['Sum of Net_Gas'] * gas_gpt) + 
        (df['Net NGL'] * ngl_gpt)
    )
    
    df['Sev Tax'] = (
        (df['Sum of Net_Oil'] * df['Real Oil'] * oil_tax) + 
        (df['Sum of Net_Gas'] * df['Real Gas'] * gas_shrink * gas_tax) + 
        (df['Net NGL'] * df['Real NGL'] * ngl_tax)
    )
    
    df['AdVal Tax'] = df['Gross Revenue'] * ad_val
    
    # Calculate final cash flow
    df['Net Cash Flow'] = df['Gross Revenue'] - df['GPT'] - df['Sev Tax'] - df['AdVal Tax']
    
    return df

def OwnerPlotNet(welllist_df, production_df):
    """
    Aggregates summed production data for wells marked with 'Y' and generates forecasts.
    Includes both gross and net (NRI-adjusted) volumes for all production and forecast data.
    """
    # Convert and clean DataFrames
    welllist_df = welllist_df.to_pandas() if not isinstance(welllist_df, pd.DataFrame) else welllist_df
    production_df = production_df.to_pandas() if not isinstance(production_df, pd.DataFrame) else production_df
    
    # Basic data preparation
    production_df.columns = production_df.columns.str.strip()
    welllist_df.columns = welllist_df.columns.str.strip()
    
    # Ensure ProducingMonth is datetime
    production_df['ProducingMonth'] = pd.to_datetime(production_df['ProducingMonth'])
    
    # Filter wells marked with 'Y' and merge with production data
    wells_with_y = welllist_df[welllist_df['Owner Plot'] == 'Y']
    
    # First filter historical data by date range
    production_df = production_df[production_df['ProducingMonth'] >= '2018-01-01']
    
    # Merge production data with well list to get OwnerNRI
    merged_df = production_df.merge(wells_with_y[['API', 'OwnerNRI']], 
                                  left_on='API_UWI', 
                                  right_on='API', 
                                  how='inner')
    
    # Create net production columns while keeping original values
    merged_df['NetLiquidsProd_BBL'] = merged_df['LiquidsProd_BBL'] * merged_df['OwnerNRI']
    merged_df['NetGasProd_MCF'] = merged_df['GasProd_MCF'] * merged_df['OwnerNRI']
    merged_df['NetWaterProd_BBL'] = merged_df['WaterProd_BBL'] * merged_df['OwnerNRI']
    
    # Aggregate both gross and net production data
    aggregated_data = merged_df.groupby('ProducingMonth', as_index=False).agg({
        'API_UWI': 'nunique',
        'LiquidsProd_BBL': 'sum',
        'GasProd_MCF': 'sum',
        'WaterProd_BBL': 'sum',
        'NetLiquidsProd_BBL': 'sum',
        'NetGasProd_MCF': 'sum',
        'NetWaterProd_BBL': 'sum'
    }).rename(columns={'API_UWI': 'WellCount'})
    
    # Calculate cumulative volumes for both gross and net
    aggregated_data = aggregated_data.sort_values('ProducingMonth')
    # Gross cumulatives
    aggregated_data['CumLiquids_BBL'] = aggregated_data['LiquidsProd_BBL'].cumsum()
    aggregated_data['CumGas_MCF'] = aggregated_data['GasProd_MCF'].cumsum()
    aggregated_data['CumWater_BBL'] = aggregated_data['WaterProd_BBL'].cumsum()
    # Net cumulatives
    aggregated_data['NetCumLiquids_BBL'] = aggregated_data['NetLiquidsProd_BBL'].cumsum()
    aggregated_data['NetCumGas_MCF'] = aggregated_data['NetGasProd_MCF'].cumsum()
    aggregated_data['NetCumWater_BBL'] = aggregated_data['NetWaterProd_BBL'].cumsum()
    
    # Generate both gross and net forecasts
    all_forecasts = []
    
    for _, well in wells_with_y.iterrows():
        try:
            fcst_start = pd.to_datetime(well['Fcst Start Date']).replace(day=1)
            oil_months = min(int(float(well['Oil Yrs Remain']) * 12), 600)
            gas_months = min(int(float(well['Gas Yrs Remain']) * 12), 600)
            owner_nri = float(well['OwnerNRI'])
            
            # Get well's historical production
            well_hist = production_df[production_df['API_UWI'] == well['API']]
            
            if oil_months > 0 or gas_months > 0:
                dates = pd.date_range(start=fcst_start, periods=max(oil_months, gas_months), freq='MS')
                
                if gas_months > 0:
                    last_gas_prod = well_hist[well_hist['ProducingMonth'] <= fcst_start]['GasProd_MCF'].tail(3)
                    if not last_gas_prod.empty:
                        avg_last_gas = last_gas_prod.mean()
                        gas_qi = min(float(well['Fcst Qi Gas']), avg_last_gas * 1.1)
                    else:
                        gas_qi = float(well['Fcst Qi Gas'])
                    
                    gas_b = float(well['Gas Decline'])
                    gas_decline = 1 - np.exp(-gas_b/12)
                    gas_declines = (1 - gas_decline) ** np.arange(gas_months)
                    gas_fcst = gas_qi * gas_declines
                    net_gas_fcst = gas_fcst * owner_nri
                    well_fcst = pd.DataFrame({
                        'ProducingMonth': dates[:gas_months],
                        'GasFcst_MCF': gas_fcst,
                        'NetGasFcst_MCF': net_gas_fcst
                    })
                    all_forecasts.append(well_fcst)
                
                if oil_months > 0:
                    last_oil_prod = well_hist[well_hist['ProducingMonth'] <= fcst_start]['LiquidsProd_BBL'].tail(3)
                    if not last_oil_prod.empty:
                        avg_last_oil = last_oil_prod.mean()
                        oil_qi = min(float(well['Fcst Qi Oil']), avg_last_oil * 1.1)
                    else:
                        oil_qi = float(well['Fcst Qi Oil'])
                    
                    oil_b = float(well['Oil Decline'])
                    oil_decline = 1 - np.exp(-oil_b/12)
                    oil_declines = (1 - oil_decline) ** np.arange(oil_months)
                    oil_fcst = oil_qi * oil_declines
                    net_oil_fcst = oil_fcst * owner_nri
                    well_fcst = pd.DataFrame({
                        'ProducingMonth': dates[:oil_months],
                        'OilFcst_BBL': oil_fcst,
                        'NetOilFcst_BBL': net_oil_fcst
                    })
                    all_forecasts.append(well_fcst)
                
        except Exception as e:
            print(f"Error processing well: {str(e)}")
            continue
    
    if all_forecasts:
        fcst_df = pd.concat(all_forecasts, ignore_index=True)
        fcst_df = fcst_df.groupby('ProducingMonth', as_index=False).sum()
        
        # Ensure all forecast columns exist before merge
        for col in ['OilFcst_BBL', 'GasFcst_MCF', 'NetOilFcst_BBL', 'NetGasFcst_MCF']:
            if col not in fcst_df.columns:
                fcst_df[col] = np.nan
                
        final_df = pd.merge(aggregated_data, fcst_df, on='ProducingMonth', how='outer')
    else:
        final_df = aggregated_data.copy()
        final_df['OilFcst_BBL'] = np.nan
        final_df['GasFcst_MCF'] = np.nan
        final_df['NetOilFcst_BBL'] = np.nan
        final_df['NetGasFcst_MCF'] = np.nan
    
    # Sort by date for cumulative calculations
    final_df = final_df.sort_values('ProducingMonth')
    
    # Calculate forecast cumulatives for both gross and net
    final_df['GasFcstCum_MCF'] = final_df['GasFcst_MCF'].fillna(0).cumsum()
    final_df['OilFcstCum_BBL'] = final_df['OilFcst_BBL'].fillna(0).cumsum()
    final_df['NetGasFcstCum_MCF'] = final_df['NetGasFcst_MCF'].fillna(0).cumsum()
    final_df['NetOilFcstCum_BBL'] = final_df['NetOilFcst_BBL'].fillna(0).cumsum()
    
    # Replace zeros with NaN for all numeric columns except WellCount
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('WellCount')
    final_df[numeric_cols] = final_df[numeric_cols].replace(0, np.nan)
    
    # Filter to end of 2030
    final_df = final_df[final_df['ProducingMonth'] <= '2030-12-31']
    
    # Add End of Month date column
    final_df['EOM_Date'] = final_df['ProducingMonth'].dt.to_period('M').dt.to_timestamp('M')
    
    # Create blended columns for both gross and net
    final_df['Oil_Blend'] = final_df['LiquidsProd_BBL'].fillna(final_df['OilFcst_BBL'])
    final_df['Gas_Blend'] = final_df['GasProd_MCF'].fillna(final_df['GasFcst_MCF'])
    final_df['Net_Oil_Blend'] = final_df['NetLiquidsProd_BBL'].fillna(final_df['NetOilFcst_BBL'])
    final_df['Net_Gas_Blend'] = final_df['NetGasProd_MCF'].fillna(final_df['NetGasFcst_MCF'])
    
    # Column ordering - include both gross and net columns
    col_order = ['ProducingMonth', 'EOM_Date', 'WellCount',
                 # Gross Production
                 'LiquidsProd_BBL', 'GasProd_MCF', 'WaterProd_BBL',
                 'CumLiquids_BBL', 'CumGas_MCF', 'CumWater_BBL',
                 # Net Production
                 'NetLiquidsProd_BBL', 'NetGasProd_MCF', 'NetWaterProd_BBL',
                 'NetCumLiquids_BBL', 'NetCumGas_MCF', 'NetCumWater_BBL',
                 # Gross Forecasts
                 'OilFcst_BBL', 'GasFcst_MCF',
                 'OilFcstCum_BBL', 'GasFcstCum_MCF',
                 # Net Forecasts
                 'NetOilFcst_BBL', 'NetGasFcst_MCF',
                 'NetOilFcstCum_BBL', 'NetGasFcstCum_MCF',
                 # Gross Blends
                 'Oil_Blend', 'Gas_Blend',
                 # Net Blends
                 'Net_Oil_Blend', 'Net_Gas_Blend']
    
    # Ensure all columns exist and add any additional columns at the end
    existing_cols = [col for col in col_order if col in final_df.columns]
    additional_cols = [col for col in final_df.columns if col not in col_order]
    final_df = final_df[existing_cols + additional_cols]
    
    return final_df

def WellPlotterNET(welllist_df, production_df):
    """
    Aggregates production data for wells marked with 'Y', generates forecasts, 
    and calculates both gross and net (NRI-adjusted) volumes.
    """
    # Convert and clean DataFrames
    welllist_df = welllist_df.to_pandas() if not isinstance(welllist_df, pd.DataFrame) else welllist_df
    production_df = production_df.to_pandas() if not isinstance(production_df, pd.DataFrame) else production_df
    
    production_df.columns = production_df.columns.str.strip()
    welllist_df.columns = welllist_df.columns.str.strip()
    production_df['ProducingMonth'] = pd.to_datetime(production_df['ProducingMonth'])
    
    wells_with_y = welllist_df[welllist_df['Plot'] == 'Y']
    production_df = production_df[production_df['ProducingMonth'] >= '2018-01-01']
    
    # Merge and calculate net values
    merged_df = production_df.merge(wells_with_y[['API', 'OwnerNRI']], 
                                  left_on='API_UWI', 
                                  right_on='API', 
                                  how='inner')
    
    # Calculate net production values
    for col in ['LiquidsProd_BBL', 'GasProd_MCF', 'WaterProd_BBL', 
                'CumLiquids_BBL', 'CumGas_MCF', 'CumWater_BBL']:
        merged_df[f'Net{col}'] = merged_df[col] * merged_df['OwnerNRI']
    
    # Aggregate data
    agg_columns = {
        'API_UWI': 'nunique',
        'LiquidsProd_BBL': 'sum',
        'GasProd_MCF': 'sum',
        'WaterProd_BBL': 'sum',
        'CumLiquids_BBL': 'sum',
        'CumGas_MCF': 'sum',
        'CumWater_BBL': 'sum',
        'NetLiquidsProd_BBL': 'sum',
        'NetGasProd_MCF': 'sum',
        'NetWaterProd_BBL': 'sum',
        'NetCumLiquids_BBL': 'sum',
        'NetCumGas_MCF': 'sum',
        'NetCumWater_BBL': 'sum'
    }
    
    aggregated_data = merged_df.groupby('ProducingMonth', as_index=False).agg(agg_columns).rename(columns={'API_UWI': 'WellCount'})
    
    # Generate forecasts
    all_forecasts = []
    
    for _, well in wells_with_y.iterrows():
        try:
            fcst_start = pd.to_datetime(well['Fcst Start Date']).replace(day=1)
            oil_months = min(int(float(well['Oil Yrs Remain']) * 12), 600)
            gas_months = min(int(float(well['Gas Yrs Remain']) * 12), 600)
            owner_nri = float(well['OwnerNRI'])
            well_hist = production_df[production_df['API_UWI'] == well['API']]
            
            if oil_months > 0 or gas_months > 0:
                dates = pd.date_range(start=fcst_start, periods=max(oil_months, gas_months), freq='MS')
                
                if gas_months > 0:
                    last_gas_prod = well_hist[well_hist['ProducingMonth'] <= fcst_start]['GasProd_MCF'].tail(3)
                    gas_qi = float(well['Fcst Qi Gas'])
                    if not last_gas_prod.empty:
                        gas_qi = min(gas_qi, last_gas_prod.mean() * 1.1)
                    
                    gas_b = float(well['Gas Decline'])
                    gas_decline = 1 - np.exp(-gas_b/12)
                    gas_declines = (1 - gas_decline) ** np.arange(gas_months)
                    gas_fcst = gas_qi * gas_declines
                    net_gas_fcst = gas_fcst * owner_nri
                    well_fcst = pd.DataFrame({
                        'ProducingMonth': dates[:gas_months],
                        'GasFcst_MCF': gas_fcst,
                        'NetGasFcst_MCF': net_gas_fcst
                    })
                    all_forecasts.append(well_fcst)
                
                if oil_months > 0:
                    last_oil_prod = well_hist[well_hist['ProducingMonth'] <= fcst_start]['LiquidsProd_BBL'].tail(3)
                    oil_qi = float(well['Fcst Qi Oil'])
                    if not last_oil_prod.empty:
                        oil_qi = min(oil_qi, last_oil_prod.mean() * 1.1)
                    
                    oil_b = float(well['Oil Decline'])
                    oil_decline = 1 - np.exp(-oil_b/12)
                    oil_declines = (1 - oil_decline) ** np.arange(oil_months)
                    oil_fcst = oil_qi * oil_declines
                    net_oil_fcst = oil_fcst * owner_nri
                    well_fcst = pd.DataFrame({
                        'ProducingMonth': dates[:oil_months],
                        'OilFcst_BBL': oil_fcst,
                        'NetOilFcst_BBL': net_oil_fcst
                    })
                    all_forecasts.append(well_fcst)
                
        except Exception as e:
            print(f"Error processing well: {str(e)}")
            continue
    
    # Combine forecasts with historical data
    if all_forecasts:
        fcst_df = pd.concat(all_forecasts, ignore_index=True)
        fcst_df = fcst_df.groupby('ProducingMonth', as_index=False).sum()
        
        for col in ['OilFcst_BBL', 'GasFcst_MCF', 'NetOilFcst_BBL', 'NetGasFcst_MCF']:
            if col not in fcst_df.columns:
                fcst_df[col] = np.nan
                
        final_df = pd.merge(aggregated_data, fcst_df, on='ProducingMonth', how='outer')
    else:
        final_df = aggregated_data.copy()
        for col in ['OilFcst_BBL', 'GasFcst_MCF', 'NetOilFcst_BBL', 'NetGasFcst_MCF']:
            final_df[col] = np.nan
    
    final_df = final_df.sort_values('ProducingMonth')
    
    # Calculate forecast cumulatives
    for prefix in ['', 'Net']:
        gas_idx = final_df[f'{prefix}CumGas_MCF'].last_valid_index()
        oil_idx = final_df[f'{prefix}CumLiquids_BBL'].last_valid_index()
        
        if gas_idx is not None:
            last_cum_gas = final_df.loc[gas_idx, f'{prefix}CumGas_MCF']
            mask = final_df.index > gas_idx
            final_df.loc[mask, f'{prefix}GasFcstCum_MCF'] = (
                last_cum_gas + final_df.loc[mask, f'{prefix}GasFcst_MCF'].fillna(0).cumsum()
            )
        
        if oil_idx is not None:
            last_cum_oil = final_df.loc[oil_idx, f'{prefix}CumLiquids_BBL']
            mask = final_df.index > oil_idx
            final_df.loc[mask, f'{prefix}OilFcstCum_BBL'] = (
                last_cum_oil + final_df.loc[mask, f'{prefix}OilFcst_BBL'].fillna(0).cumsum()
            )
    
    # Clean up and format
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('WellCount')
    final_df[numeric_cols] = final_df[numeric_cols].replace(0, np.nan)
    
    final_df = final_df[final_df['ProducingMonth'] <= '2030-12-31']
    final_df['EOM_Date'] = final_df['ProducingMonth'].dt.to_period('M').dt.to_timestamp('M')
    
    # Create blended columns
    final_df['Oil_Blend'] = final_df['LiquidsProd_BBL'].fillna(final_df['OilFcst_BBL'])
    final_df['Gas_Blend'] = final_df['GasProd_MCF'].fillna(final_df['GasFcst_MCF'])
    final_df['Net_Oil_Blend'] = final_df['NetLiquidsProd_BBL'].fillna(final_df['NetOilFcst_BBL'])
    final_df['Net_Gas_Blend'] = final_df['NetGasProd_MCF'].fillna(final_df['NetGasFcst_MCF'])
    
    # Set column order
    col_order = ['ProducingMonth', 'EOM_Date', 'WellCount',
                 'LiquidsProd_BBL', 'GasProd_MCF', 'WaterProd_BBL',
                 'CumLiquids_BBL', 'CumGas_MCF', 'CumWater_BBL',
                 'NetLiquidsProd_BBL', 'NetGasProd_MCF', 'NetWaterProd_BBL',
                 'NetCumLiquids_BBL', 'NetCumGas_MCF', 'NetCumWater_BBL',
                 'OilFcst_BBL', 'GasFcst_MCF', 'OilFcstCum_BBL', 'GasFcstCum_MCF',
                 'NetOilFcst_BBL', 'NetGasFcst_MCF', 'NetOilFcstCum_BBL', 'NetGasFcstCum_MCF',
                 'Oil_Blend', 'Gas_Blend', 'Net_Oil_Blend', 'Net_Gas_Blend']
    
    existing_cols = [col for col in col_order if col in final_df.columns]
    additional_cols = [col for col in final_df.columns if col not in col_order]
    final_df = final_df[existing_cols + additional_cols]
    
    return final_df

import pandas as pd
import numpy as np

def is_valid_dataframe(df, required_cols, df_name):
    """Validates that DataFrame has required columns"""
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{df_name} missing required columns: {missing_cols}")

def WELL_CF(welllist_df, production_df, Comm_Assump_df, price_df=None):
    """
    Combines well plotting and cash flow calculations with well-specific NRI calculations.
    """
    # Convert inputs to pandas
    Comm_Assump_df = Comm_Assump_df.to_pandas() if not isinstance(Comm_Assump_df, pd.DataFrame) else Comm_Assump_df
    welllist_df = welllist_df.to_pandas() if not isinstance(welllist_df, pd.DataFrame) else welllist_df
    production_df = production_df.to_pandas() if not isinstance(production_df, pd.DataFrame) else production_df
    if price_df is not None:
        price_df = price_df.to_pandas() if not isinstance(price_df, pd.DataFrame) else price_df
    
    # Extract forecast parameters
    fcst_end_date = pd.to_datetime(Comm_Assump_df['Eff_Date'].iloc[0]) + pd.DateOffset(months=int(Comm_Assump_df['Fcst Months'].iloc[0]))

    def _well_plotter2(welllist_df, production_df, fcst_end_date):
        production_df.columns = production_df.columns.str.strip()
        welllist_df.columns = welllist_df.columns.str.strip()
        production_df['ProducingMonth'] = pd.to_datetime(production_df['ProducingMonth'])
        
        wells_with_y = welllist_df[welllist_df['Plot'] == 'Y']
        production_df = production_df[production_df['ProducingMonth'] >= '2018-01-01']
        
        merged_df = production_df.merge(wells_with_y[['API', 'OwnerNRI']], 
                                      left_on='API_UWI', 
                                      right_on='API', 
                                      how='inner')
        
        # Calculate net values
        merged_df['Net_LiquidsProd_BBL'] = (merged_df['LiquidsProd_BBL'] * merged_df['OwnerNRI']).round(2)
        merged_df['Net_GasProd_MCF'] = (merged_df['GasProd_MCF'] * merged_df['OwnerNRI']).round(2)
        merged_df['Net_WaterProd_BBL'] = (merged_df['WaterProd_BBL'] * merged_df['OwnerNRI']).round(2)
        
        agg_columns = {
            'API_UWI': 'nunique',
            'LiquidsProd_BBL': 'sum',
            'GasProd_MCF': 'sum',
            'WaterProd_BBL': 'sum',
            'Net_LiquidsProd_BBL': 'sum',
            'Net_GasProd_MCF': 'sum',
            'Net_WaterProd_BBL': 'sum'
        }
        
        aggregated_data = merged_df.groupby('ProducingMonth', as_index=False).agg(agg_columns).rename(columns={'API_UWI': 'WellCount'})
        for col in aggregated_data.select_dtypes(include=['float64']).columns:
            aggregated_data[col] = aggregated_data[col].round(2)
        
        all_forecasts = []
        active_wells_by_month = {}  # Track active wells per month
        
        for _, well in wells_with_y.iterrows():
            try:
                fcst_start = pd.to_datetime(well['Fcst Start Date']).replace(day=1)
                oil_months = min(int(float(well['Oil Yrs Remain']) * 12), 600)
                gas_months = min(int(float(well['Gas Yrs Remain']) * 12), 600)
                owner_nri = float(well['OwnerNRI'])
                well_hist = production_df[production_df['API_UWI'] == well['API']]
                
                if oil_months > 0 or gas_months > 0:
                    dates = pd.date_range(start=fcst_start, periods=max(oil_months, gas_months), freq='MS')
                    
                    # Track well in active months
                    for date in dates:
                        if date not in active_wells_by_month:
                            active_wells_by_month[date] = set()
                        active_wells_by_month[date].add(well['API'])
                    
                    if gas_months > 0:
                        last_gas_prod = well_hist[well_hist['ProducingMonth'] <= fcst_start]['GasProd_MCF'].tail(3)
                        gas_qi = float(well['Fcst Qi Gas'])
                        if not last_gas_prod.empty:
                            gas_qi = min(gas_qi, last_gas_prod.mean() * 1.1)
                        
                        gas_b = float(well['Gas Decline'])
                        gas_decline = 1 - np.exp(-gas_b/12)
                        gas_declines = (1 - gas_decline) ** np.arange(gas_months)
                        gas_fcst = gas_qi * gas_declines
                        net_gas_fcst = (gas_fcst * owner_nri).round(2)
                        gas_fcst = gas_fcst.round(2)
                        well_fcst = pd.DataFrame({
                            'ProducingMonth': dates[:gas_months],
                            'GasFcst_MCF': gas_fcst,
                            'Net_GasFcst_MCF': net_gas_fcst,
                            'API': well['API']
                        })
                        all_forecasts.append(well_fcst)
                    
                    if oil_months > 0:
                        last_oil_prod = well_hist[well_hist['ProducingMonth'] <= fcst_start]['LiquidsProd_BBL'].tail(3)
                        oil_qi = float(well['Fcst Qi Oil'])
                        if not last_oil_prod.empty:
                            oil_qi = min(oil_qi, last_oil_prod.mean() * 1.1)
                        
                        oil_b = float(well['Oil Decline'])
                        oil_decline = 1 - np.exp(-oil_b/12)
                        oil_declines = (1 - oil_decline) ** np.arange(oil_months)
                        oil_fcst = oil_qi * oil_declines
                        net_oil_fcst = (oil_fcst * owner_nri).round(2)
                        oil_fcst = oil_fcst.round(2)
                        well_fcst = pd.DataFrame({
                            'ProducingMonth': dates[:oil_months],
                            'OilFcst_BBL': oil_fcst,
                            'Net_OilFcst_BBL': net_oil_fcst,
                            'API': well['API']
                        })
                        all_forecasts.append(well_fcst)
            except Exception as e:
                print(f"Error processing well: {str(e)}")
                continue
        
        if all_forecasts:
            fcst_df = pd.concat(all_forecasts, ignore_index=True)
            
            # Create well count forecast
            well_count_forecast = pd.DataFrame({
                'ProducingMonth': list(active_wells_by_month.keys()),
                'ForecastWellCount': [len(wells) for wells in active_wells_by_month.values()]
            })
            
            fcst_df = fcst_df.groupby('ProducingMonth', as_index=False).sum()
            fcst_df = pd.merge(fcst_df, well_count_forecast, on='ProducingMonth', how='left')
            
            forecast_cols = ['OilFcst_BBL', 'GasFcst_MCF', 'Net_OilFcst_BBL', 'Net_GasFcst_MCF']
            for col in forecast_cols:
                if col not in fcst_df.columns:
                    fcst_df[col] = np.nan
                else:
                    fcst_df[col] = fcst_df[col].round(2)
            final_df = pd.merge(aggregated_data, fcst_df, on='ProducingMonth', how='outer')
            
            # Blend historical and forecast well counts
            final_df['WellCount'] = final_df['WellCount'].fillna(final_df['ForecastWellCount'])
            final_df.drop('ForecastWellCount', axis=1, inplace=True)
        else:
            final_df = aggregated_data.copy()
            final_df[['OilFcst_BBL', 'GasFcst_MCF', 'Net_OilFcst_BBL', 'Net_GasFcst_MCF']] = np.nan
        
        final_df = final_df.sort_values('ProducingMonth')
        
        # Calculate cumulative values
        prod_cols = [('LiquidsProd_BBL', 'OilFcst_BBL'), ('GasProd_MCF', 'GasFcst_MCF')]
        for prod_col, fcst_col in prod_cols:
            for prefix in ['', 'Net_']:
                final_df[f'{prefix}Cum{prod_col}'] = final_df[f'{prefix}{prod_col}'].cumsum().round(2)
                
                last_idx = final_df[f'{prefix}Cum{prod_col}'].last_valid_index()
                if last_idx is not None:
                    last_cum = final_df.loc[last_idx, f'{prefix}Cum{prod_col}']
                    mask = final_df.index > last_idx
                    final_df.loc[mask, f'{prefix}{fcst_col.replace("Fcst", "FcstCum")}'] = (
                        last_cum + final_df.loc[mask, f'{prefix}{fcst_col}'].fillna(0).cumsum()
                    ).round(2)
        
        numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('WellCount')
        final_df[numeric_cols] = final_df[numeric_cols].replace(0, np.nan)
        
        final_df = final_df[final_df['ProducingMonth'] <= fcst_end_date]
        final_df['EOM_Date'] = final_df['ProducingMonth'].dt.to_period('M').dt.to_timestamp('M')
        
        final_df['Oil_Blend'] = final_df['LiquidsProd_BBL'].fillna(final_df['OilFcst_BBL']).round(2)
        final_df['Gas_Blend'] = final_df['GasProd_MCF'].fillna(final_df['GasFcst_MCF']).round(2)
        final_df['Net_Oil_Blend'] = final_df['Net_LiquidsProd_BBL'].fillna(final_df['Net_OilFcst_BBL']).round(2)
        final_df['Net_Gas_Blend'] = final_df['Net_GasProd_MCF'].fillna(final_df['Net_GasFcst_MCF']).round(2)
        
        return final_df

    # Run WellPlotter2
    final_df = _well_plotter2(welllist_df, production_df, fcst_end_date)
    
    # Validate inputs for cash flow calculations
    comm_required_cols = ['Eff_Date', 'Fcst Months', 'Yield (bbl/MMcf)', 'Oil Basis (%)',
                         'Oil Basis ($/bbl)', 'Gas Basis (%)', 'Gas Basis ($/mcf)', 
                         'NGL Basis (%)', 'NGL Basis ($/bbl)', 'Shrink (% Remaining)',
                         'Oil GPT ($/bbl)', 'Gas GPT ($/mcf)', 'NGL GPT ($/mcf)', 
                         'Sev Tax Oil (%)', 'Sev Tax Gas (%)', 'Sev Tax NGL (%)', 'Ad Val Tax']
    
    is_valid_dataframe(Comm_Assump_df, comm_required_cols, "Comm_Assump_df")
    
    if price_df is not None:
        price_required_cols = ['Month', 'Live Oil', 'Live Gas']
        is_valid_dataframe(price_df, price_required_cols, "price_df")
    
    # Extract parameters
    params = {col: Comm_Assump_df[col].iloc[0] for col in comm_required_cols}
    
    # Create date range for cash flow calculations
    date_range = pd.date_range(start=params['Eff_Date'], 
                             periods=int(params['Fcst Months']), 
                             freq="M")
    
    # Initialize cash flow DataFrame
    cf_df = pd.DataFrame({"Date": date_range})
    cf_df['Date'] = cf_df['Date'].dt.date
    
    # Convert EOM_Date to date format for merging
    final_df['Date'] = pd.to_datetime(final_df['EOM_Date']).dt.date
    
    # Merge with production data
    cf_df = pd.merge(cf_df, 
                    final_df[['Date', 'WellCount', 'Oil_Blend', 'Gas_Blend', 
                             'Net_Oil_Blend', 'Net_Gas_Blend']],
                    on='Date', 
                    how='left')
    
    # Calculate Net NGL
    cf_df['Net_NGL'] = ((cf_df['Net_Gas_Blend']/1000) * params['Yield (bbl/MMcf)']).round(2)
    
    # Add price information
    if price_df is not None:
        price_df['Date'] = pd.to_datetime(price_df['Month']).dt.date
        cf_df = pd.merge(cf_df, price_df[['Date', 'Live Oil']], 
                        on='Date', how='left')
        cf_df.rename(columns={'Live Oil': 'Full Oil Price'}, inplace=True)
        
        cf_df = pd.merge(cf_df, price_df[['Date', 'Live Gas']], 
                        on='Date', how='left')
        cf_df.rename(columns={'Live Gas': 'Full Gas Price'}, inplace=True)
    else:
        cf_df['Full Oil Price'] = cf_df['Full Gas Price'] = 0
    
    # Calculate real prices with differentials
    cf_df['Real Oil'] = ((cf_df['Full Oil Price'] * params['Oil Basis (%)']) + params['Oil Basis ($/bbl)']).round(2)
    cf_df['Real Gas'] = ((cf_df['Full Gas Price'] * params['Gas Basis (%)']) + params['Gas Basis ($/mcf)']).round(2)
    cf_df['Real NGL'] = ((cf_df['Full Oil Price'] * params['NGL Basis (%)']) + params['NGL Basis ($/bbl)']).round(2)
    
    # Calculate revenue and taxes
    cf_df['Gross Revenue'] = (
        (cf_df['Net_Oil_Blend'] * cf_df['Real Oil']) +
        (cf_df['Net_Gas_Blend'] * cf_df['Real Gas'] * params['Shrink (% Remaining)']) +
        (cf_df['Net_NGL'] * cf_df['Real NGL'])
    ).round(2)
    
    cf_df['GPT'] = (
        (cf_df['Net_Oil_Blend'] * params['Oil GPT ($/bbl)']) +
        (cf_df['Net_Gas_Blend'] * params['Gas GPT ($/mcf)']) +
        (cf_df['Net_NGL'] * params['NGL GPT ($/mcf)'])
    ).round(2)
    
    cf_df['Sev Tax'] = (
        (cf_df['Net_Oil_Blend'] * cf_df['Real Oil'] * params['Sev Tax Oil (%)']) +
        (cf_df['Net_Gas_Blend'] * cf_df['Real Gas'] * params['Shrink (% Remaining)'] * params['Sev Tax Gas (%)']) +
        (cf_df['Net_NGL'] * cf_df['Real NGL'] * params['Sev Tax NGL (%)'])
    ).round(2)
    
    cf_df['AdVal Tax'] = (cf_df['Gross Revenue'] * params['Ad Val Tax']).round(2)
    
    # Calculate final cash flow
    cf_df['Net Cash Flow'] = (cf_df['Gross Revenue'] - cf_df['GPT'] - cf_df['Sev Tax'] - cf_df['AdVal Tax']).round(2)
    
    # Calculate time periods for discounting (in years)
    cf_df['Years'] = (pd.to_datetime(cf_df['Date']) - pd.to_datetime(params['Eff_Date'])).dt.days / 365.25
    
    # Calculate PVs and ROIs
    for rate in [0, 8, 10, 12, 14, 16, 18, 20]:
        # Calculate PV
        if rate == 0:
            cf_df.loc[0, f'PV{rate}'] = cf_df['Net Cash Flow'].sum().round(2)
        else:
            discount_factor = (1 + rate/100) ** -cf_df['Years']
            cf_df.loc[0, f'PV{rate}'] = (cf_df['Net Cash Flow'] * discount_factor).sum().round(2)
        
        # Calculate MOIC ROIs using this PV as investment base
        investment = abs(cf_df.loc[0, f'PV{rate}'])
        cumulative_cf = cf_df['Net Cash Flow'].cumsum()
        
        for period in [1, 3, 5, 10]:
            year_mask = cf_df['Years'] <= period
            period_cf = cumulative_cf[year_mask].iloc[-1] if any(year_mask) else 0
            cf_df.loc[0, f'ROI_{period}yr_PV{rate}'] = (period_cf / investment).round(2)

    return cf_df

import pandas as pd
import numpy as np

def is_valid_dataframe(df, required_cols, df_name):
   """Validates that DataFrame has required columns"""
   missing_cols = set(required_cols) - set(df.columns)
   if missing_cols:
       raise ValueError(f"{df_name} missing required columns: {missing_cols}")

def WELL_CF(welllist_df, production_df, Comm_Assump_df, price_df=None):
   """
   Combines well plotting and cash flow calculations with well-specific NRI calculations.
   """
   # Convert inputs to pandas
   Comm_Assump_df = Comm_Assump_df.to_pandas() if not isinstance(Comm_Assump_df, pd.DataFrame) else Comm_Assump_df
   welllist_df = welllist_df.to_pandas() if not isinstance(welllist_df, pd.DataFrame) else welllist_df
   production_df = production_df.to_pandas() if not isinstance(production_df, pd.DataFrame) else production_df
   if price_df is not None:
       price_df = price_df.to_pandas() if not isinstance(price_df, pd.DataFrame) else price_df
   
   # Extract forecast parameters
   fcst_end_date = pd.to_datetime(Comm_Assump_df['Eff_Date'].iloc[0]) + pd.DateOffset(months=int(Comm_Assump_df['Fcst Months'].iloc[0]))

   def _well_plotter2(welllist_df, production_df, fcst_end_date):
       production_df.columns = production_df.columns.str.strip()
       welllist_df.columns = welllist_df.columns.str.strip()
       production_df['ProducingMonth'] = pd.to_datetime(production_df['ProducingMonth'])
       
       wells_with_y = welllist_df[welllist_df['Plot'] == 'Y']
       production_df = production_df[production_df['ProducingMonth'] >= '2018-01-01']
       
       merged_df = production_df.merge(wells_with_y[['API', 'OwnerNRI']], 
                                     left_on='API_UWI', 
                                     right_on='API', 
                                     how='inner')
       
       merged_df['Net_LiquidsProd_BBL'] = (merged_df['LiquidsProd_BBL'] * merged_df['OwnerNRI']).round(2)
       merged_df['Net_GasProd_MCF'] = (merged_df['GasProd_MCF'] * merged_df['OwnerNRI']).round(2)
       merged_df['Net_WaterProd_BBL'] = (merged_df['WaterProd_BBL'] * merged_df['OwnerNRI']).round(2)
       
       agg_columns = {
           'API_UWI': 'nunique',
           'LiquidsProd_BBL': 'sum',
           'GasProd_MCF': 'sum',
           'WaterProd_BBL': 'sum',
           'Net_LiquidsProd_BBL': 'sum',
           'Net_GasProd_MCF': 'sum',
           'Net_WaterProd_BBL': 'sum'
       }
       
       aggregated_data = merged_df.groupby('ProducingMonth', as_index=False).agg(agg_columns).rename(columns={'API_UWI': 'WellCount'})
       for col in aggregated_data.select_dtypes(include=['float64']).columns:
           aggregated_data[col] = aggregated_data[col].round(2)
       
       all_forecasts = []
       active_wells_by_month = {}
       
       for _, well in wells_with_y.iterrows():
           try:
               fcst_start = pd.to_datetime(well['Fcst Start Date']).replace(day=1)
               oil_months = min(int(float(well['Oil Yrs Remain']) * 12), 600)
               gas_months = min(int(float(well['Gas Yrs Remain']) * 12), 600)
               owner_nri = float(well['OwnerNRI'])
               well_hist = production_df[production_df['API_UWI'] == well['API']]
               
               if oil_months > 0 or gas_months > 0:
                   dates = pd.date_range(start=fcst_start, periods=max(oil_months, gas_months), freq='MS')
                   
                   for date in dates:
                       if date not in active_wells_by_month:
                           active_wells_by_month[date] = set()
                       active_wells_by_month[date].add(well['API'])
                   
                   if gas_months > 0:
                       last_gas_prod = well_hist[well_hist['ProducingMonth'] <= fcst_start]['GasProd_MCF'].tail(3)
                       gas_qi = float(well['Fcst Qi Gas'])
                       if not last_gas_prod.empty:
                           gas_qi = min(gas_qi, last_gas_prod.mean() * 1.1)
                       
                       gas_b = float(well['Gas Decline'])
                       gas_decline = 1 - np.exp(-gas_b/12)
                       gas_declines = (1 - gas_decline) ** np.arange(gas_months)
                       gas_fcst = gas_qi * gas_declines
                       net_gas_fcst = (gas_fcst * owner_nri).round(2)
                       gas_fcst = gas_fcst.round(2)
                       well_fcst = pd.DataFrame({
                           'ProducingMonth': dates[:gas_months],
                           'GasFcst_MCF': gas_fcst,
                           'Net_GasFcst_MCF': net_gas_fcst,
                           'API': well['API']
                       })
                       all_forecasts.append(well_fcst)
                   
                   if oil_months > 0:
                       last_oil_prod = well_hist[well_hist['ProducingMonth'] <= fcst_start]['LiquidsProd_BBL'].tail(3)
                       oil_qi = float(well['Fcst Qi Oil'])
                       if not last_oil_prod.empty:
                           oil_qi = min(oil_qi, last_oil_prod.mean() * 1.1)
                       
                       oil_b = float(well['Oil Decline'])
                       oil_decline = 1 - np.exp(-oil_b/12)
                       oil_declines = (1 - oil_decline) ** np.arange(oil_months)
                       oil_fcst = oil_qi * oil_declines
                       net_oil_fcst = (oil_fcst * owner_nri).round(2)
                       oil_fcst = oil_fcst.round(2)
                       well_fcst = pd.DataFrame({
                           'ProducingMonth': dates[:oil_months],
                           'OilFcst_BBL': oil_fcst,
                           'Net_OilFcst_BBL': net_oil_fcst,
                           'API': well['API']
                       })
                       all_forecasts.append(well_fcst)
           except Exception as e:
               print(f"Error processing well: {str(e)}")
               continue
       
       if all_forecasts:
           fcst_df = pd.concat(all_forecasts, ignore_index=True)
           
           well_count_forecast = pd.DataFrame({
               'ProducingMonth': list(active_wells_by_month.keys()),
               'ForecastWellCount': [len(wells) for wells in active_wells_by_month.values()]
           })
           
           fcst_df = fcst_df.groupby('ProducingMonth', as_index=False).sum()
           fcst_df = pd.merge(fcst_df, well_count_forecast, on='ProducingMonth', how='left')
           
           forecast_cols = ['OilFcst_BBL', 'GasFcst_MCF', 'Net_OilFcst_BBL', 'Net_GasFcst_MCF']
           for col in forecast_cols:
               if col not in fcst_df.columns:
                   fcst_df[col] = np.nan
               else:
                   fcst_df[col] = fcst_df[col].round(2)
           final_df = pd.merge(aggregated_data, fcst_df, on='ProducingMonth', how='outer')
           
           final_df['WellCount'] = final_df['WellCount'].fillna(final_df['ForecastWellCount'])
           final_df.drop('ForecastWellCount', axis=1, inplace=True)
       else:
           final_df = aggregated_data.copy()
           final_df[['OilFcst_BBL', 'GasFcst_MCF', 'Net_OilFcst_BBL', 'Net_GasFcst_MCF']] = np.nan
       
       final_df = final_df.sort_values('ProducingMonth')
       
       prod_cols = [('LiquidsProd_BBL', 'OilFcst_BBL'), ('GasProd_MCF', 'GasFcst_MCF')]
       for prod_col, fcst_col in prod_cols:
           for prefix in ['', 'Net_']:
               final_df[f'{prefix}Cum{prod_col}'] = final_df[f'{prefix}{prod_col}'].cumsum().round(2)
               
               last_idx = final_df[f'{prefix}Cum{prod_col}'].last_valid_index()
               if last_idx is not None:
                   last_cum = final_df.loc[last_idx, f'{prefix}Cum{prod_col}']
                   mask = final_df.index > last_idx
                   final_df.loc[mask, f'{prefix}{fcst_col.replace("Fcst", "FcstCum")}'] = (
                       last_cum + final_df.loc[mask, f'{prefix}{fcst_col}'].fillna(0).cumsum()
                   ).round(2)
       
       numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
       numeric_cols.remove('WellCount')
       final_df[numeric_cols] = final_df[numeric_cols].replace(0, np.nan)
       
       final_df = final_df[final_df['ProducingMonth'] <= fcst_end_date]
       final_df['EOM_Date'] = final_df['ProducingMonth'].dt.to_period('M').dt.to_timestamp('M')
       
       final_df['Oil_Blend'] = final_df['LiquidsProd_BBL'].fillna(final_df['OilFcst_BBL']).round(2)
       final_df['Gas_Blend'] = final_df['GasProd_MCF'].fillna(final_df['GasFcst_MCF']).round(2)
       final_df['Net_Oil_Blend'] = final_df['Net_LiquidsProd_BBL'].fillna(final_df['Net_OilFcst_BBL']).round(2)
       final_df['Net_Gas_Blend'] = final_df['Net_GasProd_MCF'].fillna(final_df['Net_GasFcst_MCF']).round(2)
       
       return final_df

   # Run WellPlotter2
   final_df = _well_plotter2(welllist_df, production_df, fcst_end_date)
   
   # Validate inputs for cash flow calculations
   comm_required_cols = ['Eff_Date', 'Fcst Months', 'Yield (bbl/MMcf)', 'Oil Basis (%)',
                        'Oil Basis ($/bbl)', 'Gas Basis (%)', 'Gas Basis ($/mcf)', 
                        'NGL Basis (%)', 'NGL Basis ($/bbl)', 'Shrink (% Remaining)',
                        'Oil GPT ($/bbl)', 'Gas GPT ($/mcf)', 'NGL GPT ($/mcf)', 
                        'Sev Tax Oil (%)', 'Sev Tax Gas (%)', 'Sev Tax NGL (%)', 'Ad Val Tax']
   
   is_valid_dataframe(Comm_Assump_df, comm_required_cols, "Comm_Assump_df")
   
   if price_df is not None:
       price_required_cols = ['Month'] + [col for col in price_df.columns if '_OIL' in col or '_GAS' in col]
       is_valid_dataframe(price_df, price_required_cols, "price_df")
   
   # Extract parameters
   params = {col: Comm_Assump_df[col].iloc[0] for col in comm_required_cols}
   
   # Create date range for cash flow calculations
   date_range = pd.date_range(start=params['Eff_Date'], 
                            periods=int(params['Fcst Months']), 
                            freq="M")
   
   # Initialize cash flow DataFrame
   cf_df = pd.DataFrame({"Date": date_range})
   cf_df['Date'] = cf_df['Date'].dt.date
   
   # Convert EOM_Date to date format for merging
   final_df['Date'] = pd.to_datetime(final_df['EOM_Date']).dt.date
   
   # Merge with production data
   cf_df = pd.merge(cf_df, 
                   final_df[['Date', 'WellCount', 'Oil_Blend', 'Gas_Blend', 
                            'Net_Oil_Blend', 'Net_Gas_Blend']],
                   on='Date', 
                   how='left')
   
   # Calculate Net NGL
   cf_df['Net_NGL'] = ((cf_df['Net_Gas_Blend']/1000) * params['Yield (bbl/MMcf)']).round(2)
   
   # Add price information based on price deck
   if price_df is not None:
       price_df['Date'] = pd.to_datetime(price_df['Month']).dt.date
       cf_df = pd.merge(cf_df, price_df, on='Date', how='left')
       
       deck_num = str(int(welllist_df['Price deck'].iloc[0]))
       oil_col = f'{deck_num}_OIL'
       gas_col = f'{deck_num}_GAS'
       
       cf_df['Full Oil Price'] = cf_df[oil_col] if oil_col in cf_df.columns else 0
       cf_df['Full Gas Price'] = cf_df[gas_col] if gas_col in cf_df.columns else 0
   else:
       cf_df['Full Oil Price'] = cf_df['Full Gas Price'] = 0
   
   # Calculate real prices with differentials
   cf_df['Real Oil'] = ((cf_df['Full Oil Price'] * params['Oil Basis (%)']) + params['Oil Basis ($/bbl)']).round(2)
   cf_df['Real Gas'] = ((cf_df['Full Gas Price'] * params['Gas Basis (%)']) + params['Gas Basis ($/mcf)']).round(2)
   cf_df['Real NGL'] = ((cf_df['Full Oil Price'] * params['NGL Basis (%)']) + params['NGL Basis ($/bbl)']).round(2)
   
  # Calculate revenue and taxes
   cf_df['Gross Revenue'] = (
       (cf_df['Net_Oil_Blend'] * cf_df['Real Oil']) +
       (cf_df['Net_Gas_Blend'] * cf_df['Real Gas'] * params['Shrink (% Remaining)']) +
       (cf_df['Net_NGL'] * cf_df['Real NGL'])
   ).round(2)
   
   cf_df['GPT'] = (
       (cf_df['Net_Oil_Blend'] * params['Oil GPT ($/bbl)']) +
       (cf_df['Net_Gas_Blend'] * params['Gas GPT ($/mcf)']) +
       (cf_df['Net_NGL'] * params['NGL GPT ($/mcf)'])
   ).round(2)
   
   cf_df['Sev Tax'] = (
       (cf_df['Net_Oil_Blend'] * cf_df['Real Oil'] * params['Sev Tax Oil (%)']) +
       (cf_df['Net_Gas_Blend'] * cf_df['Real Gas'] * params['Shrink (% Remaining)'] * params['Sev Tax Gas (%)']) +
       (cf_df['Net_NGL'] * cf_df['Real NGL'] * params['Sev Tax NGL (%)'])
   ).round(2)
   
   cf_df['AdVal Tax'] = (cf_df['Gross Revenue'] * params['Ad Val Tax']).round(2)
   
   # Calculate final cash flow
   cf_df['Net Cash Flow'] = (cf_df['Gross Revenue'] - cf_df['GPT'] - cf_df['Sev Tax'] - cf_df['AdVal Tax']).round(2)
   
   # Calculate time periods for discounting (in years)
   cf_df['Years'] = (pd.to_datetime(cf_df['Date']) - pd.to_datetime(params['Eff_Date'])).dt.days / 365.25
   
   # Calculate PVs and ROIs
   for rate in [0, 8, 10, 12, 14, 16, 18, 20]:
       # Calculate PV
       if rate == 0:
           cf_df.loc[0, f'PV{rate}'] = cf_df['Net Cash Flow'].sum().round(2)
       else:
           discount_factor = (1 + rate/100) ** -cf_df['Years']
           cf_df.loc[0, f'PV{rate}'] = (cf_df['Net Cash Flow'] * discount_factor).sum().round(2)
       
       # Calculate MOIC ROIs using this PV as investment base
       investment = abs(cf_df.loc[0, f'PV{rate}'])
       cumulative_cf = cf_df['Net Cash Flow'].cumsum()
       
       for period in [1, 3, 5, 10]:
           year_mask = cf_df['Years'] <= period
           period_cf = cumulative_cf[year_mask].iloc[-1] if any(year_mask) else 0
           cf_df.loc[0, f'ROI_{period}yr_PV{rate}'] = (period_cf / investment).round(2)

   return cf_df


import pandas as pd
import numpy as np

def is_valid_dataframe(df, required_cols, df_name):
   """Validates that DataFrame has required columns"""
   missing_cols = set(required_cols) - set(df.columns)
   if missing_cols:
       raise ValueError(f"{df_name} missing required columns: {missing_cols}")

def WELL_CF_PD(welllist_df, production_df, Comm_Assump_df, price_df=None):
   """
   Combines well plotting and cash flow calculations with well-specific NRI calculations.
   """
   # Convert inputs to pandas
   Comm_Assump_df = Comm_Assump_df.to_pandas() if not isinstance(Comm_Assump_df, pd.DataFrame) else Comm_Assump_df
   welllist_df = welllist_df.to_pandas() if not isinstance(welllist_df, pd.DataFrame) else welllist_df
   production_df = production_df.to_pandas() if not isinstance(production_df, pd.DataFrame) else production_df
   if price_df is not None:
       price_df = price_df.to_pandas() if not isinstance(price_df, pd.DataFrame) else price_df
   
   # Extract forecast parameters
   fcst_end_date = pd.to_datetime(Comm_Assump_df['Eff_Date'].iloc[0]) + pd.DateOffset(months=int(Comm_Assump_df['Fcst Months'].iloc[0]))

   def _well_plotter2(welllist_df, production_df, fcst_end_date):
       production_df.columns = production_df.columns.str.strip()
       welllist_df.columns = welllist_df.columns.str.strip()
       production_df['ProducingMonth'] = pd.to_datetime(production_df['ProducingMonth'])
       
       wells_with_y = welllist_df[welllist_df['Plot'] == 'Y']
       production_df = production_df[production_df['ProducingMonth'] >= '2018-01-01']
       
       merged_df = production_df.merge(wells_with_y[['API', 'OwnerNRI']], 
                                     left_on='API_UWI', 
                                     right_on='API', 
                                     how='inner')
       
       merged_df['Net_LiquidsProd_BBL'] = (merged_df['LiquidsProd_BBL'] * merged_df['OwnerNRI']).round(2)
       merged_df['Net_GasProd_MCF'] = (merged_df['GasProd_MCF'] * merged_df['OwnerNRI']).round(2)
       merged_df['Net_WaterProd_BBL'] = (merged_df['WaterProd_BBL'] * merged_df['OwnerNRI']).round(2)
       
       agg_columns = {
           'API_UWI': 'nunique',
           'LiquidsProd_BBL': 'sum',
           'GasProd_MCF': 'sum',
           'WaterProd_BBL': 'sum',
           'Net_LiquidsProd_BBL': 'sum',
           'Net_GasProd_MCF': 'sum',
           'Net_WaterProd_BBL': 'sum'
       }
       
       aggregated_data = merged_df.groupby('ProducingMonth', as_index=False).agg(agg_columns).rename(columns={'API_UWI': 'WellCount'})
       for col in aggregated_data.select_dtypes(include=['float64']).columns:
           aggregated_data[col] = aggregated_data[col].round(2)
       
       all_forecasts = []
       active_wells_by_month = {}
       
       for _, well in wells_with_y.iterrows():
           try:
               fcst_start = pd.to_datetime(well['Fcst Start Date']).replace(day=1)
               oil_months = min(int(float(well['Oil Yrs Remain']) * 12), 600)
               gas_months = min(int(float(well['Gas Yrs Remain']) * 12), 600)
               owner_nri = float(well['OwnerNRI'])
               well_hist = production_df[production_df['API_UWI'] == well['API']]
               
               if oil_months > 0 or gas_months > 0:
                   dates = pd.date_range(start=fcst_start, periods=max(oil_months, gas_months), freq='MS')
                   
                   for date in dates:
                       if date not in active_wells_by_month:
                           active_wells_by_month[date] = set()
                       active_wells_by_month[date].add(well['API'])
                   
                   if gas_months > 0:
                       last_gas_prod = well_hist[well_hist['ProducingMonth'] <= fcst_start]['GasProd_MCF'].tail(3)
                       gas_qi = float(well['Fcst Qi Gas'])
                       if not last_gas_prod.empty:
                           gas_qi = min(gas_qi, last_gas_prod.mean() * 1.1)
                       
                       gas_b = float(well['Gas Decline'])
                       gas_decline = 1 - np.exp(-gas_b/12)
                       gas_declines = (1 - gas_decline) ** np.arange(gas_months)
                       gas_fcst = gas_qi * gas_declines
                       net_gas_fcst = (gas_fcst * owner_nri).round(2)
                       gas_fcst = gas_fcst.round(2)
                       well_fcst = pd.DataFrame({
                           'ProducingMonth': dates[:gas_months],
                           'GasFcst_MCF': gas_fcst,
                           'Net_GasFcst_MCF': net_gas_fcst,
                           'API': well['API']
                       })
                       all_forecasts.append(well_fcst)
                   
                   if oil_months > 0:
                       last_oil_prod = well_hist[well_hist['ProducingMonth'] <= fcst_start]['LiquidsProd_BBL'].tail(3)
                       oil_qi = float(well['Fcst Qi Oil'])
                       if not last_oil_prod.empty:
                           oil_qi = min(oil_qi, last_oil_prod.mean() * 1.1)
                       
                       oil_b = float(well['Oil Decline'])
                       oil_decline = 1 - np.exp(-oil_b/12)
                       oil_declines = (1 - oil_decline) ** np.arange(oil_months)
                       oil_fcst = oil_qi * oil_declines
                       net_oil_fcst = (oil_fcst * owner_nri).round(2)
                       oil_fcst = oil_fcst.round(2)
                       well_fcst = pd.DataFrame({
                           'ProducingMonth': dates[:oil_months],
                           'OilFcst_BBL': oil_fcst,
                           'Net_OilFcst_BBL': net_oil_fcst,
                           'API': well['API']
                       })
                       all_forecasts.append(well_fcst)
           except Exception as e:
               print(f"Error processing well: {str(e)}")
               continue
       
       if all_forecasts:
           fcst_df = pd.concat(all_forecasts, ignore_index=True)
           
           well_count_forecast = pd.DataFrame({
               'ProducingMonth': list(active_wells_by_month.keys()),
               'ForecastWellCount': [len(wells) for wells in active_wells_by_month.values()]
           })
           
           fcst_df = fcst_df.groupby('ProducingMonth', as_index=False).sum()
           fcst_df = pd.merge(fcst_df, well_count_forecast, on='ProducingMonth', how='left')
           
           forecast_cols = ['OilFcst_BBL', 'GasFcst_MCF', 'Net_OilFcst_BBL', 'Net_GasFcst_MCF']
           for col in forecast_cols:
               if col not in fcst_df.columns:
                   fcst_df[col] = np.nan
               else:
                   fcst_df[col] = fcst_df[col].round(2)
           final_df = pd.merge(aggregated_data, fcst_df, on='ProducingMonth', how='outer')
           
           final_df['WellCount'] = final_df['WellCount'].fillna(final_df['ForecastWellCount'])
           final_df.drop('ForecastWellCount', axis=1, inplace=True)
       else:
           final_df = aggregated_data.copy()
           final_df[['OilFcst_BBL', 'GasFcst_MCF', 'Net_OilFcst_BBL', 'Net_GasFcst_MCF']] = np.nan
       
       final_df = final_df.sort_values('ProducingMonth')
       
       prod_cols = [('LiquidsProd_BBL', 'OilFcst_BBL'), ('GasProd_MCF', 'GasFcst_MCF')]
       for prod_col, fcst_col in prod_cols:
           for prefix in ['', 'Net_']:
               final_df[f'{prefix}Cum{prod_col}'] = final_df[f'{prefix}{prod_col}'].cumsum().round(2)
               
               last_idx = final_df[f'{prefix}Cum{prod_col}'].last_valid_index()
               if last_idx is not None:
                   last_cum = final_df.loc[last_idx, f'{prefix}Cum{prod_col}']
                   mask = final_df.index > last_idx
                   final_df.loc[mask, f'{prefix}{fcst_col.replace("Fcst", "FcstCum")}'] = (
                       last_cum + final_df.loc[mask, f'{prefix}{fcst_col}'].fillna(0).cumsum()
                   ).round(2)
       
       numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
       numeric_cols.remove('WellCount')
       final_df[numeric_cols] = final_df[numeric_cols].replace(0, np.nan)
       
       final_df = final_df[final_df['ProducingMonth'] <= fcst_end_date]
       final_df['EOM_Date'] = final_df['ProducingMonth'].dt.to_period('M').dt.to_timestamp('M')
       
       final_df['Oil_Blend'] = final_df['LiquidsProd_BBL'].fillna(final_df['OilFcst_BBL']).round(2)
       final_df['Gas_Blend'] = final_df['GasProd_MCF'].fillna(final_df['GasFcst_MCF']).round(2)
       final_df['Net_Oil_Blend'] = final_df['Net_LiquidsProd_BBL'].fillna(final_df['Net_OilFcst_BBL']).round(2)
       final_df['Net_Gas_Blend'] = final_df['Net_GasProd_MCF'].fillna(final_df['Net_GasFcst_MCF']).round(2)
       
       return final_df

   # Run WellPlotter2
   final_df = _well_plotter2(welllist_df, production_df, fcst_end_date)
   
   # Validate inputs for cash flow calculations
   comm_required_cols = ['Eff_Date', 'Fcst Months', 'Yield (bbl/MMcf)', 'Oil Basis (%)',
                        'Oil Basis ($/bbl)', 'Gas Basis (%)', 'Gas Basis ($/mcf)', 
                        'NGL Basis (%)', 'NGL Basis ($/bbl)', 'Shrink (% Remaining)',
                        'Oil GPT ($/bbl)', 'Gas GPT ($/mcf)', 'NGL GPT ($/mcf)', 
                        'Sev Tax Oil (%)', 'Sev Tax Gas (%)', 'Sev Tax NGL (%)', 'Ad Val Tax']
   
   is_valid_dataframe(Comm_Assump_df, comm_required_cols, "Comm_Assump_df")
   
   if price_df is not None:
       price_required_cols = ['Month'] + [col for col in price_df.columns if '_OIL' in col or '_GAS' in col]
       is_valid_dataframe(price_df, price_required_cols, "price_df")
   
   # Extract parameters
   params = {col: Comm_Assump_df[col].iloc[0] for col in comm_required_cols}
   
   # Create date range for cash flow calculations
   date_range = pd.date_range(start=params['Eff_Date'], 
                            periods=int(params['Fcst Months']), 
                            freq="M")
   
   # Initialize cash flow DataFrame
   cf_df = pd.DataFrame({"Date": date_range})
   cf_df['Date'] = cf_df['Date'].dt.date
   
   # Convert EOM_Date to date format for merging
   final_df['Date'] = pd.to_datetime(final_df['EOM_Date']).dt.date
   
   # Merge with production data
   cf_df = pd.merge(cf_df, 
                   final_df[['Date', 'WellCount', 'Oil_Blend', 'Gas_Blend', 
                            'Net_Oil_Blend', 'Net_Gas_Blend']],
                   on='Date', 
                   how='left')
   
   # Calculate Net NGL
   cf_df['Net_NGL'] = ((cf_df['Net_Gas_Blend']/1000) * params['Yield (bbl/MMcf)']).round(2)
   
   # Add price information based on price deck
   if price_df is not None:
       price_df['Date'] = pd.to_datetime(price_df['Month']).dt.date
       cf_df = pd.merge(cf_df, price_df, on='Date', how='left')
       
       deck_num = str(int(welllist_df['Price Deck'].iloc[0]))
       oil_col = f'{deck_num}_OIL'
       gas_col = f'{deck_num}_GAS'
       
       cf_df['Full Oil Price'] = cf_df[oil_col] if oil_col in cf_df.columns else 0
       cf_df['Full Gas Price'] = cf_df[gas_col] if gas_col in cf_df.columns else 0
   else:
       cf_df['Full Oil Price'] = cf_df['Full Gas Price'] = 0
   
   # Calculate real prices with differentials
   cf_df['Real Oil'] = ((cf_df['Full Oil Price'] * params['Oil Basis (%)']) + params['Oil Basis ($/bbl)']).round(2)
   cf_df['Real Gas'] = ((cf_df['Full Gas Price'] * params['Gas Basis (%)']) + params['Gas Basis ($/mcf)']).round(2)
   cf_df['Real NGL'] = ((cf_df['Full Oil Price'] * params['NGL Basis (%)']) + params['NGL Basis ($/bbl)']).round(2)
   
  # Calculate revenue and taxes
   cf_df['Gross Revenue'] = (
       (cf_df['Net_Oil_Blend'] * cf_df['Real Oil']) +
       (cf_df['Net_Gas_Blend'] * cf_df['Real Gas'] * params['Shrink (% Remaining)']) +
       (cf_df['Net_NGL'] * cf_df['Real NGL'])
   ).round(2)
   
   cf_df['GPT'] = (
       (cf_df['Net_Oil_Blend'] * params['Oil GPT ($/bbl)']) +
       (cf_df['Net_Gas_Blend'] * params['Gas GPT ($/mcf)']) +
       (cf_df['Net_NGL'] * params['NGL GPT ($/mcf)'])
   ).round(2)
   
   cf_df['Sev Tax'] = (
       (cf_df['Net_Oil_Blend'] * cf_df['Real Oil'] * params['Sev Tax Oil (%)']) +
       (cf_df['Net_Gas_Blend'] * cf_df['Real Gas'] * params['Shrink (% Remaining)'] * params['Sev Tax Gas (%)']) +
       (cf_df['Net_NGL'] * cf_df['Real NGL'] * params['Sev Tax NGL (%)'])
   ).round(2)
   
   cf_df['AdVal Tax'] = (cf_df['Gross Revenue'] * params['Ad Val Tax']).round(2)
   
# Calculate final cash flow
   cf_df['Net Cash Flow'] = (cf_df['Gross Revenue'] - cf_df['GPT'] - cf_df['Sev Tax'] - cf_df['AdVal Tax']).round(2)
   
   # Calculate time periods for discounting (in years)
   cf_df['Years'] = (pd.to_datetime(cf_df['Date']) - pd.to_datetime(params['Eff_Date'])).dt.days / 365.25
   
   # Calculate summary statistics first
   summary_cols = {
       'Total_Oil_Blend': cf_df['Oil_Blend'].sum().round(2),
       'Total_Gas_Blend': cf_df['Gas_Blend'].sum().round(2),
       'Total_Net_Oil_Blend': cf_df['Net_Oil_Blend'].sum().round(2),
       'Total_Net_Gas_Blend': cf_df['Net_Gas_Blend'].sum().round(2),
       'Avg_Full_Oil_Price': cf_df['Full Oil Price'].mean().round(2),
       'Avg_Full_Gas_Price': cf_df['Full Gas Price'].mean().round(2),
       'Avg_Real_Oil_Price': cf_df['Real Oil'].mean().round(2),
       'Avg_Real_Gas_Price': cf_df['Real Gas'].mean().round(2),
       'Avg_Real_NGL_Price': cf_df['Real NGL'].mean().round(2),
       'Total_Revenue': cf_df['Gross Revenue'].sum().round(2),
       'Total_GPT': cf_df['GPT'].sum().round(2),
       'Total_Sev_Tax': cf_df['Sev Tax'].sum().round(2),
       'Total_AdVal_Tax': cf_df['AdVal Tax'].sum().round(2)
   }

   # Calculate PVs and ROIs
   pv_roi_results = {}
   for rate in [0, 8, 10, 12, 14, 16, 18, 20]:
       if rate == 0:
           pv_roi_results[f'PV{rate}'] = cf_df['Net Cash Flow'].sum().round(2)
       else:
           discount_factor = (1 + rate/100) ** -cf_df['Years']
           pv_roi_results[f'PV{rate}'] = (cf_df['Net Cash Flow'] * discount_factor).sum().round(2)
       
       investment = abs(pv_roi_results[f'PV{rate}'])
       cumulative_cf = cf_df['Net Cash Flow'].cumsum()
       
       for period in [1, 3, 5, 10]:
           year_mask = cf_df['Years'] <= period
           period_cf = cumulative_cf[year_mask].iloc[-1] if any(year_mask) else 0
           pv_roi_results[f'ROI_{period}yr_PV{rate}'] = (period_cf / investment).round(2)

   # Add summary stats and PV/ROI results in correct order on row 0
   for col, value in summary_cols.items():
       cf_df.loc[0, col] = value
   for col, value in pv_roi_results.items():
       cf_df.loc[0, col] = value

   # Drop price deck columns from output
   columns_to_keep = ['Date', 'WellCount', 'Oil_Blend', 'Gas_Blend', 'Net_Oil_Blend', 'Net_Gas_Blend', 
                     'Net_NGL', 'Full Oil Price', 'Full Gas Price', 'Real Oil', 'Real Gas', 'Real NGL',
                     'Gross Revenue', 'GPT', 'Sev Tax', 'AdVal Tax', 'Net Cash Flow', 'Years'] + \
                    list(summary_cols.keys()) + list(pv_roi_results.keys())
   
   cf_df = cf_df[columns_to_keep]

   return cf_df

import pandas as pd
import numpy as np

def calculate_single_well_forecast(well, production_df, fcst_end_date):
    fcst_start = pd.to_datetime(well['Fcst Start Date']).replace(day=1)
    oil_months = min(int(float(well['Oil Yrs Remain']) * 12), 600)
    gas_months = min(int(float(well['Gas Yrs Remain']) * 12), 600)
    well_hist = production_df[production_df['API_UWI'] == well['API']]
    
    dates = pd.date_range(start=fcst_start, end=fcst_end_date, freq='M')
    data = {'ProducingMonth': dates,
            'Oil_Blend': np.zeros(len(dates)),
            'Gas_Blend': np.zeros(len(dates))}
    
    if gas_months > 0:
        last_gas_prod = well_hist[well_hist['ProducingMonth'] <= fcst_start]['GasProd_MCF'].tail(3)
        gas_qi = float(well['Fcst Qi Gas'])
        if not last_gas_prod.empty:
            gas_qi = min(gas_qi, last_gas_prod.mean() * 1.1)
        
        gas_b = float(well['Gas Decline'])
        gas_decline = 1 - np.exp(-gas_b/12)
        n_periods = min(gas_months, len(dates))
        gas_declines = (1 - gas_decline) ** np.arange(n_periods)
        data['Gas_Blend'][:n_periods] = (gas_qi * gas_declines).round(2)
    
    if oil_months > 0:
        last_oil_prod = well_hist[well_hist['ProducingMonth'] <= fcst_start]['LiquidsProd_BBL'].tail(3)
        oil_qi = float(well['Fcst Qi Oil'])
        if not last_oil_prod.empty:
            oil_qi = min(oil_qi, last_oil_prod.mean() * 1.1)
        
        oil_b = float(well['Oil Decline'])
        oil_decline = 1 - np.exp(-oil_b/12)
        n_periods = min(oil_months, len(dates))
        oil_declines = (1 - oil_decline) ** np.arange(n_periods)
        data['Oil_Blend'][:n_periods] = (oil_qi * oil_declines).round(2)
    
    return pd.DataFrame(data)

def WELL_CF_SINGLE(welllist_df, production_df, Comm_Assump_df, price_df=None):
    Comm_Assump_df = Comm_Assump_df.to_pandas() if not isinstance(Comm_Assump_df, pd.DataFrame) else Comm_Assump_df
    welllist_df = welllist_df.to_pandas() if not isinstance(welllist_df, pd.DataFrame) else welllist_df
    production_df = production_df.to_pandas() if not isinstance(production_df, pd.DataFrame) else production_df
    if price_df is not None:
        price_df = price_df.to_pandas() if not isinstance(price_df, pd.DataFrame) else price_df
    
    production_df.columns = production_df.columns.str.strip()
    welllist_df.columns = welllist_df.columns.str.strip()
    production_df['ProducingMonth'] = pd.to_datetime(production_df['ProducingMonth'])
    
    # Extract parameters
    fcst_end_date = pd.to_datetime(Comm_Assump_df['Eff_Date'].iloc[0]) + pd.DateOffset(months=int(Comm_Assump_df['Fcst Months'].iloc[0]))
    params = {col: Comm_Assump_df[col].iloc[0] for col in Comm_Assump_df.columns}
    
    wells_with_y = welllist_df[welllist_df['Plot'] == 'Y']
    well_results = []
    
    for _, well in wells_with_y.iterrows():
        try:
            print(f"Processing well {well['API']}")
            well_hist = production_df[production_df['API_UWI'] == well['API']].copy()
            print(f"Historical data points: {len(well_hist)}")
            if len(well_hist) > 0:
                print(f"Sample production: Oil={well_hist['LiquidsProd_BBL'].mean():.2f}, Gas={well_hist['GasProd_MCF'].mean():.2f}")
            
            well_fcst = calculate_single_well_forecast(well, production_df, fcst_end_date)
            print(f"Forecast data points: {len(well_fcst)}")
            if len(well_fcst) > 0:
                print(f"Sample forecast: Oil={well_fcst['Oil_Blend'].mean():.2f}, Gas={well_fcst['Gas_Blend'].mean():.2f}")
            
            # Create cash flow DataFrame
            cf_df = pd.DataFrame()
            cf_df['Date'] = pd.date_range(start=params['Eff_Date'], periods=int(params['Fcst Months']), freq='M').date
            
            # Add production data
            well_fcst['Date'] = pd.to_datetime(well_fcst['ProducingMonth']).dt.date
            cf_df = pd.merge(cf_df, well_fcst[['Date', 'Oil_Blend', 'Gas_Blend']], on='Date', how='left')
            cf_df[['Oil_Blend', 'Gas_Blend']] = cf_df[['Oil_Blend', 'Gas_Blend']].fillna(0)
            
            # Calculate NGL
            cf_df['NGL'] = ((cf_df['Gas_Blend']/1000) * params['Yield (bbl/MMcf)']).round(2)
            
            # Add price data
            if price_df is not None:
                price_df.columns = price_df.columns.str.strip()
                price_df['Date'] = pd.to_datetime(price_df['Month']).dt.date
                cf_df = pd.merge(cf_df, price_df, on='Date', how='left')
                
                deck_num = str(int(well['Price Deck']))
                oil_col = next(col for col in cf_df.columns if f'{deck_num}_OIL' in col)
                gas_col = next(col for col in cf_df.columns if f'{deck_num}_GAS' in col)
                
                print(f"Price deck: {deck_num}, Available columns: {price_df.columns.tolist()}")
                cf_df['Full Oil Price'] = cf_df[oil_col].fillna(0)
                cf_df['Full Gas Price'] = cf_df[gas_col].fillna(0)
                print(f"Average prices: Oil=${cf_df['Full Oil Price'].mean():.2f}, Gas=${cf_df['Full Gas Price'].mean():.2f}")
            else:
                cf_df['Full Oil Price'] = cf_df['Full Gas Price'] = 0
            
            # Calculate real prices
            cf_df['Real Oil'] = ((cf_df['Full Oil Price'] * params['Oil Basis (%)']) + params['Oil Basis ($/bbl)']).round(2)
            cf_df['Real Gas'] = ((cf_df['Full Gas Price'] * params['Gas Basis (%)']) + params['Gas Basis ($/mcf)']).round(2)
            cf_df['Real NGL'] = ((cf_df['Full Oil Price'] * params['NGL Basis (%)']) + params['NGL Basis ($/bbl)']).round(2)
            
            # Calculate revenue and taxes
            cf_df['Gross Revenue'] = (
                (cf_df['Oil_Blend'] * cf_df['Real Oil']) +
                (cf_df['Gas_Blend'] * cf_df['Real Gas'] * params['Shrink (% Remaining)']) +
                (cf_df['NGL'] * cf_df['Real NGL'])
            ).round(2)
            
            print(f"\nRevenue Calculations:")
            print(f"Monthly Revenue Mean: ${cf_df['Gross Revenue'].mean():.2f}")
            print(f"Total Revenue: ${cf_df['Gross Revenue'].sum():.2f}")
            print(f"Sample month data:")
            print(cf_df[['Date', 'Oil_Blend', 'Gas_Blend', 'Real Oil', 'Real Gas', 'Gross Revenue']].head())
            
            cf_df['GPT'] = (
                (cf_df['Oil_Blend'] * params['Oil GPT ($/bbl)']) +
                (cf_df['Gas_Blend'] * params['Gas GPT ($/mcf)']) +
                (cf_df['NGL'] * params['NGL GPT ($/mcf)'])
            ).round(2)
            
            cf_df['Sev Tax'] = (
                (cf_df['Oil_Blend'] * cf_df['Real Oil'] * params['Sev Tax Oil (%)']) +
                (cf_df['Gas_Blend'] * cf_df['Real Gas'] * params['Shrink (% Remaining)'] * params['Sev Tax Gas (%)']) +
                (cf_df['NGL'] * cf_df['Real NGL'] * params['Sev Tax NGL (%)'])
            ).round(2)
            
            cf_df['AdVal Tax'] = (cf_df['Gross Revenue'] * params['Ad Val Tax']).round(2)
            cf_df['Net Cash Flow'] = (cf_df['Gross Revenue'] - cf_df['GPT'] - cf_df['Sev Tax'] - cf_df['AdVal Tax']).round(2)
            
            # Calculate time periods for discounting
            cf_df['Years'] = ((pd.to_datetime(cf_df['Date']) - pd.to_datetime(params['Eff_Date'])).dt.days / 365.25)
            
            # Calculate summary statistics
            well_summary = {
                'API': well['API'],
                'Total_Oil_Blend': cf_df['Oil_Blend'].sum().round(2),
                'Total_Gas_Blend': cf_df['Gas_Blend'].sum().round(2),
                'Total_Revenue': cf_df['Gross Revenue'].sum().round(2),
                'Total_GPT': cf_df['GPT'].sum().round(2),
                'Total_Sev_Tax': cf_df['Sev Tax'].sum().round(2),
                'Total_AdVal_Tax': cf_df['AdVal Tax'].sum().round(2)
            }
            
            # Calculate PVs
            for rate in [0, 8, 10, 12, 14, 16, 18, 20]:
                if rate == 0:
                    well_summary[f'PV{rate}'] = cf_df['Net Cash Flow'].sum().round(2)
                else:
                    discount_factor = (1 + rate/100) ** -cf_df['Years']
                    well_summary[f'PV{rate}'] = (cf_df['Net Cash Flow'] * discount_factor).sum().round(2)
            
            well_results.append(well_summary)
            
        except Exception as e:
            print(f"Error processing well {well['API']}: {str(e)}")
            continue
    
    if well_results:
        results_df = pd.DataFrame(well_results)
        return results_df
    else:
        return pd.DataFrame()

import pandas as pd
import numpy as np

def calculate_single_well_forecast(well, production_df, fcst_end_date):
    fcst_start = pd.to_datetime(well['Fcst Start Date']).replace(day=1)
    oil_months = min(int(float(well['Oil Yrs Remain']) * 12), 600)
    gas_months = min(int(float(well['Gas Yrs Remain']) * 12), 600)
    well_hist = production_df[production_df['API_UWI'] == well['API']]
    
    dates = pd.date_range(start=fcst_start, end=fcst_end_date, freq='M')
    data = {'ProducingMonth': dates,
            'Oil_Blend': np.zeros(len(dates)),
            'Gas_Blend': np.zeros(len(dates))}
    
    if gas_months > 0:
        last_gas_prod = well_hist[well_hist['ProducingMonth'] <= fcst_start]['GasProd_MCF'].tail(3)
        gas_qi = float(well['Fcst Qi Gas'])
        if not last_gas_prod.empty:
            gas_qi = min(gas_qi, last_gas_prod.mean() * 1.1)
        
        gas_b = float(well['Gas Decline'])
        gas_decline = 1 - np.exp(-gas_b/12)
        n_periods = min(gas_months, len(dates))
        gas_declines = (1 - gas_decline) ** np.arange(n_periods)
        data['Gas_Blend'][:n_periods] = (gas_qi * gas_declines).round(2)
    
    if oil_months > 0:
        last_oil_prod = well_hist[well_hist['ProducingMonth'] <= fcst_start]['LiquidsProd_BBL'].tail(3)
        oil_qi = float(well['Fcst Qi Oil'])
        if not last_oil_prod.empty:
            oil_qi = min(oil_qi, last_oil_prod.mean() * 1.1)
        
        oil_b = float(well['Oil Decline'])
        oil_decline = 1 - np.exp(-oil_b/12)
        n_periods = min(oil_months, len(dates))
        oil_declines = (1 - oil_decline) ** np.arange(n_periods)
        data['Oil_Blend'][:n_periods] = (oil_qi * oil_declines).round(2)
    
    return pd.DataFrame(data)

def WELL_CF_SINGLE_v2(welllist_df, production_df, Comm_Assump_df, price_df=None):
    # Preprocess DataFrames
    Comm_Assump_df = Comm_Assump_df.to_pandas() if not isinstance(Comm_Assump_df, pd.DataFrame) else Comm_Assump_df
    welllist_df = welllist_df.to_pandas() if not isinstance(welllist_df, pd.DataFrame) else welllist_df
    production_df = production_df.to_pandas() if not isinstance(production_df, pd.DataFrame) else production_df
    if price_df is not None:
        price_df = price_df.to_pandas() if not isinstance(price_df, pd.DataFrame) else price_df
    
    # Clean column names
    production_df.columns = production_df.columns.str.strip()
    welllist_df.columns = welllist_df.columns.str.strip()
    production_df['ProducingMonth'] = pd.to_datetime(production_df['ProducingMonth'])
    
    # Extract parameters
    fcst_end_date = pd.to_datetime(Comm_Assump_df['Eff_Date'].iloc[0]) + pd.DateOffset(months=int(Comm_Assump_df['Fcst Months'].iloc[0]))
    params = {col: Comm_Assump_df[col].iloc[0] for col in Comm_Assump_df.columns}
    
    wells_with_y = welllist_df[welllist_df['Plot'] == 'Y']
    well_results = []
    
    for _, well in wells_with_y.iterrows():
        try:
            print(f"Processing well {well['API']}")
            well_fcst = calculate_single_well_forecast(well, production_df, fcst_end_date)
            
            # Create cash flow DataFrame
            cf_df = pd.DataFrame()
            cf_df['Date'] = pd.date_range(start=params['Eff_Date'], periods=int(params['Fcst Months']), freq='M').date
            
            # Add production data
            well_fcst['Date'] = pd.to_datetime(well_fcst['ProducingMonth']).dt.date
            cf_df = pd.merge(cf_df, well_fcst[['Date', 'Oil_Blend', 'Gas_Blend']], on='Date', how='left')
            cf_df[['Oil_Blend', 'Gas_Blend']] = cf_df[['Oil_Blend', 'Gas_Blend']].fillna(0)
            
            # Calculate NGL
            cf_df['NGL'] = ((cf_df['Gas_Blend'] / 1000) * params['Yield (bbl/MMcf)']).round(2)
            
            # Add price data
            if price_df is not None:
                price_df.columns = price_df.columns.str.strip()
                price_df['Date'] = pd.to_datetime(price_df['Month']).dt.date
                cf_df = pd.merge(cf_df, price_df, on='Date', how='left')
                
                deck_num = str(int(well['Price Deck']))
                oil_col = next(col for col in cf_df.columns if f'{deck_num}_OIL' in col)
                gas_col = next(col for col in cf_df.columns if f'{deck_num}_GAS' in col)
                
                cf_df['Full Oil Price'] = cf_df[oil_col].fillna(0)
                cf_df['Full Gas Price'] = cf_df[gas_col].fillna(0)
            else:
                cf_df['Full Oil Price'] = cf_df['Full Gas Price'] = 0
            
            # Calculate real prices with separate NGL pricing
            cf_df['Real Oil'] = ((cf_df['Full Oil Price'] * params['Oil Basis (%)']) + params['Oil Basis ($/bbl)']).round(2)
            cf_df['Real Gas'] = ((cf_df['Full Gas Price'] * params['Gas Basis (%)']) + params['Gas Basis ($/mcf)']).round(2)
            cf_df['Real NGL'] = ((cf_df['Full Oil Price'] * params['NGL Basis (%)']) + params['NGL Basis ($/bbl)']).round(2)
            
            # Calculate revenue and taxes using separate NGL pricing
            cf_df['Gross Revenue'] = (
                (cf_df['Oil_Blend'] * cf_df['Real Oil']) +
                (cf_df['Gas_Blend'] * cf_df['Real Gas'] * params['Shrink (% Remaining)']) +
                (cf_df['NGL'] * cf_df['Real NGL'])
            ).round(2)
            
            cf_df['GPT'] = (
                (cf_df['Oil_Blend'] * params['Oil GPT ($/bbl)']) +
                (cf_df['Gas_Blend'] * params['Gas GPT ($/mcf)']) +
                (cf_df['NGL'] * params['NGL GPT ($/mcf)'])
            ).round(2)
            
            cf_df['Sev Tax'] = (
                (cf_df['Oil_Blend'] * cf_df['Real Oil'] * params['Sev Tax Oil (%)']) +
                (cf_df['Gas_Blend'] * cf_df['Real Gas'] * params['Sev Tax Gas (%)']) +
                (cf_df['NGL'] * cf_df['Real NGL'] * params['Sev Tax NGL (%)'])
            ).round(2)
            
            cf_df['AdVal Tax'] = (cf_df['Gross Revenue'] * params['Ad Val Tax']).round(2)
            cf_df['Net Cash Flow'] = (cf_df['Gross Revenue'] - cf_df['GPT'] - cf_df['Sev Tax'] - cf_df['AdVal Tax']).round(2)
            
            # Calculate time periods for discounting
            cf_df['Years'] = ((pd.to_datetime(cf_df['Date']) - pd.to_datetime(params['Eff_Date'])).dt.days / 365.25)
            
            # Calculate summary statistics
            well_summary = {
                'API': well['API'],
                'Total_Oil_Blend': cf_df['Oil_Blend'].sum().round(2),
                'Total_Gas_Blend': cf_df['Gas_Blend'].sum().round(2),
                'Total_Revenue': cf_df['Gross Revenue'].sum().round(2),
                'Total_GPT': cf_df['GPT'].sum().round(2),
                'Total_Sev_Tax': cf_df['Sev Tax'].sum().round(2),
                'Total_AdVal_Tax': cf_df['AdVal Tax'].sum().round(2)
            }
            
            # Calculate PVs
            for rate in [0, 8, 10, 12, 14, 16, 18, 20]:
                if rate == 0:
                    well_summary[f'PV{rate}'] = cf_df['Net Cash Flow'].sum().round(2)
                else:
                    discount_factor = (1 + rate/100) ** -cf_df['Years']
                    well_summary[f'PV{rate}'] = (cf_df['Net Cash Flow'] * discount_factor).sum().round(2)
            
            well_results.append(well_summary)
            
        except Exception as e:
            print(f"Error processing well {well['API']}: {str(e)}")
            continue
    
    if well_results:
        results_df = pd.DataFrame(well_results)
        return results_df
    else:
        return pd.DataFrame()

import pandas as pd
import numpy as np

def is_valid_dataframe(df, required_cols, df_name):
    """Validates that DataFrame has required columns"""
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{df_name} missing required columns: {missing_cols}")

def OWNER_CF_PD(welllist_df, production_df, Comm_Assump_df, price_df=None):
    """
    Combines well plotting and cash flow calculations with well-specific NRI calculations.
    """
    # Convert inputs to pandas
    Comm_Assump_df = Comm_Assump_df.to_pandas() if not isinstance(Comm_Assump_df, pd.DataFrame) else Comm_Assump_df
    welllist_df = welllist_df.to_pandas() if not isinstance(welllist_df, pd.DataFrame) else welllist_df
    production_df = production_df.to_pandas() if not isinstance(production_df, pd.DataFrame) else production_df
    if price_df is not None:
        price_df = price_df.to_pandas() if not isinstance(price_df, pd.DataFrame) else price_df
    
    # Extract forecast parameters
    fcst_end_date = pd.to_datetime(Comm_Assump_df['Eff_Date'].iloc[0]) + pd.DateOffset(months=int(Comm_Assump_df['Fcst Months'].iloc[0]))

    def _well_plotter2(welllist_df, production_df, fcst_end_date):
        production_df.columns = production_df.columns.str.strip()
        welllist_df.columns = welllist_df.columns.str.strip()
        production_df['ProducingMonth'] = pd.to_datetime(production_df['ProducingMonth'])
        
        wells_with_y = welllist_df[welllist_df['Owner Plot'] == 'Y']
        production_df = production_df[production_df['ProducingMonth'] >= '2018-01-01']
        
        merged_df = production_df.merge(wells_with_y[['API', 'OwnerNRI']], 
                                      left_on='API_UWI', 
                                      right_on='API', 
                                      how='inner')
        
        # Calculate net values
        merged_df['Net_LiquidsProd_BBL'] = (merged_df['LiquidsProd_BBL'] * merged_df['OwnerNRI']).round(2)
        merged_df['Net_GasProd_MCF'] = (merged_df['GasProd_MCF'] * merged_df['OwnerNRI']).round(2)
        merged_df['Net_WaterProd_BBL'] = (merged_df['WaterProd_BBL'] * merged_df['OwnerNRI']).round(2)
        
        agg_columns = {
            'API_UWI': 'nunique',
            'LiquidsProd_BBL': 'sum',
            'GasProd_MCF': 'sum',
            'WaterProd_BBL': 'sum',
            'Net_LiquidsProd_BBL': 'sum',
            'Net_GasProd_MCF': 'sum',
            'Net_WaterProd_BBL': 'sum'
        }
        
        aggregated_data = merged_df.groupby('ProducingMonth', as_index=False).agg(agg_columns).rename(columns={'API_UWI': 'WellCount'})
        for col in aggregated_data.select_dtypes(include=['float64']).columns:
            aggregated_data[col] = aggregated_data[col].round(2)
        
        all_forecasts = []
        active_wells_by_month = {}  # Track active wells per month
        
        for _, well in wells_with_y.iterrows():
            try:
                fcst_start = pd.to_datetime(well['Fcst Start Date']).replace(day=1)
                oil_months = min(int(float(well['Oil Yrs Remain']) * 12), 600)
                gas_months = min(int(float(well['Gas Yrs Remain']) * 12), 600)
                owner_nri = float(well['OwnerNRI'])
                well_hist = production_df[production_df['API_UWI'] == well['API']]
                
                if oil_months > 0 or gas_months > 0:
                    dates = pd.date_range(start=fcst_start, periods=max(oil_months, gas_months), freq='MS')
                    
                    # Track well in active months
                    for date in dates:
                        if date not in active_wells_by_month:
                            active_wells_by_month[date] = set()
                        active_wells_by_month[date].add(well['API'])
                    
                    if gas_months > 0:
                        last_gas_prod = well_hist[well_hist['ProducingMonth'] <= fcst_start]['GasProd_MCF'].tail(3)
                        gas_qi = float(well['Fcst Qi Gas'])
                        if not last_gas_prod.empty:
                            gas_qi = min(gas_qi, last_gas_prod.mean() * 1.1)
                        
                        gas_b = float(well['Gas Decline'])
                        gas_decline = 1 - np.exp(-gas_b/12)
                        gas_declines = (1 - gas_decline) ** np.arange(gas_months)
                        gas_fcst = gas_qi * gas_declines
                        net_gas_fcst = (gas_fcst * owner_nri).round(2)
                        gas_fcst = gas_fcst.round(2)
                        well_fcst = pd.DataFrame({
                            'ProducingMonth': dates[:gas_months],
                            'GasFcst_MCF': gas_fcst,
                            'Net_GasFcst_MCF': net_gas_fcst,
                            'API': well['API']
                        })
                        all_forecasts.append(well_fcst)
                    
                    if oil_months > 0:
                        last_oil_prod = well_hist[well_hist['ProducingMonth'] <= fcst_start]['LiquidsProd_BBL'].tail(3)
                        oil_qi = float(well['Fcst Qi Oil'])
                        if not last_oil_prod.empty:
                            oil_qi = min(oil_qi, last_oil_prod.mean() * 1.1)
                        
                        oil_b = float(well['Oil Decline'])
                        oil_decline = 1 - np.exp(-oil_b/12)
                        oil_declines = (1 - oil_decline) ** np.arange(oil_months)
                        oil_fcst = oil_qi * oil_declines
                        net_oil_fcst = (oil_fcst * owner_nri).round(2)
                        oil_fcst = oil_fcst.round(2)
                        well_fcst = pd.DataFrame({
                            'ProducingMonth': dates[:oil_months],
                            'OilFcst_BBL': oil_fcst,
                            'Net_OilFcst_BBL': net_oil_fcst,
                            'API': well['API']
                        })
                        all_forecasts.append(well_fcst)
            except Exception as e:
                print(f"Error processing well: {str(e)}")
                continue
        
        if all_forecasts:
            fcst_df = pd.concat(all_forecasts, ignore_index=True)
            
            # Create well count forecast
            well_count_forecast = pd.DataFrame({
                'ProducingMonth': list(active_wells_by_month.keys()),
                'ForecastWellCount': [len(wells) for wells in active_wells_by_month.values()]
            })
            
            fcst_df = fcst_df.groupby('ProducingMonth', as_index=False).sum()
            fcst_df = pd.merge(fcst_df, well_count_forecast, on='ProducingMonth', how='left')
            
            forecast_cols = ['OilFcst_BBL', 'GasFcst_MCF', 'Net_OilFcst_BBL', 'Net_GasFcst_MCF']
            for col in forecast_cols:
                if col not in fcst_df.columns:
                    fcst_df[col] = np.nan
                else:
                    fcst_df[col] = fcst_df[col].round(2)
            final_df = pd.merge(aggregated_data, fcst_df, on='ProducingMonth', how='outer')
            
            # Blend historical and forecast well counts
            final_df['WellCount'] = final_df['WellCount'].fillna(final_df['ForecastWellCount'])
            final_df.drop('ForecastWellCount', axis=1, inplace=True)
        else:
            final_df = aggregated_data.copy()
            final_df[['OilFcst_BBL', 'GasFcst_MCF', 'Net_OilFcst_BBL', 'Net_GasFcst_MCF']] = np.nan
        
        final_df = final_df.sort_values('ProducingMonth')
        
        # Calculate cumulative values
        prod_cols = [('LiquidsProd_BBL', 'OilFcst_BBL'), ('GasProd_MCF', 'GasFcst_MCF')]
        for prod_col, fcst_col in prod_cols:
            for prefix in ['', 'Net_']:
                final_df[f'{prefix}Cum{prod_col}'] = final_df[f'{prefix}{prod_col}'].cumsum().round(2)
                
                last_idx = final_df[f'{prefix}Cum{prod_col}'].last_valid_index()
                if last_idx is not None:
                    last_cum = final_df.loc[last_idx, f'{prefix}Cum{prod_col}']
                    mask = final_df.index > last_idx
                    final_df.loc[mask, f'{prefix}{fcst_col.replace("Fcst", "FcstCum")}'] = (
                        last_cum + final_df.loc[mask, f'{prefix}{fcst_col}'].fillna(0).cumsum()
                    ).round(2)
        
        numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('WellCount')
        final_df[numeric_cols] = final_df[numeric_cols].replace(0, np.nan)
        
        final_df = final_df[final_df['ProducingMonth'] <= fcst_end_date]
        final_df['EOM_Date'] = final_df['ProducingMonth'].dt.to_period('M').dt.to_timestamp('M')
        
        final_df['Oil_Blend'] = final_df['LiquidsProd_BBL'].fillna(final_df['OilFcst_BBL']).round(2)
        final_df['Gas_Blend'] = final_df['GasProd_MCF'].fillna(final_df['GasFcst_MCF']).round(2)
        final_df['Net_Oil_Blend'] = final_df['Net_LiquidsProd_BBL'].fillna(final_df['Net_OilFcst_BBL']).round(2)
        final_df['Net_Gas_Blend'] = final_df['Net_GasProd_MCF'].fillna(final_df['Net_GasFcst_MCF']).round(2)
        
        return final_df

    # Run WellPlotter2
    final_df = _well_plotter2(welllist_df, production_df, fcst_end_date)
    
    # Validate inputs for cash flow calculations
    comm_required_cols = ['Eff_Date', 'Fcst Months', 'Yield (bbl/MMcf)', 'Oil Basis (%)',
                         'Oil Basis ($/bbl)', 'Gas Basis (%)', 'Gas Basis ($/mcf)', 
                         'NGL Basis (%)', 'NGL Basis ($/bbl)', 'Shrink (% Remaining)',
                         'Oil GPT ($/bbl)', 'Gas GPT ($/mcf)', 'NGL GPT ($/mcf)', 
                         'Sev Tax Oil (%)', 'Sev Tax Gas (%)', 'Sev Tax NGL (%)', 'Ad Val Tax']
    
    is_valid_dataframe(Comm_Assump_df, comm_required_cols, "Comm_Assump_df")
    
    if price_df is not None:
        price_required_cols = ['Month'] + [col for col in price_df.columns if '_OIL' in col or '_GAS' in col]
        is_valid_dataframe(price_df, price_required_cols, "price_df")
    
    # Extract parameters
    params = {col: Comm_Assump_df[col].iloc[0] for col in comm_required_cols}
    
    # Create date range for cash flow calculations
    date_range = pd.date_range(start=params['Eff_Date'], 
                             periods=int(params['Fcst Months']), 
                             freq="M")
    
    # Initialize cash flow DataFrame
    cf_df = pd.DataFrame({"Date": date_range})
    cf_df['Date'] = cf_df['Date'].dt.date
    
    # Convert EOM_Date to date format for merging
    final_df['Date'] = pd.to_datetime(final_df['EOM_Date']).dt.date
    
    # Merge with production data
    cf_df = pd.merge(cf_df, 
                    final_df[['Date', 'WellCount', 'Oil_Blend', 'Gas_Blend', 
                             'Net_Oil_Blend', 'Net_Gas_Blend']],
                    on='Date', 
                    how='left')
    
    # Calculate Net NGL
    cf_df['Net_NGL'] = ((cf_df['Net_Gas_Blend']/1000) * params['Yield (bbl/MMcf)']).round(2)

# Add price information based on price deck
    if price_df is not None:
        price_df['Date'] = pd.to_datetime(price_df['Month']).dt.date
        cf_df = pd.merge(cf_df, price_df, on='Date', how='left')
        
        deck_num = str(int(welllist_df['Price Deck'].iloc[0]))
        oil_col = f'{deck_num}_OIL'
        gas_col = f'{deck_num}_GAS'
        
        cf_df['Full Oil Price'] = cf_df[oil_col] if oil_col in cf_df.columns else 0
        cf_df['Full Gas Price'] = cf_df[gas_col] if gas_col in cf_df.columns else 0
    else:
        cf_df['Full Oil Price'] = cf_df['Full Gas Price'] = 0
    
    # Calculate real prices with differentials
    cf_df['Real Oil'] = ((cf_df['Full Oil Price'] * params['Oil Basis (%)']) + params['Oil Basis ($/bbl)']).round(2)
    cf_df['Real Gas'] = ((cf_df['Full Gas Price'] * params['Gas Basis (%)']) + params['Gas Basis ($/mcf)']).round(2)
    cf_df['Real NGL'] = ((cf_df['Full Oil Price'] * params['NGL Basis (%)']) + params['NGL Basis ($/bbl)']).round(2)
    
    # Calculate revenue and taxes
    cf_df['Gross Revenue'] = (
        (cf_df['Net_Oil_Blend'] * cf_df['Real Oil']) +
        (cf_df['Net_Gas_Blend'] * cf_df['Real Gas'] * params['Shrink (% Remaining)']) +
        (cf_df['Net_NGL'] * cf_df['Real NGL'])
    ).round(2)
    
    cf_df['GPT'] = (
        (cf_df['Net_Oil_Blend'] * params['Oil GPT ($/bbl)']) +
        (cf_df['Net_Gas_Blend'] * params['Gas GPT ($/mcf)']) +
        (cf_df['Net_NGL'] * params['NGL GPT ($/mcf)'])
    ).round(2)
    
    cf_df['Sev Tax'] = (
        (cf_df['Net_Oil_Blend'] * cf_df['Real Oil'] * params['Sev Tax Oil (%)']) +
        (cf_df['Net_Gas_Blend'] * cf_df['Real Gas'] * params['Shrink (% Remaining)'] * params['Sev Tax Gas (%)']) +
        (cf_df['Net_NGL'] * cf_df['Real NGL'] * params['Sev Tax NGL (%)'])
    ).round(2)
    
    cf_df['AdVal Tax'] = (cf_df['Gross Revenue'] * params['Ad Val Tax']).round(2)
    
    # Calculate final cash flow
    cf_df['Net Cash Flow'] = (cf_df['Gross Revenue'] - cf_df['GPT'] - cf_df['Sev Tax'] - cf_df['AdVal Tax']).round(2)
    
    # Calculate time periods for discounting (in years)
    cf_df['Years'] = (pd.to_datetime(cf_df['Date']) - pd.to_datetime(params['Eff_Date'])).dt.days / 365.25
    
    # Calculate PVs and ROIs
    for rate in [0, 8, 10, 12, 14, 16, 18, 20]:
        # Calculate PV
        if rate == 0:
            cf_df.loc[0, f'PV{rate}'] = cf_df['Net Cash Flow'].sum().round(2)
        else:
            discount_factor = (1 + rate/100) ** -cf_df['Years']
            cf_df.loc[0, f'PV{rate}'] = (cf_df['Net Cash Flow'] * discount_factor).sum().round(2)
        
        # Calculate MOIC ROIs using this PV as investment base
        investment = abs(cf_df.loc[0, f'PV{rate}'])
        cumulative_cf = cf_df['Net Cash Flow'].cumsum()
        
        for period in [1, 3, 5, 10]:
            year_mask = cf_df['Years'] <= period
            period_cf = cumulative_cf[year_mask].iloc[-1] if any(year_mask) else 0
            cf_df.loc[0, f'ROI_{period}yr_PV{rate}'] = (period_cf / investment).round(2)

    return cf_df

import numpy as np
import pandas as pd

def WellPlotterNetH(welllist_df, production_df):
    """
    Aggregates production data for wells marked with 'Y', generates forecasts, 
    and calculates both gross and net (NRI-adjusted) volumes.
    """
    # Convert and clean DataFrames
    welllist_df = welllist_df.to_pandas() if not isinstance(welllist_df, pd.DataFrame) else welllist_df
    production_df = production_df.to_pandas() if not isinstance(production_df, pd.DataFrame) else production_df
    
    production_df.columns = production_df.columns.str.strip()
    welllist_df.columns = welllist_df.columns.str.strip()
    production_df['ProducingMonth'] = pd.to_datetime(production_df['ProducingMonth'])
    
    wells_with_y = welllist_df[welllist_df['Plot'] == 'Y']
    production_df = production_df[production_df['ProducingMonth'] >= '2018-01-01']
    
    # Merge and calculate net values
    merged_df = production_df.merge(wells_with_y[['API', 'OwnerNRI']], 
                                    left_on='API_UWI', 
                                    right_on='API', 
                                    how='inner')
    
    # Calculate net production values
    for col in ['LiquidsProd_BBL', 'GasProd_MCF', 'WaterProd_BBL', 
                'CumLiquids_BBL', 'CumGas_MCF', 'CumWater_BBL']:
        merged_df[f'Net{col}'] = merged_df[col] * merged_df['OwnerNRI']
    
    # Aggregate data
    agg_columns = {
        'API_UWI': 'nunique',
        'LiquidsProd_BBL': 'sum',
        'GasProd_MCF': 'sum',
        'WaterProd_BBL': 'sum',
        'CumLiquids_BBL': 'sum',
        'CumGas_MCF': 'sum',
        'CumWater_BBL': 'sum',
        'NetLiquidsProd_BBL': 'sum',
        'NetGasProd_MCF': 'sum',
        'NetWaterProd_BBL': 'sum',
        'NetCumLiquids_BBL': 'sum',
        'NetCumGas_MCF': 'sum',
        'NetCumWater_BBL': 'sum'
    }
    
    aggregated_data = merged_df.groupby('ProducingMonth', as_index=False).agg(agg_columns, numeric_only=True).rename(columns={'API_UWI': 'WellCount'})
    
    # Generate forecasts
    all_forecasts = []
    
    for _, well in wells_with_y.iterrows():
        try:
            fcst_start = pd.to_datetime(well['Fcst Start Date']).replace(day=1)
            oil_months = min(int(float(well['Oil Yrs Remain']) * 12), 600)
            gas_months = min(int(float(well['Gas Yrs Remain']) * 12), 600)
            owner_nri = float(well['OwnerNRI'])
            oil_decline_type = well.get('Oil Decline Type', 'E')
            gas_decline_type = well.get('Gas Decline Type', 'E')
            
            max_months = max(oil_months, gas_months)
            dates = pd.date_range(start=fcst_start, periods=max_months, freq='MS')
            
            oil_fcst, gas_fcst = np.zeros(len(dates)), np.zeros(len(dates))
            decline_mode = [''] * len(dates)
            
            def apply_hyperbolic_decline(qi, b, di, dt, terminal_decline_annual):
                production = []
                decline_modes = []
                terminal_decline_monthly = 1 - (1 - terminal_decline_annual) ** (1/12)  # Convert to monthly decline
                for t in range(dt):
                    if t == 0:
                        q = qi
                    else:
                        q_prev = production[-1]
                        q = q_prev / ((1 + b * di * t) ** (1 / b))
                        decline_rate = (q_prev - q) / q_prev
                        
                        if decline_rate < terminal_decline_monthly:
                            q = q_prev * (1 - terminal_decline_monthly)
                            decline_modes.append('E')
                        else:
                            decline_modes.append('H')
                    production.append(q)
                return production, decline_modes
            
            if oil_months > 0:
                oil_qi = float(well['Fcst Qi Oil'])
                oil_b = float(well.get('Oil B-Factor', 1.0))
                oil_d_init = float(well['Oil Decline']) / 12
                oil_d_final = float(well.get('Oil Terminal Decline', 0.1))  # Keep as annual rate for conversion
                oil_fcst[:oil_months], decline_mode[:oil_months] = apply_hyperbolic_decline(oil_qi, oil_b, oil_d_init, oil_months, oil_d_final)
            
            if gas_months > 0:
                gas_qi = float(well['Fcst Qi Gas'])
                gas_b = float(well.get('Gas B-Factor', 1.0))
                gas_d_init = float(well['Gas Decline']) / 12
                gas_d_final = float(well.get('Gas Terminal Decline', 0.1))  # Keep as annual rate for conversion
                gas_fcst[:gas_months], _ = apply_hyperbolic_decline(gas_qi, gas_b, gas_d_init, gas_months, gas_d_final)
            
            well_fcst = pd.DataFrame({'ProducingMonth': dates, 'OilFcst_BBL': oil_fcst, 'GasFcst_MCF': gas_fcst, 'DeclineType': decline_mode})
            all_forecasts.append(well_fcst)
        except Exception as e:
            print(f"Error processing well: {str(e)}")
            continue
    
    if all_forecasts:
        fcst_df = pd.concat(all_forecasts, ignore_index=True).groupby('ProducingMonth', as_index=False).agg({'OilFcst_BBL': 'sum', 'GasFcst_MCF': 'sum', 'DeclineType': 'first'})
        final_df = pd.merge(aggregated_data, fcst_df, on='ProducingMonth', how='outer')
    else:
        final_df = aggregated_data.copy()
    
    final_df['EOM_Date'] = final_df['ProducingMonth'].dt.to_period('M').dt.to_timestamp('M')
    
    return final_df


import pandas as pd
import numpy as np

def WellPlotterNETHYP(welllist_df, production_df):
    """
    Aggregates production data for wells marked with 'Y', generates forecasts using either
    exponential or hyperbolic decline based on decline type specified, and calculates 
    both gross and net (NRI-adjusted) volumes. For hyperbolic decline, switches to
    exponential decline when terminal decline rate is reached. Forecasts continue until
    reaching the specified Qf (final rate) for each well.
    """
    # Convert and clean DataFrames
    welllist_df = welllist_df.to_pandas() if not isinstance(welllist_df, pd.DataFrame) else welllist_df
    production_df = production_df.to_pandas() if not isinstance(production_df, pd.DataFrame) else production_df
    
    production_df.columns = production_df.columns.str.strip()
    welllist_df.columns = welllist_df.columns.str.strip()
    production_df['ProducingMonth'] = pd.to_datetime(production_df['ProducingMonth'])
    
    wells_with_y = welllist_df[welllist_df['Plot'] == 'Y']
    production_df = production_df[production_df['ProducingMonth'] >= '2018-01-01']
    
    # Merge and calculate net values
    merged_df = production_df.merge(wells_with_y[['API', 'OwnerNRI']], 
                                  left_on='API_UWI', 
                                  right_on='API', 
                                  how='inner')
    
    # Calculate net production values
    for col in ['LiquidsProd_BBL', 'GasProd_MCF', 'WaterProd_BBL', 
                'CumLiquids_BBL', 'CumGas_MCF', 'CumWater_BBL']:
        merged_df[f'Net{col}'] = merged_df[col] * merged_df['OwnerNRI']
    
    # Aggregate data
    agg_columns = {
        'API_UWI': 'nunique',
        'LiquidsProd_BBL': 'sum',
        'GasProd_MCF': 'sum',
        'WaterProd_BBL': 'sum',
        'CumLiquids_BBL': 'sum',
        'CumGas_MCF': 'sum',
        'CumWater_BBL': 'sum',
        'NetLiquidsProd_BBL': 'sum',
        'NetGasProd_MCF': 'sum',
        'NetWaterProd_BBL': 'sum',
        'NetCumLiquids_BBL': 'sum',
        'NetCumGas_MCF': 'sum',
        'NetCumWater_BBL': 'sum'
    }
    
    aggregated_data = merged_df.groupby('ProducingMonth').agg(agg_columns).reset_index()
    aggregated_data = aggregated_data.rename(columns={'API_UWI': 'WellCount'})
    
    def calculate_decline_rates(qi, qf, decline_type, b_factor=None, initial_decline=None, terminal_decline=None, max_months=600):
        """
        Calculate production rates using either exponential or hyperbolic decline.
        For hyperbolic decline, switches to exponential at terminal decline rate.
        Returns tuple of (rates array, decline_types array)
        """
        rates = []
        decline_types = []
        current_rate = qi
        
        if decline_type == 'E':  # Pure exponential decline
            monthly_decline = 1 - np.exp(-initial_decline/12)
            
            while current_rate > qf and len(rates) < max_months:
                rates.append(current_rate)
                decline_types.append('E')
                current_rate *= (1 - monthly_decline)
                
        else:  # Hyperbolic decline with terminal transition
            t = 0
            monthly_terminal = 1 - np.exp(-terminal_decline/12)
            
            while current_rate > qf and len(rates) < max_months:
                # Calculate current annual decline rate
                current_decline = initial_decline / (1 + b_factor * initial_decline * t/12)
                
                # Check for transition to terminal decline
                if current_decline <= terminal_decline:
                    # Save the rate at transition point
                    transition_rate = current_rate
                    
                    # Switch to exponential decline using terminal decline rate
                    while current_rate > qf and len(rates) < max_months:
                        rates.append(current_rate)
                        decline_types.append('E')
                        current_rate *= (1 - monthly_terminal)
                    break
                
                # Still in hyperbolic decline
                rates.append(current_rate)
                decline_types.append('H')
                # Calculate next rate using hyperbolic formula
                current_rate = qi / np.power(1 + b_factor * initial_decline * (t + 1)/12, 1/b_factor)
                t += 1
        
        return np.array(rates), np.array(decline_types)
    
    # Generate forecasts
    all_forecasts = []
    
    for _, well in wells_with_y.iterrows():
        try:
            fcst_start = pd.to_datetime(well['Fcst Start Date']).replace(day=1)
            owner_nri = float(well['OwnerNRI'])
            well_hist = production_df[production_df['API_UWI'] == well['API']]
            
            # Initialize forecast period to maximum months
            max_months = 600
            dates = pd.date_range(start=fcst_start, periods=max_months, freq='MS')
            
            # Generate gas forecast
            try:
                gas_qi = float(well['Fcst Qi Gas'])
                gas_qf = float(well['Qf Gas'])
                
                if gas_qi > gas_qf:
                    last_gas_prod = well_hist[well_hist['ProducingMonth'] <= fcst_start]['GasProd_MCF'].tail(3)
                    if not last_gas_prod.empty:
                        gas_qi = min(gas_qi, last_gas_prod.mean() * 1.1)
                    
                    gas_decline_type = well['Gas Decline Type']
                    gas_rates, gas_decline_types = calculate_decline_rates(
                        qi=gas_qi,
                        qf=gas_qf,
                        decline_type=gas_decline_type,
                        b_factor=float(well['Gas B-Factor']),
                        initial_decline=float(well['Gas Decline']),
                        terminal_decline=float(well['Gas Terminal Decline'])
                    )
                    
                    if len(gas_rates) > 0:
                        net_gas_fcst = gas_rates * owner_nri
                        data_length = len(gas_rates)
                        
                        # Create DataFrame with matching lengths
                        well_fcst = pd.DataFrame({
                            'ProducingMonth': dates[:data_length],
                            'GasFcst_MCF': gas_rates,
                            'NetGasFcst_MCF': net_gas_fcst,
                            'Gas_Decline_Type': gas_decline_types
                        })
                        all_forecasts.append(well_fcst)
            except Exception as e:
                print(f"Error processing gas forecast for well {well['API']}: {str(e)}")
            
            # Generate oil forecast
            try:
                oil_qi = float(well['Fcst Qi Oil'])
                oil_qf = float(well['Qf Oil'])
                
                if oil_qi > oil_qf:
                    last_oil_prod = well_hist[well_hist['ProducingMonth'] <= fcst_start]['LiquidsProd_BBL'].tail(3)
                    if not last_oil_prod.empty:
                        oil_qi = min(oil_qi, last_oil_prod.mean() * 1.1)
                    
                    oil_decline_type = well['Oil Decline Type']
                    oil_rates, oil_decline_types = calculate_decline_rates(
                        qi=oil_qi,
                        qf=oil_qf,
                        decline_type=oil_decline_type,
                        b_factor=float(well['Oil B-Factor']),
                        initial_decline=float(well['Oil Decline']),
                        terminal_decline=float(well['Oil Terminal Decline'])
                    )
                    
                    if len(oil_rates) > 0:
                        net_oil_fcst = oil_rates * owner_nri
                        data_length = len(oil_rates)
                        
                        # Create DataFrame with matching lengths
                        well_fcst = pd.DataFrame({
                            'ProducingMonth': dates[:data_length],
                            'OilFcst_BBL': oil_rates,
                            'NetOilFcst_BBL': net_oil_fcst,
                            'Oil_Decline_Type': oil_decline_types
                        })
                        all_forecasts.append(well_fcst)
            except Exception as e:
                print(f"Error processing oil forecast for well {well['API']}: {str(e)}")
                
        except Exception as e:
            print(f"Error processing well {well['API']}: {str(e)}")
            continue
    
    # Combine forecasts with historical data
    if all_forecasts:
        fcst_df = pd.concat(all_forecasts, ignore_index=True)
        
        # Define numeric columns to sum
        numeric_cols = ['GasFcst_MCF', 'NetGasFcst_MCF', 'OilFcst_BBL', 'NetOilFcst_BBL']
        
        # Group by month and sum only numeric columns
        fcst_df = fcst_df.groupby('ProducingMonth').agg({
            col: 'sum' for col in numeric_cols if col in fcst_df.columns
        }).reset_index()
        
        # Preserve decline type columns
        for col in ['Oil_Decline_Type', 'Gas_Decline_Type']:
            if all_forecasts and col in all_forecasts[0].columns:
                decline_types = pd.concat(all_forecasts, ignore_index=True).groupby('ProducingMonth')[col].first()
                fcst_df[col] = fcst_df['ProducingMonth'].map(decline_types)
        
        # Ensure all forecast columns exist
        for col in numeric_cols:
            if col not in fcst_df.columns:
                fcst_df[col] = np.nan
                
        final_df = pd.merge(aggregated_data, fcst_df, on='ProducingMonth', how='outer')
    else:
        final_df = aggregated_data.copy()
        for col in ['OilFcst_BBL', 'GasFcst_MCF', 'NetOilFcst_BBL', 'NetGasFcst_MCF']:
            final_df[col] = np.nan
    
    final_df = final_df.sort_values('ProducingMonth')
    
    # Calculate forecast cumulatives
    for prefix in ['', 'Net']:
        gas_idx = final_df[f'{prefix}CumGas_MCF'].last_valid_index()
        oil_idx = final_df[f'{prefix}CumLiquids_BBL'].last_valid_index()
        
        if gas_idx is not None:
            last_cum_gas = final_df.loc[gas_idx, f'{prefix}CumGas_MCF']
            mask = final_df.index > gas_idx
            final_df.loc[mask, f'{prefix}GasFcstCum_MCF'] = (
                last_cum_gas + final_df.loc[mask, f'{prefix}GasFcst_MCF'].fillna(0).cumsum()
            )
        
        if oil_idx is not None:
            last_cum_oil = final_df.loc[oil_idx, f'{prefix}CumLiquids_BBL']
            mask = final_df.index > oil_idx
            final_df.loc[mask, f'{prefix}OilFcstCum_BBL'] = (
                last_cum_oil + final_df.loc[mask, f'{prefix}OilFcst_BBL'].fillna(0).cumsum()
            )
    
    # Clean up and format
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('WellCount')
    final_df[numeric_cols] = final_df[numeric_cols].replace(0, np.nan)
    
    final_df['EOM_Date'] = final_df['ProducingMonth'].dt.to_period('M').dt.to_timestamp('M')
    
    # Create blended columns
    final_df['Oil_Blend'] = final_df['LiquidsProd_BBL'].fillna(final_df['OilFcst_BBL'])
    final_df['Gas_Blend'] = final_df['GasProd_MCF'].fillna(final_df['GasFcst_MCF'])
    final_df['Net_Oil_Blend'] = final_df['NetLiquidsProd_BBL'].fillna(final_df['NetOilFcst_BBL'])
    final_df['Net_Gas_Blend'] = final_df['NetGasProd_MCF'].fillna(final_df['NetGasFcst_MCF'])
    
    # Set column order
    col_order = ['ProducingMonth', 'EOM_Date', 'WellCount',
                 'LiquidsProd_BBL', 'GasProd_MCF', 'WaterProd_BBL',
                 'CumLiquids_BBL', 'CumGas_MCF', 'CumWater_BBL',
                 'NetLiquidsProd_BBL', 'NetGasProd_MCF', 'NetWaterProd_BBL',
                 'NetCumLiquids_BBL', 'NetCumGas_MCF', 'NetCumWater_BBL',
                 'OilFcst_BBL', 'GasFcst_MCF', 'OilFcstCum_BBL', 'GasFcstCum_MCF',
                 'NetOilFcst_BBL', 'NetGasFcst_MCF', 'NetOilFcstCum_BBL', 'NetGasFcstCum_MCF',
                 'Oil_Blend', 'Gas_Blend', 'Net_Oil_Blend', 'Net_Gas_Blend',
                 'Oil_Decline_Type', 'Gas_Decline_Type']
    
    existing_cols = [col for col in col_order if col in final_df.columns]
    additional_cols = [col for col in final_df.columns if col not in col_order]
    final_df = final_df[existing_cols + additional_cols]
    
    return final_df

import pandas as pd
import numpy as np

def OwnerPlotNetHYP(welllist_df, production_df):
    """
    Aggregates production data for wells marked with 'Y', generates forecasts using either
    exponential or hyperbolic decline based on decline type specified, and calculates 
    both gross and net (NRI-adjusted) volumes. For hyperbolic decline, switches to
    exponential decline when terminal decline rate is reached.
    """
    # Convert and clean DataFrames
    welllist_df = welllist_df.to_pandas() if not isinstance(welllist_df, pd.DataFrame) else welllist_df
    production_df = production_df.to_pandas() if not isinstance(production_df, pd.DataFrame) else production_df
    
    # Basic data preparation
    production_df.columns = production_df.columns.str.strip()
    welllist_df.columns = welllist_df.columns.str.strip()
    production_df['ProducingMonth'] = pd.to_datetime(production_df['ProducingMonth'])
    
    # Filter wells marked with 'Y' and merge with production data
    wells_with_y = welllist_df[welllist_df['Owner Plot'] == 'Y']
    production_df = production_df[production_df['ProducingMonth'] >= '2018-01-01']
    
    # Merge production data with well list to get OwnerNRI
    merged_df = production_df.merge(wells_with_y[['API', 'OwnerNRI']], 
                                  left_on='API_UWI', 
                                  right_on='API', 
                                  how='inner')
    
    # Create net production columns
    for col in ['LiquidsProd_BBL', 'GasProd_MCF', 'WaterProd_BBL']:
        merged_df[f'Net{col}'] = merged_df[col] * merged_df['OwnerNRI']
    
    # Aggregate both gross and net production data
    agg_columns = {
        'API_UWI': 'nunique',
        'LiquidsProd_BBL': 'sum',
        'GasProd_MCF': 'sum',
        'WaterProd_BBL': 'sum',
        'NetLiquidsProd_BBL': 'sum',
        'NetGasProd_MCF': 'sum',
        'NetWaterProd_BBL': 'sum'
    }
    
    aggregated_data = merged_df.groupby('ProducingMonth', as_index=False).agg(agg_columns)
    aggregated_data = aggregated_data.rename(columns={'API_UWI': 'WellCount'})
    
    # Calculate cumulative volumes for both gross and net
    aggregated_data = aggregated_data.sort_values('ProducingMonth')
    # Gross cumulatives
    aggregated_data['CumLiquids_BBL'] = aggregated_data['LiquidsProd_BBL'].cumsum()
    aggregated_data['CumGas_MCF'] = aggregated_data['GasProd_MCF'].cumsum()
    aggregated_data['CumWater_BBL'] = aggregated_data['WaterProd_BBL'].cumsum()
    # Net cumulatives
    aggregated_data['NetCumLiquids_BBL'] = aggregated_data['NetLiquidsProd_BBL'].cumsum()
    aggregated_data['NetCumGas_MCF'] = aggregated_data['NetGasProd_MCF'].cumsum()
    aggregated_data['NetCumWater_BBL'] = aggregated_data['NetWaterProd_BBL'].cumsum()
    
    def calculate_decline_rates(qi, qf, decline_type, b_factor=None, initial_decline=None, terminal_decline=None, max_months=600):
        """
        Calculate production rates using either exponential or hyperbolic decline.
        For hyperbolic decline, switches to exponential at terminal decline rate.
        Returns tuple of (rates array, decline_types array)
        """
        rates = []
        decline_types = []
        current_rate = qi
        
        if decline_type == 'E':  # Pure exponential decline
            monthly_decline = 1 - np.exp(-initial_decline/12)
            
            while current_rate > qf and len(rates) < max_months:
                rates.append(current_rate)
                decline_types.append('E')
                current_rate *= (1 - monthly_decline)
                
        else:  # Hyperbolic decline with terminal transition
            t = 0
            monthly_terminal = 1 - np.exp(-terminal_decline/12)
            
            while current_rate > qf and len(rates) < max_months:
                # Calculate current annual decline rate
                current_decline = initial_decline / (1 + b_factor * initial_decline * t/12)
                
                # Check for transition to terminal decline
                if current_decline <= terminal_decline:
                    # Switch to exponential decline using terminal decline rate
                    while current_rate > qf and len(rates) < max_months:
                        rates.append(current_rate)
                        decline_types.append('E')
                        current_rate *= (1 - monthly_terminal)
                    break
                
                # Still in hyperbolic decline
                rates.append(current_rate)
                decline_types.append('H')
                # Calculate next rate using hyperbolic formula
                current_rate = qi / np.power(1 + b_factor * initial_decline * (t + 1)/12, 1/b_factor)
                t += 1
        
        return np.array(rates), np.array(decline_types)
    
    # Generate forecasts
    all_forecasts = []
    
    for _, well in wells_with_y.iterrows():
        try:
            fcst_start = pd.to_datetime(well['Fcst Start Date']).replace(day=1)
            owner_nri = float(well['OwnerNRI'])
            well_hist = production_df[production_df['API_UWI'] == well['API']]
            
            # Initialize forecast period
            max_months = 600
            dates = pd.date_range(start=fcst_start, periods=max_months, freq='MS')
            
            # Initialize an empty DataFrame for this well's forecasts
            well_fcst = pd.DataFrame({'ProducingMonth': dates})
            has_forecast = False
            
            # Generate gas forecast
            try:
                gas_qi = float(well['Fcst Qi Gas'])
                gas_qf = float(well['Qf Gas'])
                
                if gas_qi > gas_qf:
                    last_gas_prod = well_hist[well_hist['ProducingMonth'] <= fcst_start]['GasProd_MCF'].tail(3)
                    if not last_gas_prod.empty:
                        gas_qi = min(gas_qi, last_gas_prod.mean() * 1.1)
                    
                    gas_decline_type = well['Gas Decline Type']
                    gas_rates, gas_decline_types = calculate_decline_rates(
                        qi=gas_qi,
                        qf=gas_qf,
                        decline_type=gas_decline_type,
                        b_factor=float(well['Gas B-Factor']),
                        initial_decline=float(well['Gas Decline']),
                        terminal_decline=float(well['Gas Terminal Decline'])
                    )
                    
                    if len(gas_rates) > 0:
                        net_gas_fcst = gas_rates * owner_nri
                        data_length = len(gas_rates)
                        
                        well_fcst.loc[:data_length-1, 'GasFcst_MCF'] = gas_rates
                        well_fcst.loc[:data_length-1, 'NetGasFcst_MCF'] = net_gas_fcst
                        well_fcst.loc[:data_length-1, 'Gas_Decline_Type'] = gas_decline_types
                        has_forecast = True
                        
            except Exception as e:
                print(f"Error processing gas forecast for well {well['API']}: {str(e)}")
            
            # Generate oil forecast
            try:
                oil_qi = float(well['Fcst Qi Oil'])
                oil_qf = float(well['Qf Oil'])
                
                if oil_qi > oil_qf:
                    last_oil_prod = well_hist[well_hist['ProducingMonth'] <= fcst_start]['LiquidsProd_BBL'].tail(3)
                    if not last_oil_prod.empty:
                        oil_qi = min(oil_qi, last_oil_prod.mean() * 1.1)
                    
                    oil_decline_type = well['Oil Decline Type']
                    oil_rates, oil_decline_types = calculate_decline_rates(
                        qi=oil_qi,
                        qf=oil_qf,
                        decline_type=oil_decline_type,
                        b_factor=float(well['Oil B-Factor']),
                        initial_decline=float(well['Oil Decline']),
                        terminal_decline=float(well['Oil Terminal Decline'])
                    )
                    
                    if len(oil_rates) > 0:
                        net_oil_fcst = oil_rates * owner_nri
                        data_length = len(oil_rates)
                        
                        well_fcst.loc[:data_length-1, 'OilFcst_BBL'] = oil_rates
                        well_fcst.loc[:data_length-1, 'NetOilFcst_BBL'] = net_oil_fcst
                        well_fcst.loc[:data_length-1, 'Oil_Decline_Type'] = oil_decline_types
                        has_forecast = True
                        
            except Exception as e:
                print(f"Error processing oil forecast for well {well['API']}: {str(e)}")
            
            # Only append if we have either oil or gas forecast
            if has_forecast:
                # Initialize all possible forecast columns to ensure they exist
                forecast_cols = ['GasFcst_MCF', 'NetGasFcst_MCF', 'OilFcst_BBL', 'NetOilFcst_BBL']
                for col in forecast_cols:
                    if col not in well_fcst.columns:
                        well_fcst[col] = np.nan
                
                # Remove rows where all forecast columns are NaN
                well_fcst = well_fcst.dropna(subset=forecast_cols, how='all')
                
                # Ensure decline type columns exist
                for col in ['Oil_Decline_Type', 'Gas_Decline_Type']:
                    if col not in well_fcst.columns:
                        well_fcst[col] = np.nan
                
                if not well_fcst.empty:
                    # Only keep necessary columns
                    keep_cols = ['ProducingMonth'] + forecast_cols + ['Oil_Decline_Type', 'Gas_Decline_Type']
                    well_fcst = well_fcst[keep_cols]
                    all_forecasts.append(well_fcst)
                
        except Exception as e:
            print(f"Error processing well {well['API']}: {str(e)}")
            continue
    
    # Combine forecasts with historical data
    if all_forecasts:
        fcst_df = pd.concat(all_forecasts, ignore_index=True)
        
        # Define numeric columns to sum
        numeric_cols = ['GasFcst_MCF', 'NetGasFcst_MCF', 'OilFcst_BBL', 'NetOilFcst_BBL']
        
        # Group by month and sum numeric columns, explicitly specifying numeric_only
        agg_dict = {col: 'sum' for col in numeric_cols if col in fcst_df.columns}
        
        # Add first() aggregation for decline type columns
        for col in ['Oil_Decline_Type', 'Gas_Decline_Type']:
            if col in fcst_df.columns:
                agg_dict[col] = 'first'
        
        fcst_df = fcst_df.groupby('ProducingMonth', as_index=False).agg(agg_dict)
        
        # Ensure all forecast columns exist
        for col in numeric_cols:
            if col not in fcst_df.columns:
                fcst_df[col] = np.nan
                
        final_df = pd.merge(aggregated_data, fcst_df, on='ProducingMonth', how='outer')
    else:
        final_df = aggregated_data.copy()
        for col in ['OilFcst_BBL', 'GasFcst_MCF', 'NetOilFcst_BBL', 'NetGasFcst_MCF']:
            final_df[col] = np.nan
    
    # Sort by date for cumulative calculations
    final_df = final_df.sort_values('ProducingMonth')
    
    # Calculate forecast cumulatives
    for prefix in ['', 'Net']:
        gas_idx = final_df[f'{prefix}CumGas_MCF'].last_valid_index()
        oil_idx = final_df[f'{prefix}CumLiquids_BBL'].last_valid_index()
        
        if gas_idx is not None:
            last_cum_gas = final_df.loc[gas_idx, f'{prefix}CumGas_MCF']
            mask = final_df.index > gas_idx
            final_df.loc[mask, f'{prefix}GasFcstCum_MCF'] = (
                last_cum_gas + final_df.loc[mask, f'{prefix}GasFcst_MCF'].fillna(0).cumsum()
            )
        
        if oil_idx is not None:
            last_cum_oil = final_df.loc[oil_idx, f'{prefix}CumLiquids_BBL']
            mask = final_df.index > oil_idx
            final_df.loc[mask, f'{prefix}OilFcstCum_BBL'] = (
                last_cum_oil + final_df.loc[mask, f'{prefix}OilFcst_BBL'].fillna(0).cumsum()
            )
    
    # Replace zeros with NaN for numeric columns except WellCount
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('WellCount')
    final_df[numeric_cols] = final_df[numeric_cols].replace(0, np.nan)
    
    # Filter to end of 2030
    final_df = final_df[final_df['ProducingMonth'] <= '2050-12-31']
    
    # Add End of Month date column
    final_df['EOM_Date'] = final_df['ProducingMonth'].dt.to_period('M').dt.to_timestamp('M')
    
    # Create blended columns
    final_df['Oil_Blend'] = final_df['LiquidsProd_BBL'].fillna(final_df['OilFcst_BBL'])
    final_df['Gas_Blend'] = final_df['GasProd_MCF'].fillna(final_df['GasFcst_MCF'])
    final_df['Net_Oil_Blend'] = final_df['NetLiquidsProd_BBL'].fillna(final_df['NetOilFcst_BBL'])
    final_df['Net_Gas_Blend'] = final_df['NetGasProd_MCF'].fillna(final_df['NetGasFcst_MCF'])
    
    # Set column order
    col_order = ['ProducingMonth', 'EOM_Date', 'WellCount',
                 'LiquidsProd_BBL', 'GasProd_MCF', 'WaterProd_BBL',
                 'CumLiquids_BBL', 'CumGas_MCF', 'CumWater_BBL',
                 'NetLiquidsProd_BBL', 'NetGasProd_MCF', 'NetWaterProd_BBL',
                 'NetCumLiquids_BBL', 'NetCumGas_MCF', 'NetCumWater_BBL',
                 'OilFcst_BBL', 'GasFcst_MCF', 'OilFcstCum_BBL', 'GasFcstCum_MCF',
                 'NetOilFcst_BBL', 'NetGasFcst_MCF', 'NetOilFcstCum_BBL', 'NetGasFcstCum_MCF',
                 'Oil_Blend', 'Gas_Blend', 'Net_Oil_Blend', 'Net_Gas_Blend',
                 'Oil_Decline_Type', 'Gas_Decline_Type']
    
    # Ensure all columns exist and add any additional columns at the end
    existing_cols = [col for col in col_order if col in final_df.columns]
    additional_cols = [col for col in final_df.columns if col not in col_order]
    final_df = final_df[existing_cols + additional_cols]
    
    return final_df