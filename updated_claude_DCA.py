import pandas as pd
import numpy as np

def SingleWellForecast(well_api, econ_input_df, raw_prod_data_df):
    """
    Generates a production forecast for a single well using either exponential or 
    hyperbolic decline based on decline type specified. Removed NRI adjustments
    and focuses on gross volumes only.
    
    Parameters:
    -----------
    well_api : str
        The API/UWI identifier for the specific well to forecast
    econ_input_df : pandas.DataFrame
        DataFrame containing well parameters including decline curve inputs (ECON_INPUT)
    raw_prod_data_df : pandas.DataFrame
        DataFrame containing historical production data (RAW_PROD_DATA)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with historical and forecast data for the specified well
    """
    # Convert and clean DataFrames
    econ_input_df = econ_input_df.to_pandas() if not isinstance(econ_input_df, pd.DataFrame) else econ_input_df
    raw_prod_data_df = raw_prod_data_df.to_pandas() if not isinstance(raw_prod_data_df, pd.DataFrame) else raw_prod_data_df
    
    # Basic data preparation
    raw_prod_data_df.columns = raw_prod_data_df.columns.str.strip()
    econ_input_df.columns = econ_input_df.columns.str.strip()
    raw_prod_data_df['PRODUCINGMONTH'] = pd.to_datetime(raw_prod_data_df['PRODUCINGMONTH'])
    
    # Filter to get only the selected well
    well_data = econ_input_df[econ_input_df['API_UWI'] == well_api]
    
    if well_data.empty:
        raise ValueError(f"Well API {well_api} not found in econ input")
    
    # Get the well's production history
    well_production = raw_prod_data_df[raw_prod_data_df['API_UWI'] == well_api]
    well_production = well_production[well_production['PRODUCINGMONTH'] >= '2018-01-01']
    
    if well_production.empty:
        raise ValueError(f"No production data found for well API {well_api}")
    
    # Aggregate production data by month (in case there are multiple entries)
    agg_columns = {
        'LIQUIDSPROD_BBL': 'sum',
        'GASPROD_MCF': 'sum',
        'WATERPROD_BBL': 'sum'
    }
    
    aggregated_data = well_production.groupby('PRODUCINGMONTH', as_index=False).agg(agg_columns)
    
    # Calculate cumulative volumes
    aggregated_data = aggregated_data.sort_values('PRODUCINGMONTH')
    aggregated_data['CumLiquids_BBL'] = aggregated_data['LIQUIDSPROD_BBL'].cumsum()
    aggregated_data['CumGas_MCF'] = aggregated_data['GASPROD_MCF'].cumsum()
    aggregated_data['CumWater_BBL'] = aggregated_data['WATERPROD_BBL'].cumsum()
    
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
    well_row = well_data.iloc[0]
    
    try:
        # Get the last production date and add 1 month for forecast start
        last_prod_date = well_production['PRODUCINGMONTH'].max()
        fcst_start = (pd.to_datetime(last_prod_date) + pd.DateOffset(months=1)).replace(day=1)
        
        # Initialize forecast period
        max_months = 600
        dates = pd.date_range(start=fcst_start, periods=max_months, freq='MS')
        
        # Initialize an empty DataFrame for this well's forecasts
        well_fcst = pd.DataFrame({'PRODUCINGMONTH': dates})
        has_forecast = False
        
        # Generate gas forecast
        try:
            # Determine gas initial rate (qi) - use user input if available, otherwise calculated
            gas_qi = float(well_row['GAS_USER_QI']) if pd.notna(well_row['GAS_USER_QI']) else float(well_row['GAS_CALC_QI'])
            gas_qf = float(well_row['GAS_Q_MIN'])
            
            if gas_qi > gas_qf:
                last_gas_prod = well_production[well_production['PRODUCINGMONTH'] <= fcst_start]['GASPROD_MCF'].tail(3)
                if not last_gas_prod.empty:
                    gas_qi = min(gas_qi, last_gas_prod.mean() * 1.1)
                
                gas_decline_type = well_row['GAS_DECLINE_TYPE']
                
                # Determine gas decline rate - use user input if available, otherwise empirical
                gas_decline = float(well_row['GAS_USER_DECLINE']) if pd.notna(well_row['GAS_USER_DECLINE']) else float(well_row['GAS_EMPIRICAL_DECLINE'])
                
                # Determine b-factor - use user input if available, otherwise calculated
                gas_b_factor = float(well_row['GAS_USER_B_FACTOR']) if pd.notna(well_row['GAS_USER_B_FACTOR']) else float(well_row['GAS_CALC_B_FACTOR'])
                
                gas_rates, gas_decline_types = calculate_decline_rates(
                    qi=gas_qi,
                    qf=gas_qf,
                    decline_type=gas_decline_type,
                    b_factor=gas_b_factor,
                    initial_decline=gas_decline,
                    terminal_decline=float(well_row['GAS_D_MIN'])
                )
                
                if len(gas_rates) > 0:
                    data_length = len(gas_rates)
                    well_fcst.loc[:data_length-1, 'GasFcst_MCF'] = gas_rates
                    well_fcst.loc[:data_length-1, 'Gas_Decline_Type'] = gas_decline_types
                    has_forecast = True
                    
        except Exception as e:
            print(f"Error processing gas forecast for well {well_api}: {str(e)}")
        
        # Generate oil forecast
        try:
            # Determine oil initial rate (qi) - use user input if available, otherwise calculated
            oil_qi = float(well_row['OIL_USER_QI']) if pd.notna(well_row['OIL_USER_QI']) else float(well_row['OIL_CALC_QI'])
            oil_qf = float(well_row['OIL_Q_MIN'])
            
            if oil_qi > oil_qf:
                last_oil_prod = well_production[well_production['PRODUCINGMONTH'] <= fcst_start]['LIQUIDSPROD_BBL'].tail(3)
                if not last_oil_prod.empty:
                    oil_qi = min(oil_qi, last_oil_prod.mean() * 1.1)
                
                oil_decline_type = well_row['OIL_DECLINE_TYPE']
                
                # Determine oil decline rate - use user input if available, otherwise empirical
                oil_decline = float(well_row['OIL_USER_DECLINE']) if pd.notna(well_row['OIL_USER_DECLINE']) else float(well_row['OIL_EMPIRICAL_DECLINE'])
                
                # Determine b-factor - use user input if available, otherwise calculated
                oil_b_factor = float(well_row['OIL_USER_B_FACTOR']) if pd.notna(well_row['OIL_USER_B_FACTOR']) else float(well_row['OIL_CALC_B_FACTOR'])
                
                oil_rates, oil_decline_types = calculate_decline_rates(
                    qi=oil_qi,
                    qf=oil_qf,
                    decline_type=oil_decline_type,
                    b_factor=oil_b_factor,
                    initial_decline=oil_decline,
                    terminal_decline=float(well_row['OIL_D_MIN'])
                )
                
                if len(oil_rates) > 0:
                    data_length = len(oil_rates)
                    well_fcst.loc[:data_length-1, 'OilFcst_BBL'] = oil_rates
                    well_fcst.loc[:data_length-1, 'Oil_Decline_Type'] = oil_decline_types
                    has_forecast = True
                    
        except Exception as e:
            print(f"Error processing oil forecast for well {well_api}: {str(e)}")
        
        # Only continue if we have either oil or gas forecast
        if has_forecast:
            # Initialize all possible forecast columns to ensure they exist
            forecast_cols = ['GasFcst_MCF', 'OilFcst_BBL']
            for col in forecast_cols:
                if col not in well_fcst.columns:
                    well_fcst[col] = np.nan
            
            # Remove rows where all forecast columns are NaN
            well_fcst = well_fcst.dropna(subset=forecast_cols, how='all')
            
            # Ensure decline type columns exist
            for col in ['Oil_Decline_Type', 'Gas_Decline_Type']:
                if col not in well_fcst.columns:
                    well_fcst[col] = np.nan
            
    except Exception as e:
        print(f"Error processing well {well_api}: {str(e)}")
        well_fcst = pd.DataFrame()
    
    # Combine forecasts with historical data
    if not well_fcst.empty:
        final_df = pd.merge(aggregated_data, well_fcst, on='PRODUCINGMONTH', how='outer')
    else:
        final_df = aggregated_data.copy()
        for col in ['OilFcst_BBL', 'GasFcst_MCF']:
            final_df[col] = np.nan
    
    # Sort by date for cumulative calculations
    final_df = final_df.sort_values('PRODUCINGMONTH')
    
    # Calculate forecast cumulatives
    gas_idx = final_df['CumGas_MCF'].last_valid_index()
    oil_idx = final_df['CumLiquids_BBL'].last_valid_index()
    
    if gas_idx is not None:
        last_cum_gas = final_df.loc[gas_idx, 'CumGas_MCF']
        mask = final_df.index > gas_idx
        final_df.loc[mask, 'GasFcstCum_MCF'] = (
            last_cum_gas + final_df.loc[mask, 'GasFcst_MCF'].fillna(0).cumsum()
        )
    
    if oil_idx is not None:
        last_cum_oil = final_df.loc[oil_idx, 'CumLiquids_BBL']
        mask = final_df.index > oil_idx
        final_df.loc[mask, 'OilFcstCum_BBL'] = (
            last_cum_oil + final_df.loc[mask, 'OilFcst_BBL'].fillna(0).cumsum()
        )
    
    # Replace zeros with NaN for numeric columns
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
    final_df[numeric_cols] = final_df[numeric_cols].replace(0, np.nan)
    
    # Filter to end of 2050
    final_df = final_df[final_df['PRODUCINGMONTH'] <= '2050-12-31']
    
    # Add End of Month date column
    final_df['EOM_Date'] = final_df['PRODUCINGMONTH'].dt.to_period('M').dt.to_timestamp('M')
    
    # Create blended columns
    final_df['Oil_Blend'] = final_df['LIQUIDSPROD_BBL'].fillna(final_df['OilFcst_BBL'])
    final_df['Gas_Blend'] = final_df['GASPROD_MCF'].fillna(final_df['GasFcst_MCF'])
    
    # Set column order
    col_order = ['PRODUCINGMONTH', 'EOM_Date',
                 'LIQUIDSPROD_BBL', 'GASPROD_MCF', 'WATERPROD_BBL',
                 'CumLiquids_BBL', 'CumGas_MCF', 'CumWater_BBL',
                 'OilFcst_BBL', 'GasFcst_MCF', 'OilFcstCum_BBL', 'GasFcstCum_MCF',
                 'Oil_Blend', 'Gas_Blend',
                 'Oil_Decline_Type', 'Gas_Decline_Type']
    
    # Ensure all columns exist and add any additional columns at the end
    existing_cols = [col for col in col_order if col in final_df.columns]
    additional_cols = [col for col in final_df.columns if col not in col_order]
    final_df = final_df[existing_cols + additional_cols]
    
    return final_df