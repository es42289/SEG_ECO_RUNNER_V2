##########################################
# SECTION 1: IMPORTS & SETUP
##########################################
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO
import base64
import folium
from streamlit_folium import folium_static
import branca.colormap as cm

# Set Streamlit Page Configuration with wider layout
st.set_page_config(
    page_title="Table Updater",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed for more space
)

# Make the app wider with custom CSS
st.markdown("""
<style>
    .block-container {
        max-width: 95% !important;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Import SingleWellForecast function
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
    
    # Handle missing WATERPROD_BBL column
    if 'WATERPROD_BBL' not in raw_prod_data_df.columns:
        raw_prod_data_df['WATERPROD_BBL'] = 0
    
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
        'GASPROD_MCF': 'sum'
    }
    
    # Add water only if it exists
    if 'WATERPROD_BBL' in well_production.columns:
        agg_columns['WATERPROD_BBL'] = 'sum'
    
    aggregated_data = well_production.groupby('PRODUCINGMONTH', as_index=False).agg(agg_columns)
    
    # Handle missing WATERPROD_BBL column in aggregated data
    if 'WATERPROD_BBL' not in aggregated_data.columns:
        aggregated_data['WATERPROD_BBL'] = 0
    
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

##########################################
# SECTION 2: SIDEBAR & DATABASE CONNECTION
##########################################
# Sidebar Inputs
st.sidebar.title("Table Updater")
table_name = st.sidebar.text_input("Enter table name", "ECON_INPUT")
primary_key = st.sidebar.text_input("Primary Key Column", "API_UWI")

# Use the active session directly (Snowflake Native App approach)
@st.cache_resource
def get_session():
    """Get the current Snowpark session"""
    try:
        from snowflake.snowpark.context import get_active_session
        return get_active_session()
    except Exception as e:
        st.error(f"Error getting Snowflake session: {e}")
        return None

# Get session
session = get_session()

##########################################
# SECTION 3: UTILITY FUNCTIONS - DATABASE
##########################################
# Get Table Columns
@st.cache_data(ttl=600)
def get_table_columns(table):
    try:
        from snowflake.snowpark.context import get_active_session
        session = get_active_session()
        
        query = f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table}' ORDER BY ORDINAL_POSITION"
        df = session.sql(query).to_pandas()
        if not df.empty:
            return df.set_index("COLUMN_NAME")["DATA_TYPE"].to_dict()
        return {}
    except Exception as e:
        st.error(f"Error getting table columns: {e}")
        return {}

# Fetch Table Data
@st.cache_data(ttl=60)
def get_table_data(table, where_clause=""):
    try:
        from snowflake.snowpark.context import get_active_session
        session = get_active_session()
        
        query = f"SELECT * FROM {table} {f'WHERE {where_clause}' if where_clause else ''} LIMIT 1000"
        return session.sql(query).to_pandas()
    except Exception as e:
        st.error(f"Error fetching table data: {e}")
        return pd.DataFrame()

# Fetch Well Data for Filtering
@st.cache_data(ttl=600)
def get_well_data():
    """Get well data using direct query execution"""
    try:
        from snowflake.snowpark.context import get_active_session
        session = get_active_session()
        
        query = """
        SELECT API_UWI, WELLNAME, STATEPROVINCE, COUNTRY, COUNTY, FIRSTPRODDATE, LATITUDE, LONGITUDE,
               ENVOPERATOR, LEASE, ENVWELLSTATUS, ENVINTERVAL, TRAJECTORY, CUMGAS_MCF, CUMOIL_BBL, TOTALPRODUCINGMONTHS
        FROM wells.minerals.vw_well_input
        """
        
        # Execute the query directly using the session
        result = session.sql(query).to_pandas()
        
        if result.empty:
            st.warning("No well data retrieved from database.")
            return None
            
        # Fill NaNs in the CUMOIL_BBL and CUMGAS_MCF columns with 0
        result["CUMOIL_BBL"] = result["CUMOIL_BBL"].fillna(0)
        result["CUMGAS_MCF"] = result["CUMGAS_MCF"].fillna(0)
        
        return result
    except Exception as e:
        st.error(f"Error fetching well data: {e}")
        return None

# Update database record function
def update_database_record(table_name, primary_key, key_value, update_values, table_columns):
    """Execute update SQL using Snowpark Session"""
    try:
        # Handle DATE fields (convert empty strings to NULL)
        update_values = {
            col: f"'{value}'" if value else "NULL"
            if table_columns.get(col, "") != "DATE" else "NULL" if value == "" else f"'{value}'"
            for col, value in update_values.items()
        }

        set_clause = ", ".join([f"{col} = {value}" for col, value in update_values.items()])
        sql = f"UPDATE {table_name} SET {set_clause} WHERE {primary_key} = '{key_value}'"

        from snowflake.snowpark.context import get_active_session
        session = get_active_session()
        session.sql(sql).collect()
        return True, "Record updated successfully!"
    except Exception as e:
        return False, f"Error updating record: {e}"

##########################################
# SECTION 4: UTILITY FUNCTIONS - PRODUCTION DATA
##########################################
# Fetch Production Data
@st.cache_data(ttl=600)
def get_production_data(api_uwi):
    try:
        from snowflake.snowpark.context import get_active_session
        session = get_active_session()
        
        query = f"""
        SELECT API_UWI, ProducingMonth, LIQUIDSPROD_BBL, GASPROD_MCF
        FROM wells.minerals.raw_prod_data
        WHERE API_UWI = '{api_uwi}'
        ORDER BY ProducingMonth;
        """
        
        # Execute query and convert to pandas
        result = session.sql(query).to_pandas()
        
        # Debug information to understand what's being returned
        if len(result) == 0:
            print(f"No production data found for API_UWI: {api_uwi}")
        else:
            print(f"Found {len(result)} production records for API_UWI: {api_uwi}")
        
        return result
    except Exception as e:
        print(f"Error fetching production data for {api_uwi}: {str(e)}")
        return pd.DataFrame()

# Get the last production date for a well, separately for oil and gas
def get_last_production_dates(api_uwi):
    # Get production data and convert directly to pandas DataFrame
    raw_data = get_production_data(api_uwi)
    
    if raw_data.empty:
        print(f"No production data for {api_uwi}")
        return None, None
    
    try:
        # Print the first few rows to debug
        print(f"First few rows of production data for {api_uwi}:")
        print(raw_data.head())
        
        # Ensure LIQUIDSPROD_BBL and GASPROD_MCF are numeric
        raw_data['LIQUIDSPROD_BBL'] = pd.to_numeric(raw_data['LIQUIDSPROD_BBL'], errors='coerce').fillna(0)
        raw_data['GASPROD_MCF'] = pd.to_numeric(raw_data['GASPROD_MCF'], errors='coerce').fillna(0)
        
        # Find the last non-zero oil production
        non_zero_oil = raw_data[raw_data['LIQUIDSPROD_BBL'] > 0]
        last_oil_date = None
        if not non_zero_oil.empty:
            # Sort by date to get the most recent
            non_zero_oil = non_zero_oil.sort_values('PRODUCINGMONTH')
            last_oil_date = non_zero_oil['PRODUCINGMONTH'].iloc[-1]
            print(f"Found last oil date: {last_oil_date}")
        
        # Find the last non-zero gas production
        non_zero_gas = raw_data[raw_data['GASPROD_MCF'] > 0]
        last_gas_date = None
        if not non_zero_gas.empty:
            # Sort by date to get the most recent
            non_zero_gas = non_zero_gas.sort_values('PRODUCINGMONTH')
            last_gas_date = non_zero_gas['PRODUCINGMONTH'].iloc[-1]
            print(f"Found last gas date: {last_gas_date}")
        
        return last_oil_date, last_gas_date
    except Exception as e:
        print(f"Error in get_last_production_dates: {e}")
        # Print the column names for debugging
        print(f"Column names: {raw_data.columns.tolist()}")
        return None, None

##########################################
# SECTION 5: UTILITY FUNCTIONS - CALCULATIONS
##########################################
# Calculate Oil_Calc_Qi and Gas_Calc_Qi based on 3-month average around the last production date
def calculate_qi_values(api_uwi):
    raw_data = get_production_data(api_uwi)
    
    if raw_data.empty:
        print(f"No production data for {api_uwi} when calculating Qi values")
        return 0, 0
    
    try:
        # Get last production dates
        last_oil_date, last_gas_date = get_last_production_dates(api_uwi)
        
        # Ensure LIQUIDSPROD_BBL and GASPROD_MCF are numeric
        raw_data['LIQUIDSPROD_BBL'] = pd.to_numeric(raw_data['LIQUIDSPROD_BBL'], errors='coerce').fillna(0)
        raw_data['GASPROD_MCF'] = pd.to_numeric(raw_data['GASPROD_MCF'], errors='coerce').fillna(0)
        
        # Calculate Oil Qi
        oil_qi = 0
        if last_oil_date is not None:
            try:
                # Convert the date column to datetime if it isn't already
                if not pd.api.types.is_datetime64_any_dtype(raw_data['PRODUCINGMONTH']):
                    raw_data['PRODUCINGMONTH'] = pd.to_datetime(raw_data['PRODUCINGMONTH'], errors='coerce')
                
                # Ensure last_oil_date is datetime
                if not isinstance(last_oil_date, pd.Timestamp):
                    last_oil_date = pd.to_datetime(last_oil_date)
                
                # Calculate window - use 6 month window centered on last production date
                start_date = last_oil_date - pd.DateOffset(months=3)
                end_date = last_oil_date + pd.DateOffset(months=3)
                
                # Create window around last production date
                window_data = raw_data[
                    (raw_data['PRODUCINGMONTH'] >= start_date) & 
                    (raw_data['PRODUCINGMONTH'] <= end_date)
                ]
                
                # Calculate average of non-zero oil production in the window
                oil_window = window_data[window_data['LIQUIDSPROD_BBL'] > 0]
                if not oil_window.empty:
                    oil_qi = oil_window['LIQUIDSPROD_BBL'].mean()
                    print(f"Calculated oil Qi from {len(oil_window)} records: {oil_qi}")
            except Exception as e:
                print(f"Error calculating oil Qi: {e}")
        
        # Calculate Gas Qi
        gas_qi = 0
        if last_gas_date is not None:
            try:
                # Convert the date column to datetime if it isn't already
                if not pd.api.types.is_datetime64_any_dtype(raw_data['PRODUCINGMONTH']):
                    raw_data['PRODUCINGMONTH'] = pd.to_datetime(raw_data['PRODUCINGMONTH'], errors='coerce')
                
                # Ensure last_gas_date is datetime
                if not isinstance(last_gas_date, pd.Timestamp):
                    last_gas_date = pd.to_datetime(last_gas_date)
                
                # Calculate window - use 6 month window centered on last production date
                start_date = last_gas_date - pd.DateOffset(months=3)
                end_date = last_gas_date + pd.DateOffset(months=3)
                
                # Create window around last production date
                window_data = raw_data[
                    (raw_data['PRODUCINGMONTH'] >= start_date) & 
                    (raw_data['PRODUCINGMONTH'] <= end_date)
                ]
                
                # Calculate average of non-zero gas production in the window
                gas_window = window_data[window_data['GASPROD_MCF'] > 0]
                if not gas_window.empty:
                    gas_qi = gas_window['GASPROD_MCF'].mean()
                    print(f"Calculated gas Qi from {len(gas_window)} records: {gas_qi}")
            except Exception as e:
                print(f"Error calculating gas Qi: {e}")
        
        return oil_qi, gas_qi
    except Exception as e:
        print(f"Error in calculate_qi_values: {e}")
        return 0, 0

# Calculate Decline Fit with User-Defined Constants
def calculate_decline_fit(production_df, months=12, default_decline=0.06, min_decline=0.06, max_decline=0.98):
    if production_df.empty:
        return default_decline, default_decline
    
    # Sort by date if not already sorted
    if 'PRODUCINGMONTH' in production_df.columns:
        production_df = production_df.sort_values('PRODUCINGMONTH')
    
    # Limit to the most recent "months" number of records
    if len(production_df) > months:
        production_df = production_df.tail(months)
        
    # Create a copy for oil calculations
    oil_df = production_df.copy()
    oil_df['GASPROD_MCF'] = oil_df['LIQUIDSPROD_BBL']  # Use liquid production in place of gas for oil calculations

    def decline_rate(rates):
        # Need at least 6 points for calculation
        if len(rates) < 6:
            return default_decline
            
        # Create a smoothed version of the data
        smoothed = pd.Series(rates).rolling(6, center=True).mean().dropna()
        
        if len(smoothed) < 2:
            return default_decline
            
        # Calculate decline from first to last point in smoothed data
        # Avoid division by zero
        if smoothed.iloc[0] <= 0:
            return default_decline
            
        decline = (smoothed.iloc[0] - smoothed.iloc[-1]) / smoothed.iloc[0]
        
        # Convert to annual rate and ensure it's within bounds
        if decline < 0:  # Handle increasing production
            return min_decline
            
        # Calculate annualized decline rate
        annual_decline = 1 - (1 - decline) ** (12 / (len(smoothed) - 1))
        
        # Ensure result is within specified bounds
        return min(max(annual_decline, min_decline), max_decline)

    # Calculate decline rates
    oil_decline = decline_rate(oil_df['LIQUIDSPROD_BBL'].values)
    gas_decline = decline_rate(production_df['GASPROD_MCF'].values)

    return oil_decline, gas_decline

##########################################
# SECTION 6: SESSION STATE INITIALIZATION
##########################################
# Initialize variables for decline calculation constants 
# For bulk calculations
if 'months_for_calc' not in st.session_state:
    st.session_state['months_for_calc'] = 24
if 'default_decline' not in st.session_state:
    st.session_state['default_decline'] = 0.06
if 'min_decline' not in st.session_state:
    st.session_state['min_decline'] = 0.06
if 'max_decline' not in st.session_state:
    st.session_state['max_decline'] = 0.98
    
# For single well calculations
if 'single_months_for_calc' not in st.session_state:
    st.session_state['single_months_for_calc'] = 24
if 'single_default_decline' not in st.session_state:
    st.session_state['single_default_decline'] = 0.06
if 'single_min_decline' not in st.session_state:
    st.session_state['single_min_decline'] = 0.06
if 'single_max_decline' not in st.session_state:
    st.session_state['single_max_decline'] = 0.98

# Initialize session state for calculated declines if it doesn't exist
if 'calculated_declines' not in st.session_state:
    st.session_state['calculated_declines'] = {}

# Initialize session state for selected wells
if 'selected_wells' not in st.session_state:
    st.session_state['selected_wells'] = []

# Initialize session state for filtered wells
if 'filtered_wells' not in st.session_state:
    st.session_state['filtered_wells'] = None

# Initialize session state for forecast settings
if 'forecast_enabled' not in st.session_state:
    st.session_state['forecast_enabled'] = False
if 'forecast_years' not in st.session_state:
    st.session_state['forecast_years'] = 5
if 'show_oil_forecast' not in st.session_state:
    st.session_state['show_oil_forecast'] = True
if 'show_gas_forecast' not in st.session_state:
    st.session_state['show_gas_forecast'] = True
if 'forecast_data' not in st.session_state:
    st.session_state['forecast_data'] = {}

##########################################
# SECTION 7: FIELD DEFINITIONS
##########################################
# Define ordered oil and gas parameter fields
oil_fields = [
    "OIL_EMPIRICAL_DECLINE",
    "OIL_PROPHET_DECLINE",
    "OIL_DECLINE_TYPE",
    "OIL_USER_DECLINE",
    "OIL_CALC_QI",
    "OIL_USER_QI",
    "OIL_CALC_B_FACTOR",
    "OIL_USER_B_FACTOR",
    "OIL_D_MIN",
    "OIL_Q_MIN",
    "OIL_FCST_YRS"
]

gas_fields = [
    "GAS_EMPIRICAL_DECLINE",
    "GAS_PROPHET_DECLINE",
    "GAS_DECLINE_TYPE",
    "GAS_USER_DECLINE",
    "GAS_CALC_QI",
    "GAS_USER_QI",
    "GAS_CALC_B_FACTOR",
    "GAS_USER_B_FACTOR",
    "GAS_D_MIN",
    "GAS_Q_MIN",
    "GAS_FCST_YRS"
]

# Function to filter and order fields in table_columns
def get_ordered_fields(fields, all_columns):
    return [(field, all_columns.get(field)) for field in fields if field in all_columns]

# Replace the map visualization section with this simplified version
# This creates a two-column layout with filters on left and map on right

# Create a two-column layout for the entire filtering and map section
filter_column, map_column = st.columns([1, 1])

# Put all filters in the left column
with filter_column:
    st.subheader("Filter Wells")
    
    # Load well data for filtering
    well_data = get_well_data()
    
    if well_data is not None:
        # Define filter columns in the specified order
        filter_columns = [
            "TRAJECTORY", 
            "COUNTY", 
            "ENVWELLSTATUS", 
            "ENVOPERATOR", 
            "WELLNAME", 
            "API_UWI"
        ]
        
        # Initialize filtered_wells to be the original well data
        filtered_wells = well_data.copy()
        
        # Apply filters - using full width since we're in a column already
        for col in filter_columns:
            if col in well_data.columns:
                # Get unique values for the current filter, based on data filtered so far
                unique_values = sorted(filtered_wells[col].dropna().unique().tolist())
                
                # Create multiselect widget
                selected_values = st.multiselect(
                    f"Select {col}:",
                    options=unique_values,
                    default=[]
                )
                
                # Apply filter if values are selected
                if selected_values:
                    filtered_wells = filtered_wells[filtered_wells[col].isin(selected_values)]
        
        # Add range sliders for production data
        # Oil production range slider
        if "CUMOIL_BBL" in well_data.columns:
            max_oil_value = int(well_data["CUMOIL_BBL"].max())
            oil_range = st.slider(
                "Total Oil Production (BBL)", 
                min_value=0, 
                max_value=max_oil_value, 
                value=(0, max_oil_value)
            )
            filtered_wells = filtered_wells[(filtered_wells["CUMOIL_BBL"] >= oil_range[0]) & (filtered_wells["CUMOIL_BBL"] <= oil_range[1])]
    
        # Total producing months range slider
        if "TOTALPRODUCINGMONTHS" in well_data.columns:
            max_months = int(well_data["TOTALPRODUCINGMONTHS"].max())
            months_range = st.slider(
                "Total Producing Months", 
                min_value=0, 
                max_value=max_months, 
                value=(0, max_months)
            )
            filtered_wells = filtered_wells[(filtered_wells["TOTALPRODUCINGMONTHS"] >= months_range[0]) & (filtered_wells["TOTALPRODUCINGMONTHS"] <= months_range[1])]
    
        # Gas production range slider
        if "CUMGAS_MCF" in well_data.columns:
            max_gas_value = int(well_data["CUMGAS_MCF"].max())
            gas_range = st.slider(
                "Total Gas Production (MCF)", 
                min_value=0, 
                max_value=max_gas_value, 
                value=(0, max_gas_value)
            )
            filtered_wells = filtered_wells[(filtered_wells["CUMGAS_MCF"] >= gas_range[0]) & (filtered_wells["CUMGAS_MCF"] <= gas_range[1])]
        
        # Display filtered well count
        st.write(f"Filtered Wells: {len(filtered_wells)}")
        
        # Add button to use these filtered wells for operations
        if not filtered_wells.empty:
            if st.button("Use Filtered Wells for Updating"):
                st.session_state['filtered_wells'] = filtered_wells
                st.session_state['selected_wells'] = filtered_wells["API_UWI"].tolist()
                st.success(f"Selected {len(filtered_wells)} wells for operations.")
                # Force refresh to show filtered data
                st.rerun()
    else:
        st.warning("Could not load well data for filtering.")

# Put the map in the right column
with map_column:
    st.subheader("Well Map")
    
    # Only proceed if we have well data and filtered wells
    if 'filtered_wells' in locals() and not filtered_wells.empty:
        if 'LATITUDE' in filtered_wells.columns and 'LONGITUDE' in filtered_wells.columns:
            # Create a copy with just the needed columns
            map_data = filtered_wells[['LATITUDE', 'LONGITUDE']].copy()
            
            # Rename columns to what st.map expects
            map_data.columns = ['latitude', 'longitude']
            
            # Drop any rows with missing coordinates
            map_data = map_data.dropna()
            
            if not map_data.empty:
                # Simple fixed-size map - no color or size variations
                st.map(map_data)
                
                # Add information about the map
                st.info(f"Map shows {len(map_data)} filtered wells.")
                
                # Show a sample of filtered wells in a table below the map
                st.subheader("Filtered Wells Sample")
                display_cols = ["API_UWI", "WELLNAME", "ENVOPERATOR", "COUNTY", "CUMOIL_BBL", "CUMGAS_MCF"]
                display_cols = [col for col in display_cols if col in filtered_wells.columns]
                st.dataframe(filtered_wells[display_cols].head(15))
            else:
                st.warning("Cannot display map: No valid coordinate data available")
        else:
            st.warning("Cannot display map: Latitude or longitude data is missing")
    else:
        st.info("No filtered wells to display. Use the filters on the left to select wells.")

# Add visualization of top operators and counties after the two-column layout
# This is outside the columns to use full width
if 'filtered_wells' in locals() and not filtered_wells.empty:
    st.subheader("Well Distribution")
    
    # Create tabs for different visualizations
    viz_tab1, viz_tab2 = st.tabs(["By Operator", "By County"])
    
    with viz_tab1:
        operator_counts = filtered_wells["ENVOPERATOR"].value_counts().reset_index()
        operator_counts.columns = ["Operator", "Count"]
        # Limit to top 10 operators
        if len(operator_counts) > 10:
            operator_counts = operator_counts.head(10)
            title_operator = "Top 10 Operators"
        else:
            title_operator = "Operators"

        chart_operator = alt.Chart(operator_counts).mark_bar().encode(
            x=alt.X("Count:Q", title="Count"),
            y=alt.Y("Operator:N", title="Operator", sort="-x"),
            tooltip=["Operator", "Count"]
        ).properties(title=title_operator, height=250)
        
        st.altair_chart(chart_operator, use_container_width=True)
    
    with viz_tab2:
        county_counts = filtered_wells["COUNTY"].value_counts().reset_index()
        county_counts.columns = ["County", "Count"]
        # Limit to top 10 counties
        if len(county_counts) > 10:
            county_counts = county_counts.head(10)
            title_county = "Top 10 Counties"
        else:
            title_county = "Counties"

        chart_county = alt.Chart(county_counts).mark_bar().encode(
            x=alt.X("Count:Q", title="Count"),
            y=alt.Y("County:N", title="County", sort="-x"),
            tooltip=["County", "Count"]
        ).properties(title=title_county, height=250)
        
        st.altair_chart(chart_county, use_container_width=True)
        
##########################################
# SECTION 9: SIDEBAR FILTERS & DATA DISPLAY
##########################################
# Sidebar Filters for original functionality
if table_name:
    table_columns = get_table_columns(table_name)
else:
    table_columns = {}

if table_columns:
    st.sidebar.subheader("Original Table Filters")
    filter_values = {col: st.sidebar.text_input(f"Filter by {col}") for col in list(table_columns.keys())[:3]}
    where_clause = " AND ".join([f"{col} LIKE '%%{value}%%'" for col, value in filter_values.items() if value])
else:
    where_clause = ""

# Fetch & Display Data
if table_name:
    data = get_table_data(table_name, where_clause)
    
    # Apply additional well filtering if wells have been selected from the map
    if st.session_state['filtered_wells'] is not None and primary_key in data.columns:
        # Get the list of selected API_UWIs
        selected_api_uwis = st.session_state['selected_wells']
        
        # Filter the data to only include the selected wells
        if selected_api_uwis:
            data = data[data[primary_key].isin(selected_api_uwis)]
            table_title = f"{table_name} Data (Filtered to {len(data)} selected wells)"
        else:
            table_title = f"{table_name} Data ({len(data)} rows)"
    else:
        table_title = f"{table_name} Data ({len(data)} rows)"
    
    # Make the data table section minimizable with an expander
    with st.expander(table_title, expanded=False):
        # Create two columns - one for econ input table, one for production data
        econ_col, prod_col = st.columns([1, 1])
        
        with econ_col:
            st.subheader("Econ Input Data")
            st.dataframe(data)
        
        with prod_col:
            st.subheader("Production Data")
            
            # Add a well selector for the production data
            if primary_key in data.columns and not data.empty:
                prod_well = st.selectbox("Select Well for Production Data", 
                                        options=data[primary_key].tolist(),
                                        index=0,
                                        key="prod_data_well_selector")
                
                # Get production data for selected well
                prod_data = get_production_data(prod_well)
                
                if prod_data.empty:
                    st.warning(f"No production data available for {prod_well}")
                else:
                    # Calculate statistics
                    total_oil = prod_data['LIQUIDSPROD_BBL'].sum()
                    total_gas = prod_data['GASPROD_MCF'].sum()
                    
                    # Display statistics and data
                    st.markdown(f"**Records:** {len(prod_data)} | **Oil:** {total_oil:,.0f} BBL | **Gas:** {total_gas:,.0f} MCF")
                    
                    # Format date column for display
                    if 'PRODUCINGMONTH' in prod_data.columns:
                        prod_data['PRODUCINGMONTH'] = pd.to_datetime(prod_data['PRODUCINGMONTH']).dt.strftime('%Y-%m-%d')
                    
                    # Display sorted data
                    st.dataframe(prod_data.sort_values('PRODUCINGMONTH', ascending=False))
                    
                    # Add download button for CSV
                    csv = prod_data.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="{prod_well}_production.csv">Download CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)
            else:
                st.warning(f"Primary key '{primary_key}' not found in the table or data is empty.")
else:
    data = pd.DataFrame()
    st.warning("Please enter a table name")

# Main content with expandable sections
st.title("Well Data Updating")

##########################################
# SECTION 10: SINGLE WELL UPDATING
##########################################
with st.expander("Single Well Updating", expanded=False):
    st.subheader("Update Individual Well Data")
    
    if not data.empty and primary_key in data.columns:
        selected_key = st.selectbox("Select Well", options=data[primary_key].tolist())

        if selected_key:
            record = data[data[primary_key] == selected_key].iloc[0]

            # Get production data and last production dates once for use throughout this section
            production_data = get_production_data(selected_key)
            last_oil_date, last_gas_date = get_last_production_dates(selected_key)
            
            # Calculate Qi values once
            oil_qi, gas_qi = calculate_qi_values(selected_key)

            # Create a layout with two columns - with the left column taking less space
            left_column, right_column = st.columns([1, 1])
            
            with left_column:
                # Put all inputs in the left column, split by oil and gas
                st.subheader("Well Parameters")
                
                # Calculate button for this well in a prominent position
                if st.button("Calculate Decline Rates", key="single_calc", use_container_width=True):
                    with st.spinner(f"Calculating decline rates for {selected_key}..."):
                        if production_data.empty:
                            st.warning(f"No production data found for {selected_key}")
                        else:
                            # Calculate decline rates using single well parameters
                            oil_decline, gas_decline = calculate_decline_fit(
                                production_data, st.session_state['single_months_for_calc'], 
                                st.session_state['single_default_decline'], 
                                st.session_state['single_min_decline'], 
                                st.session_state['single_max_decline']
                            )
                            
                            # Store in session state
                            st.session_state['calculated_declines'][selected_key] = (oil_decline, gas_decline)
                            
                            # Display results
                            st.success("Decline rates calculated")
                
                # Show calculated rates if available in a more prominent way
                if selected_key in st.session_state['calculated_declines']:
                    oil_decline, gas_decline = st.session_state['calculated_declines'][selected_key]
                    oil_col1, gas_col1 = st.columns(2)
                    with oil_col1:
                        st.metric("Oil Decline Rate", f"{oil_decline:.6f}")
                    with gas_col1:
                        st.metric("Gas Decline Rate", f"{gas_decline:.6f}")
                
                # Create two tabs for Oil and Gas parameters
                oil_tab, gas_tab = st.tabs(["Oil Parameters", "Gas Parameters"])
                
                with oil_tab:
                    # Add a small section for the oil decline constants
                    st.subheader("Oil Decline Parameters")
                    oil_param_col1, oil_param_col2 = st.columns(2)
                    with oil_param_col1:
                        st.session_state['single_months_for_calc'] = st.slider(
                            "Months for Calculation", 
                            6, 60, st.session_state['single_months_for_calc'],
                            help="Number of most recent months to use for this well"
                        )
                        st.session_state['single_default_decline'] = st.number_input(
                            "Default Rate", 
                            value=st.session_state['single_default_decline'], 
                            format="%.6f",
                            help="Fallback rate when insufficient data"
                        )
                    with oil_param_col2:
                        st.session_state['single_min_decline'] = st.number_input(
                            "Minimum Rate", 
                            value=st.session_state['single_min_decline'], 
                            format="%.6f",
                            help="Minimum allowed decline rate"
                        )
                        st.session_state['single_max_decline'] = st.number_input(
                            "Maximum Rate", 
                            value=st.session_state['single_max_decline'], 
                            format="%.6f",
                            help="Maximum allowed decline rate"
                        )
                
                with gas_tab:
                    # Add info about gas parameters or additional controls if needed
                    st.info("Gas decline calculations use the same parameters as oil.")
                    
                # Display other well information
                if 'ENVOPERATOR' in record:
                    st.markdown(f"**Operator:** {record['ENVOPERATOR']}")
                if 'WELLNAME' in record:
                    st.markdown(f"**Well Name:** {record['WELLNAME']}")
                if 'ENVWELLSTATUS' in record:
                    st.markdown(f"**Status:** {record['ENVWELLSTATUS']}")
                if 'FIRSTPRODDATE' in record:
                    st.markdown(f"**First Production:** {record['FIRSTPRODDATE']}")
                
                # Add the update form directly in the left column
                
                # Update form
                with st.form("update_form", clear_on_submit=False):
                    st.subheader("Update Record Fields")
                    
                    # Get ordered fields that actually exist in the table
                    existing_oil_fields = get_ordered_fields(oil_fields, table_columns)
                    existing_gas_fields = get_ordered_fields(gas_fields, table_columns)
                    
                    # Create two columns for oil and gas parameters
                    oil_col, gas_col = st.columns(2)
                    
                    # Oil parameters column
                    with oil_col:
                        st.markdown("### Oil Parameters")
                        oil_values = {}
                        
                        for field, data_type in existing_oil_fields:
                            current_value = record[field] if field in record else ""
                            oil_values[field] = st.text_input(
                                f"{field} (Current: {current_value})", 
                                value=str(current_value) if current_value else ""
                            )
                            
                            # Automatically populate OIL_CALC_QI if available
                            if field == "OIL_CALC_QI" and selected_key:
                                if oil_qi > 0:
                                    st.info(f"Calculated Oil Initial Rate (Qi): {oil_qi:.2f}")
                                    oil_values['OIL_CALC_QI'] = str(oil_qi)
                    
                    # Gas parameters column
                    with gas_col:
                        st.markdown("### Gas Parameters")
                        gas_values = {}
                        
                        for field, data_type in existing_gas_fields:
                            current_value = record[field] if field in record else ""
                            gas_values[field] = st.text_input(
                                f"{field} (Current: {current_value})", 
                                value=str(current_value) if current_value else ""
                            )
                            
                            # Automatically populate GAS_CALC_QI if available
                            if field == "GAS_CALC_QI" and selected_key:
                                if gas_qi > 0:
                                    st.info(f"Calculated Gas Initial Rate (Qi): {gas_qi:.2f}")
                                    gas_values['GAS_CALC_QI'] = str(gas_qi)
                    
                    # Other fields (not oil or gas specific)
                    st.markdown("### Other Parameters")
                    other_fields = {}
                    for col in table_columns:
                        if (col != primary_key and 
                            col.lower() != 'lease' and
                            col not in oil_values and 
                            col not in gas_values):
                            other_fields[col] = st.text_input(
                                f"{col} (Current: {record[col]})", 
                                value=str(record[col]) if record[col] else ""
                            )
                    
                    # Combine all update values
                    update_values = {**oil_values, **gas_values, **other_fields}
                    
                    # Add last production dates if those fields exist
                    if ('LAST_OIL_DATE' in table_columns or 'LAST_GAS_DATE' in table_columns or 
                        'LAST_PROD_DATE' in table_columns):
                        
                        # Update individual date fields
                        if last_oil_date is not None and 'LAST_OIL_DATE' in table_columns:
                            st.info(f"Last Oil Production Date: {last_oil_date}")
                            update_values['LAST_OIL_DATE'] = str(last_oil_date)
                        if last_gas_date is not None and 'LAST_GAS_DATE' in table_columns:
                            st.info(f"Last Gas Production Date: {last_gas_date}")
                            update_values['LAST_GAS_DATE'] = str(last_gas_date)
                        
                        # Determine and update the most recent production date
                        if 'LAST_PROD_DATE' in table_columns:
                            most_recent_date = None
                            if last_oil_date is not None and last_gas_date is not None:
                                most_recent_date = max(last_oil_date, last_gas_date)
                            elif last_oil_date is not None:
                                most_recent_date = last_oil_date
                            elif last_gas_date is not None:
                                most_recent_date = last_gas_date
                            
                            if most_recent_date is not None:
                                st.info(f"Last Production Date (Most Recent): {most_recent_date}")
                                update_values['LAST_PROD_DATE'] = str(most_recent_date)
                    
                    # Option to apply calculated decline rates
                    apply_calc_decline = st.checkbox("Apply calculated decline rates (if available)")
                    
                    submit_button = st.form_submit_button("Update Record")

                    if submit_button:
                        # Apply decline rates if checkbox is selected and values exist in session state
                        if apply_calc_decline and selected_key in st.session_state['calculated_declines']:
                            oil_decline, gas_decline = st.session_state['calculated_declines'][selected_key]
                            update_values["OIL_EMPIRICAL_DECLINE"] = str(oil_decline)
                            update_values["GAS_EMPIRICAL_DECLINE"] = str(gas_decline)

                        # Call the update function
                        success, message = update_database_record(
                            table_name, 
                            primary_key, 
                            selected_key, 
                            update_values, 
                            table_columns
                        )
                        
                        if success:
                            st.success(f"Record {primary_key} = {selected_key} {message}")
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.error(message)
            
            with right_column:
                st.subheader("Production Data Visualization")
                
                # Add forecast controls
                st.markdown("### Forecast Controls")
                forecast_col1, forecast_col2 = st.columns(2)

                with forecast_col1:
                    st.session_state['forecast_enabled'] = st.checkbox("Enable Forecast", st.session_state['forecast_enabled'])
                    
                    if st.session_state['forecast_enabled']:
                        st.session_state['show_oil_forecast'] = st.checkbox("Show Oil Forecast", st.session_state['show_oil_forecast'])
                        st.session_state['show_gas_forecast'] = st.checkbox("Show Gas Forecast", st.session_state['show_gas_forecast'])
                        live_forecast = st.checkbox("Live Forecast", 
                                                  value=False,
                                                  help="Automatically update forecast when parameters change")

                with forecast_col2:
                    if st.session_state['forecast_enabled']:
                        st.session_state['forecast_years'] = st.slider("Forecast Years", 1, 20, st.session_state['forecast_years'])
                        
                        if not live_forecast:
                            generate_forecast = st.button("Generate Forecast", key="gen_forecast_btn")
                            
                            if generate_forecast:
                                try:
                                    with st.spinner("Generating forecast..."):
                                        # Call the SingleWellForecast function with appropriate data
                                        forecast_df = SingleWellForecast(
                                            selected_key,  # API_UWI of selected well
                                            data,  # ECON_INPUT data
                                            production_data  # Production data for the well
                                        )
                                        # Store in session state
                                        st.session_state['forecast_data'][selected_key] = forecast_df
                                        st.success("Forecast generated successfully!")
                                except Exception as e:
                                    st.error(f"Error generating forecast: {str(e)}")
                                    st.error("Ensure all required decline curve parameters are set for this well.")
                
                # Generate live forecast if enabled
                if st.session_state['forecast_enabled'] and live_forecast:
                    # Create a dictionary to store the current parameters
                    forecast_params = {}
                    
                    # Get current values from the form fields - careful handling of potential invalid values
                    try:
                        # Oil parameters
                        forecast_params['OIL_EMPIRICAL_DECLINE'] = float(oil_values.get('OIL_EMPIRICAL_DECLINE', 0.06))
                        forecast_params['OIL_DECLINE_TYPE'] = oil_values.get('OIL_DECLINE_TYPE', 'EXP')
                        forecast_params['OIL_USER_DECLINE'] = float(oil_values.get('OIL_USER_DECLINE', 0)) if oil_values.get('OIL_USER_DECLINE') and oil_values.get('OIL_USER_DECLINE').lower() != 'nan' else None
                        forecast_params['OIL_CALC_QI'] = float(oil_values.get('OIL_CALC_QI', 0))
                        forecast_params['OIL_USER_QI'] = float(oil_values.get('OIL_USER_QI', 0)) if oil_values.get('OIL_USER_QI') and oil_values.get('OIL_USER_QI').lower() != 'nan' else None
                        forecast_params['OIL_CALC_B_FACTOR'] = float(oil_values.get('OIL_CALC_B_FACTOR', 0)) if oil_values.get('OIL_CALC_B_FACTOR') and oil_values.get('OIL_CALC_B_FACTOR').lower() != 'nan' else 0
                        forecast_params['OIL_USER_B_FACTOR'] = float(oil_values.get('OIL_USER_B_FACTOR', 0)) if oil_values.get('OIL_USER_B_FACTOR') and oil_values.get('OIL_USER_B_FACTOR').lower() != 'nan' else None
                        forecast_params['OIL_D_MIN'] = float(oil_values.get('OIL_D_MIN', 0.06))
                        forecast_params['OIL_Q_MIN'] = float(oil_values.get('OIL_Q_MIN', 1.0))
                        
                        # Gas parameters
                        forecast_params['GAS_EMPIRICAL_DECLINE'] = float(gas_values.get('GAS_EMPIRICAL_DECLINE', 0.06))
                        forecast_params['GAS_DECLINE_TYPE'] = gas_values.get('GAS_DECLINE_TYPE', 'EXP')
                        forecast_params['GAS_USER_DECLINE'] = float(gas_values.get('GAS_USER_DECLINE', 0)) if gas_values.get('GAS_USER_DECLINE') and gas_values.get('GAS_USER_DECLINE').lower() != 'nan' else None
                        forecast_params['GAS_CALC_QI'] = float(gas_values.get('GAS_CALC_QI', 0))
                        forecast_params['GAS_USER_QI'] = float(gas_values.get('GAS_USER_QI', 0)) if gas_values.get('GAS_USER_QI') and gas_values.get('GAS_USER_QI').lower() != 'nan' else None
                        forecast_params['GAS_CALC_B_FACTOR'] = float(gas_values.get('GAS_CALC_B_FACTOR', 0)) if gas_values.get('GAS_CALC_B_FACTOR') and gas_values.get('GAS_CALC_B_FACTOR').lower() != 'nan' else 0
                        forecast_params['GAS_USER_B_FACTOR'] = float(gas_values.get('GAS_USER_B_FACTOR', 0)) if gas_values.get('GAS_USER_B_FACTOR') and gas_values.get('GAS_USER_B_FACTOR').lower() != 'nan' else None
                        forecast_params['GAS_D_MIN'] = float(gas_values.get('GAS_D_MIN', 0.06))
                        forecast_params['GAS_Q_MIN'] = float(gas_values.get('GAS_Q_MIN', 1.0))
                        
                        # Create a modified copy of the data record with our updated parameters
                        temp_data = data.copy()
                        for field, value in forecast_params.items():
                            if value is not None:
                                temp_data.loc[temp_data[primary_key] == selected_key, field] = value
                        
                        # Generate live forecast
                        try:
                            with st.spinner("Updating live forecast..."):
                                live_forecast_df = SingleWellForecast(
                                    selected_key,
                                    temp_data,
                                    production_data
                                )
                                # Store in session state under a different key
                                st.session_state['live_forecast_data'] = live_forecast_df
                        except Exception as e:
                            st.error(f"Error updating live forecast: {str(e)}")
                    except Exception as e:
                        st.error(f"Invalid parameter value: {str(e)}")
                
                if production_data.empty:
                    st.warning(f"No production data available for {selected_key}")
                else:
                    # Add visualization controls
                    control_col1, control_col2, control_col3 = st.columns(3)
                    
                    with control_col1:
                        log_scale = st.checkbox("Log Scale Y-Axis", False)
                    
                    with control_col2:
                        # Add time range selector (last X months/years)
                        time_options = ["All Data", "Last Year", "Last 2 Years", "Last 5 Years"]
                        time_selection = st.selectbox("Time Range", time_options)
                    
                    with control_col3:
                        # Add option to show/hide oil or gas
                        show_oil = st.checkbox("Show Oil", True)
                        show_gas = st.checkbox("Show Gas", True)
                        # Add option to filter zeros when using log scale
                        filter_zeros = st.checkbox("Filter Zero Values", True)
                    
                    # Ensure data types are correct and dates are in datetime format
                    production_data['PRODUCINGMONTH'] = pd.to_datetime(production_data['PRODUCINGMONTH'])
                    production_data['LIQUIDSPROD_BBL'] = pd.to_numeric(production_data['LIQUIDSPROD_BBL'], errors='coerce').fillna(0)
                    production_data['GASPROD_MCF'] = pd.to_numeric(production_data['GASPROD_MCF'], errors='coerce').fillna(0)
                    
                    # Sort the data by date
                    production_data = production_data.sort_values('PRODUCINGMONTH')
                    
                    # Filter data based on time selection
                    if time_selection != "All Data":
                        end_date = production_data['PRODUCINGMONTH'].max()
                        if time_selection == "Last Year":
                            start_date = end_date - pd.DateOffset(years=1)
                        elif time_selection == "Last 2 Years":
                            start_date = end_date - pd.DateOffset(years=2)
                        elif time_selection == "Last 5 Years":
                            start_date = end_date - pd.DateOffset(years=5)
                        
                        filtered_data = production_data[production_data['PRODUCINGMONTH'] >= start_date]
                        if not filtered_data.empty:
                            production_data = filtered_data
                        else:
                            st.warning(f"No data available for the selected time range. Showing all data.")
                    
                    # Create a DataFrame for visualization with selected series
                    chart_data = pd.DataFrame()
                    chart_data['Production Month'] = production_data['PRODUCINGMONTH']
                    
                    if show_oil:
                        if filter_zeros and log_scale:
                            # Filter out zero values for oil when using log scale
                            oil_data = production_data[production_data['LIQUIDSPROD_BBL'] > 0]
                            if not oil_data.empty:
                                chart_data['Oil (BBL)'] = pd.Series(dtype='float64')  # Initialize as empty series
                                # Map oil data to chart_data based on matching dates
                                for idx, row in oil_data.iterrows():
                                    date_mask = chart_data['Production Month'] == row['PRODUCINGMONTH']
                                    if date_mask.any():
                                        chart_data.loc[date_mask, 'Oil (BBL)'] = row['LIQUIDSPROD_BBL']
                        else:
                            chart_data['Oil (BBL)'] = production_data['LIQUIDSPROD_BBL']
                    
                    if show_gas:
                        if filter_zeros and log_scale:
                            # Filter out zero values for gas when using log scale
                            gas_data = production_data[production_data['GASPROD_MCF'] > 0]
                            if not gas_data.empty:
                                chart_data['Gas (MCF)'] = pd.Series(dtype='float64')  # Initialize as empty series
                                # Map gas data to chart_data based on matching dates
                                for idx, row in gas_data.iterrows():
                                    date_mask = chart_data['Production Month'] == row['PRODUCINGMONTH']
                                    if date_mask.any():
                                        chart_data.loc[date_mask, 'Gas (MCF)'] = row['GASPROD_MCF']
                        else:
                            chart_data['Gas (MCF)'] = production_data['GASPROD_MCF']
                    
                    # Melt the dataframe to create a format suitable for Altair
                    if ('Oil (BBL)' in chart_data.columns or 'Gas (MCF)' in chart_data.columns):
                        id_vars = ['Production Month']
                        value_vars = []
                        if 'Oil (BBL)' in chart_data.columns:
                            value_vars.append('Oil (BBL)')
                        if 'Gas (MCF)' in chart_data.columns:
                            value_vars.append('Gas (MCF)')
                        
                        melted_data = chart_data.melt(
                            id_vars=id_vars,
                            value_vars=value_vars,
                            var_name='Production Type',
                            value_name='Volume'
                        )
                        
                        # Drop rows with NaN Volume (filtered zeros)
                        melted_data = melted_data.dropna(subset=['Volume'])
                        
                        # Create scale based on log toggle
                        if log_scale:
                            y_scale = alt.Scale(type='log', domainMin=1)
                            y_title = 'Production Volume (Log Scale)'
                        else:
                            y_scale = alt.Scale(zero=True)
                            y_title = 'Production Volume'
                        
                        # Include forecast data if enabled and available
                        forecast_df = None
                        if st.session_state['forecast_enabled']:
                            # Decision tree for forecast data source:
                            # 1. Use live forecast if it's enabled
                            # 2. Otherwise use saved forecast if available
                            if live_forecast and 'live_forecast_data' in st.session_state:
                                forecast_df = st.session_state['live_forecast_data']
                                forecast_source = "Live forecast"
                            elif selected_key in st.session_state['forecast_data']:
                                forecast_df = st.session_state['forecast_data'][selected_key]
                                forecast_source = "Saved forecast"
                            
                            if forecast_df is not None:
                                # Limit forecast to specified number of years
                                max_forecast_date = pd.to_datetime(melted_data['Production Month'].max()) + pd.DateOffset(years=st.session_state['forecast_years'])
                                forecast_df = forecast_df[forecast_df['PRODUCINGMONTH'] <= max_forecast_date]
                                
                                # Create forecast data for visualization
                                forecast_chart_data = pd.DataFrame()
                                forecast_chart_data['Production Month'] = forecast_df['PRODUCINGMONTH']
                                
                                if st.session_state['show_oil_forecast'] and 'OilFcst_BBL' in forecast_df.columns:
                                    forecast_chart_data['Oil Forecast (BBL)'] = forecast_df['OilFcst_BBL']
                                
                                if st.session_state['show_gas_forecast'] and 'GasFcst_MCF' in forecast_df.columns:
                                    forecast_chart_data['Gas Forecast (MCF)'] = forecast_df['GasFcst_MCF']
                                
                                # Melt the forecast dataframe
                                forecast_value_vars = []
                                if 'Oil Forecast (BBL)' in forecast_chart_data.columns:
                                    forecast_value_vars.append('Oil Forecast (BBL)')
                                if 'Gas Forecast (MCF)' in forecast_chart_data.columns:
                                    forecast_value_vars.append('Gas Forecast (MCF)')
                                
                                if forecast_value_vars:
                                    melted_forecast = forecast_chart_data.melt(
                                        id_vars=['Production Month'],
                                        value_vars=forecast_value_vars,
                                        var_name='Production Type',
                                        value_name='Volume'
                                    )
                                    
                                    # Drop any NaN values in the forecast data
                                    melted_forecast = melted_forecast.dropna(subset=['Volume'])
                                    
                                    # Append to the historical data
                                    melted_data = pd.concat([melted_data, melted_forecast], ignore_index=True)
                        
                        # Update the color mapping to use green for oil and red for gas
                        color_mapping = {
                            'Oil (BBL)': '#1e8f4e',        # Green for oil
                            'Gas (MCF)': '#d62728',         # Red for gas
                            'Oil Forecast (BBL)': '#1e8f4e',  # Same green for oil forecast
                            'Gas Forecast (MCF)': '#d62728'   # Same red for gas forecast
                        }
                        
                        chart = alt.Chart(melted_data).encode(
                            x=alt.X('Production Month:T', title='Production Month'),
                            y=alt.Y('Volume:Q', scale=y_scale, title=y_title),
                            color=alt.Color('Production Type:N', 
                                            scale=alt.Scale(domain=list(color_mapping.keys()), 
                                                            range=list(color_mapping.values())),
                                            legend=alt.Legend(title='Production Type')),
                            tooltip=['Production Month', 'Production Type', 'Volume']
                        )
                        
                        # Create separate line marks for historical data (solid) and forecast data (dashed)
                        historical_chart = chart.transform_filter(
                            alt.FieldOneOfPredicate(field='Production Type', oneOf=['Oil (BBL)', 'Gas (MCF)'])
                        ).mark_line(point=True)
                        
                        forecast_chart = chart.transform_filter(
                            alt.FieldOneOfPredicate(field='Production Type', oneOf=['Oil Forecast (BBL)', 'Gas Forecast (MCF)'])
                        ).mark_line(point=True, strokeDash=[6, 2])  # Dashed line for forecast
                        
                        # Create chart title with forecast indicator if applicable
                        chart_title = f"Production History and Forecast for {selected_key}"
                        if forecast_df is not None and 'forecast_source' in locals():
                            chart_title += f" ({forecast_source})"
                        
                        # Combine the charts
                        final_chart = (historical_chart + forecast_chart).properties(
                            title=chart_title,
                            height=400
                        ).interactive()
                        
                        # Display the chart
                        st.altair_chart(final_chart, use_container_width=True)
                        
                        # Add a note about zero filtering if active
                        if log_scale and filter_zeros:
                            st.caption("Note: Zero values are filtered out when using log scale to prevent display artifacts.")
                    else:
                        st.warning("Please select at least one data series to display (Oil or Gas)")
                    
                    # Display production statistics and dates in a well-organized format below the plot
                    st.markdown("---")
                    
                    # Create two columns for oil and gas statistics
                    stat_col1, stat_col2 = st.columns(2)
                    
                    with stat_col1:
                        st.markdown("### Oil Statistics")
                        st.metric("Total Oil (BBL)", f"{production_data['LIQUIDSPROD_BBL'].sum():,.0f}")
                        st.metric("Avg Oil (BBL/month)", f"{production_data['LIQUIDSPROD_BBL'].mean():,.0f}")
                    
                    with stat_col2:
                        st.markdown("### Gas Statistics")
                        st.metric("Total Gas (MCF)", f"{production_data['GASPROD_MCF'].sum():,.0f}")
                        st.metric("Avg Gas (MCF/month)", f"{production_data['GASPROD_MCF'].mean():,.0f}")
                    
                    # Display all dates and rates in organized sections
                    date_col1, date_col2 = st.columns(2)
                    
                    with date_col1:
                        st.markdown("### Production Dates")
                        if last_oil_date is not None:
                            st.markdown(f"**Last Oil Date:** {last_oil_date.strftime('%Y-%m-%d')}")
                        if last_gas_date is not None:
                            st.markdown(f"**Last Gas Date:** {last_gas_date.strftime('%Y-%m-%d')}")
                            
                        # Calculate overall last production date
                        most_recent_date = None
                        if last_oil_date is not None and last_gas_date is not None:
                            most_recent_date = max(last_oil_date, last_gas_date)
                        elif last_oil_date is not None:
                            most_recent_date = last_oil_date
                        elif last_gas_date is not None:
                            most_recent_date = last_gas_date
                            
                        if most_recent_date is not None:
                            st.markdown(f"**Last Production Date:** {most_recent_date.strftime('%Y-%m-%d')}")
                    with date_col2:
                        # Initial Rates (Qi)
                        st.markdown("### Initial Rates (Qi)")
                        if oil_qi > 0:
                            st.markdown(f"**Oil Qi:** {oil_qi:.2f} BBL/month")
                        if gas_qi > 0:
                            st.markdown(f"**Gas Qi:** {gas_qi:.2f} MCF/month")
                    
                    # Add forecast statistics section
                    # Determine which forecast to use for statistics
                    forecast_df = None
                    if st.session_state['forecast_enabled']:
                        if live_forecast and 'live_forecast_data' in st.session_state:
                            forecast_df = st.session_state['live_forecast_data']
                            forecast_label = "Live Forecast"
                        elif selected_key in st.session_state['forecast_data']:
                            forecast_df = st.session_state['forecast_data'][selected_key]
                            forecast_label = "Saved Forecast"
                        
                        if forecast_df is not None:
                            st.markdown("---")
                            st.markdown(f"### {forecast_label} Statistics")
                            
                            # Calculate forecast totals
                            oil_forecast_total = forecast_df['OilFcst_BBL'].sum()
                            gas_forecast_total = forecast_df['GasFcst_MCF'].sum()
                            
                            # Create two columns for oil and gas forecast stats
                            fcst_col1, fcst_col2 = st.columns(2)
                            
                            with fcst_col1:
                                st.markdown("#### Oil Forecast")
                                st.metric("Total Forecast Oil (BBL)", f"{oil_forecast_total:,.0f}")
                                
                                # Calculate EUR (Estimated Ultimate Recovery) for oil
                                total_historical_oil = production_data['LIQUIDSPROD_BBL'].sum()
                                oil_eur = total_historical_oil + oil_forecast_total
                                st.metric("Oil EUR (BBL)", f"{oil_eur:,.0f}")
                            
                            with fcst_col2:
                                st.markdown("#### Gas Forecast")
                                st.metric("Total Forecast Gas (MCF)", f"{gas_forecast_total:,.0f}")
                                
                                # Calculate EUR for gas
                                total_historical_gas = production_data['GASPROD_MCF'].sum()
                                gas_eur = total_historical_gas + gas_forecast_total
                                st.metric("Gas EUR (MCF)", f"{gas_eur:,.0f}")
                            
                            # Show forecast parameters used
                            with st.expander("Forecast Parameters", expanded=False):
                                param_col1, param_col2 = st.columns(2)
                                
                                with param_col1:
                                    st.markdown("#### Oil Parameters")
                                    oil_params = {
                                        "Initial Rate (Qi)": forecast_df.loc[0, "OilFcst_BBL"] if "OilFcst_BBL" in forecast_df.columns else "N/A",
                                        "Decline Type": oil_values.get("OIL_DECLINE_TYPE", "EXP"),
                                        "Decline Rate": oil_values.get("OIL_USER_DECLINE", oil_values.get("OIL_EMPIRICAL_DECLINE", "N/A")),
                                        "B Factor": oil_values.get("OIL_USER_B_FACTOR", oil_values.get("OIL_CALC_B_FACTOR", "N/A")),
                                        "Terminal Decline": oil_values.get("OIL_D_MIN", "N/A"),
                                        "Minimum Rate": oil_values.get("OIL_Q_MIN", "N/A")
                                    }
                                    
                                    for param, value in oil_params.items():
                                        st.markdown(f"**{param}:** {value}")
                                
                                with param_col2:
                                    st.markdown("#### Gas Parameters")
                                    gas_params = {
                                        "Initial Rate (Qi)": forecast_df.loc[0, "GasFcst_MCF"] if "GasFcst_MCF" in forecast_df.columns else "N/A",
                                        "Decline Type": gas_values.get("GAS_DECLINE_TYPE", "EXP"),
                                        "Decline Rate": gas_values.get("GAS_USER_DECLINE", gas_values.get("GAS_EMPIRICAL_DECLINE", "N/A")),
                                        "B Factor": gas_values.get("GAS_USER_B_FACTOR", gas_values.get("GAS_CALC_B_FACTOR", "N/A")),
                                        "Terminal Decline": gas_values.get("GAS_D_MIN", "N/A"),
                                        "Minimum Rate": gas_values.get("GAS_Q_MIN", "N/A")
                                    }
                                    
                                    for param, value in gas_params.items():
                                        st.markdown(f"**{param}:** {value}")
                            
                            # Add a download button for the forecast data
                            st.markdown("### Download Forecast")
                            csv = forecast_df.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="{selected_key}_forecast.csv">Download Forecast CSV</a>'
                            st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("No data available or primary key not found in table")
        
##########################################
# SECTION 11: BULK DECLINE CALCULATION
##########################################
with st.expander("Bulk Decline Calculation", expanded=False):
    st.subheader("Calculate Decline Rates for Multiple Wells")
    
    # Decline Fit Constants inside this section
    st.subheader("Decline Fit Constants")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state['months_for_calc'] = st.slider(
            "Months for Decline Calculation", 
            6, 60, st.session_state['months_for_calc'],
            help="Number of most recent months of production data to use for decline calculation"
        )
        st.session_state['default_decline'] = st.number_input(
            "Default Decline Rate", 
            value=st.session_state['default_decline'], 
            format="%.6f",
            help="Fallback decline rate when insufficient data is available"
        )
    with col2:
        st.session_state['min_decline'] = st.number_input(
            "Minimum Decline Rate", 
            value=st.session_state['min_decline'], 
            format="%.6f",
            help="Minimum allowed decline rate"
        )
        st.session_state['max_decline'] = st.number_input(
            "Maximum Decline Rate", 
            value=st.session_state['max_decline'], 
            format="%.6f",
            help="Maximum allowed decline rate"
        )
    
    st.subheader("Additional Calculations")
    additional_calcs = st.multiselect(
        "Select calculations to perform",
        ["Decline Rates", "Last Production Dates", "Initial Rate (Qi) Values"],
        default=["Decline Rates", "Last Production Dates", "Initial Rate (Qi) Values"],
        help="Choose which values to calculate for each well"
    )
    
    if not data.empty:
        calc_all = st.button("Calculate Selected Values for All Filtered Records")
        
        if calc_all:
            progress_bar = st.progress(0)
            status_text = st.empty()
            calculation_results = []
            
            # Store calculated values for each well
            calc_values = {}
            
            for i, (index, row) in enumerate(data.iterrows()):
                api_uwi = row[primary_key]
                status_text.text(f"Processing {i+1}/{len(data)}: {api_uwi}")
                
                # Initialize result dictionary with API_UWI
                result = {primary_key: api_uwi}
                
                # Initialize storage for this well's calculated values
                calc_values[api_uwi] = {}
                
                # Calculate decline rates if selected
                if "Decline Rates" in additional_calcs:
                    # Fetch production data
                    production_data = get_production_data(api_uwi)
                    
                    if not production_data.empty:
                        # Calculate decline rates
                        oil_decline, gas_decline = calculate_decline_fit(
                            production_data, st.session_state['months_for_calc'], 
                            st.session_state['default_decline'], 
                            st.session_state['min_decline'], 
                            st.session_state['max_decline']
                        )
                        
                        # Store in calculation results and session state
                        result["Oil Decline Rate"] = oil_decline
                        result["Gas Decline Rate"] = gas_decline
                        result["Production Records"] = len(production_data)
                        
                        # Store in session state
                        st.session_state['calculated_declines'][api_uwi] = (oil_decline, gas_decline)
                        
                        # Store in our calculation values dictionary
                        calc_values[api_uwi]["OIL_EMPIRICAL_DECLINE"] = oil_decline
                        calc_values[api_uwi]["GAS_EMPIRICAL_DECLINE"] = gas_decline
                    else:
                        result["Oil Decline Rate"] = st.session_state['default_decline']
                        result["Gas Decline Rate"] = st.session_state['default_decline']
                        result["Production Records"] = 0
                        
                        # Store default values
                        calc_values[api_uwi]["OIL_EMPIRICAL_DECLINE"] = st.session_state['default_decline']
                        calc_values[api_uwi]["GAS_EMPIRICAL_DECLINE"] = st.session_state['default_decline']
                
                # Calculate Last Production Dates if selected
                if "Last Production Dates" in additional_calcs:
                    last_oil_date, last_gas_date = get_last_production_dates(api_uwi)
                    
                    # Add to result dictionary for display
                    result["Last Oil Date"] = last_oil_date
                    result["Last Gas Date"] = last_gas_date
                    
                    # Determine the most recent production date (greater of the two)
                    most_recent_date = None
                    if last_oil_date is not None and last_gas_date is not None:
                        most_recent_date = max(last_oil_date, last_gas_date)
                    elif last_oil_date is not None:
                        most_recent_date = last_oil_date
                    elif last_gas_date is not None:
                        most_recent_date = last_gas_date
                    
                    # Add most recent date to result dictionary for display
                    result["Last Production Date"] = most_recent_date
                    
                    # Store individual dates for database update
                    if last_oil_date is not None:
                        calc_values[api_uwi]["LAST_OIL_DATE"] = last_oil_date
                    if last_gas_date is not None:
                        calc_values[api_uwi]["LAST_GAS_DATE"] = last_gas_date
                    # Store the most recent date for LAST_PROD_DATE
                    if most_recent_date is not None:
                        calc_values[api_uwi]["LAST_PROD_DATE"] = most_recent_date
                
                # Calculate Initial Rate (Qi) Values if selected
                if "Initial Rate (Qi) Values" in additional_calcs:
                    oil_qi, gas_qi = calculate_qi_values(api_uwi)
                    
                    # Add to result dictionary for display
                    result["Oil Initial Rate (Qi)"] = oil_qi
                    result["Gas Initial Rate (Qi)"] = gas_qi
                    
                    # Store for database update
                    if oil_qi > 0:
                        calc_values[api_uwi]["OIL_CALC_QI"] = oil_qi
                    if gas_qi > 0:
                        calc_values[api_uwi]["GAS_CALC_QI"] = gas_qi
                
                # Add the result to our list
                calculation_results.append(result)
                
                # Update progress
                progress_bar.progress((i + 1) / len(data))
            
            status_text.text("Calculation complete!")
            st.success(f"Calculated values for {len(calculation_results)} wells")
            
            # Store calculation values in session state for future use
            st.session_state['calc_values'] = calc_values
            
            # Display results in a dataframe
            result_fields = [primary_key]
            if "Decline Rates" in additional_calcs:
                result_fields.extend(["Oil Decline Rate", "Gas Decline Rate", "Production Records"])
            if "Last Production Dates" in additional_calcs:
                result_fields.extend(["Last Oil Date", "Last Gas Date", "Last Production Date"])
            if "Initial Rate (Qi) Values" in additional_calcs:
                result_fields.extend(["Oil Initial Rate (Qi)", "Gas Initial Rate (Qi)"])
                
            # Create dataframe with only the calculated fields
            display_fields = [field for field in result_fields if any(field in result for result in calculation_results)]
            decline_df = pd.DataFrame(calculation_results)[display_fields]
            st.dataframe(decline_df)
            
            # Option to update database with calculated values
            if st.button("Apply Calculated Rates to Database"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                update_errors = []
                success_count = 0
                
                for i, api_uwi in enumerate(calc_values.keys()):
                    status_text.text(f"Updating {i+1}/{len(calc_values)}: {api_uwi}")
                    
                    # Get the calculated values for this well
                    well_values = calc_values[api_uwi]
                    
                    # Skip if no values to update
                    if not well_values:
                        continue
                    
                    # Call the update function
                    success, message = update_database_record(
                        table_name, 
                        primary_key, 
                        api_uwi,
                        well_values, 
                        table_columns
                    )
                    
                    if success:
                        success_count += 1
                    else:
                        update_errors.append(f"Error updating record {api_uwi}: {message}")
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(calc_values))
                
                if update_errors:
                    st.error("Some updates failed:\n" + "\n".join(update_errors))
                
                status_text.text(f"Update complete! Successfully updated {success_count} records.")
                if success_count > 0:
                    st.success(f"Successfully updated {success_count} records with calculated values")
                    st.cache_data.clear()
    else:
        st.warning("No data available for bulk calculation")