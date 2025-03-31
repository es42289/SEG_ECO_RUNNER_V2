import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO
import base64
from datetime import date
from dateutil.relativedelta import relativedelta

#############################################
# SECTION 1: APP CONFIGURATION
#############################################

# Set Streamlit Page Configuration with wider layout
st.set_page_config(
    page_title="Fast Edit Wells",
    layout="wide",
    initial_sidebar_state="expanded"  # Start with sidebar expanded to show well list
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
    
    /* Custom CSS for the selected well in the list */
    .selected-well {
        background-color: #e6f3ff;
        padding: 5px;
        border-radius: 5px;
        border-left: 3px solid #1e90ff;
        margin-bottom: 4px;
    }
    
    /* Custom CSS for non-selected wells */
    .well-item {
        padding: 5px;
        border-radius: 5px;
        margin-bottom: 4px;
    }
    
    /* Custom CSS for processed wells */
    .processed-well {
        color: #8f8f8f;
    }
    
    /* Section headers */
    .section-header {
        background-color: #f5f5f5;
        padding: 5px 10px;
        border-radius: 5px;
        margin-top: 10px;
        margin-bottom: 10px;
        font-weight: bold;
    }
    
    /* Sidebar well list scrollable container */
    .well-list-container {
        max-height: 400px;
        overflow-y: auto;
        margin-bottom: 20px;
        border: 1px solid #f0f0f0;
        border-radius: 5px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

#############################################
# SECTION 2: SESSION STATE INITIALIZATION
#############################################

# Initialize session state variables
if 'selected_well' not in st.session_state:
    st.session_state['selected_well'] = None
if 'forecast_data' not in st.session_state:
    st.session_state['forecast_data'] = {}
if 'wells_processed' not in st.session_state:
    st.session_state['wells_processed'] = []
if 'forecast_updating' not in st.session_state:
    st.session_state['forecast_updating'] = False
if 'forecast_update_success' not in st.session_state:
    st.session_state['forecast_update_success'] = False
if 'update_timestamp' not in st.session_state:
    st.session_state['update_timestamp'] = pd.Timestamp.now()
if 'wells_data' not in st.session_state:
    st.session_state['wells_data'] = None

#############################################
# SECTION 3: DATABASE CONNECTION FUNCTIONS
#############################################

@st.cache_resource
def get_session():
    """Get the current Snowpark session"""
    try:
        from snowflake.snowpark.context import get_active_session
        return get_active_session()
    except Exception as e:
        st.error(f"Error getting Snowflake session: {e}")
        return None

@st.cache_data(ttl=60)
def get_fast_edit_wells():
    """Get all wells with FAST_EDIT=1"""
    try:
        from snowflake.snowpark.context import get_active_session
        session = get_active_session()
        
        # Join ECON_INPUT with vw_well to get well names and trajectory
        query = """
        SELECT e.*, w.WELLNAME, w.TRAJECTORY
        FROM wells.minerals.ECON_INPUT e
        JOIN wells.minerals.vw_well_input w
        ON e.API_UWI = w.API_UWI
        WHERE e.FAST_EDIT = 1
        ORDER BY w.WELLNAME
        """
        
        result = session.sql(query).to_pandas()
        return result
    except Exception as e:
        st.error(f"Error fetching fast edit wells: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_production_data(api_uwi):
    """Get production data for a well"""
    try:
        from snowflake.snowpark.context import get_active_session
        session = get_active_session()
        
        query = f"""
        SELECT API_UWI, ProducingMonth, LIQUIDSPROD_BBL, GASPROD_MCF
        FROM wells.minerals.raw_prod_data
        WHERE API_UWI = '{api_uwi}'
        ORDER BY ProducingMonth
        """
        
        result = session.sql(query).to_pandas()
        return result
    except Exception as e:
        st.error(f"Error fetching production data: {e}")
        return pd.DataFrame()
    
def update_well_parameters(api_uwi, update_values):
    """Update well parameters in database"""
    try:
        from snowflake.snowpark.context import get_active_session
        session = get_active_session()
        
        # Handle NULL values in update statement
        set_clauses = []
        for col, value in update_values.items():
            if value == '' or value is None:
                set_clauses.append(f"{col} = NULL")
            else:
                set_clauses.append(f"{col} = '{value}'")
        
        set_clause = ", ".join(set_clauses)
        
        sql = f"UPDATE wells.minerals.ECON_INPUT SET {set_clause} WHERE API_UWI = '{api_uwi}'"
        session.sql(sql).collect()
        
        # Clear the cache to force a refresh of the wells data
        get_fast_edit_wells.clear()
        
        return True, "Well updated successfully"
    except Exception as e:
        return False, f"Error updating well: {e}"

#############################################
# SECTION 4: HELPER FUNCTIONS
#############################################

# Function to validate and fix forecast parameters
def fix_forecast_data_issues(well_row):
    """Fix common issues with forecast parameters"""
    fixed_row = well_row.copy()
    
    # Fix decline types (ensure they are E or H)
    if pd.isna(fixed_row['OIL_DECLINE_TYPE']) or fixed_row['OIL_DECLINE_TYPE'] not in ['E', 'H', 'EXP', 'HYP']:
        fixed_row['OIL_DECLINE_TYPE'] = 'E'  # Default to exponential
    elif fixed_row['OIL_DECLINE_TYPE'] == 'EXP':
        fixed_row['OIL_DECLINE_TYPE'] = 'E'
    elif fixed_row['OIL_DECLINE_TYPE'] == 'HYP':
        fixed_row['OIL_DECLINE_TYPE'] = 'H'
    
    if pd.isna(fixed_row['GAS_DECLINE_TYPE']) or fixed_row['GAS_DECLINE_TYPE'] not in ['E', 'H', 'EXP', 'HYP']:
        fixed_row['GAS_DECLINE_TYPE'] = 'E'  # Default to exponential
    elif fixed_row['GAS_DECLINE_TYPE'] == 'EXP':
        fixed_row['GAS_DECLINE_TYPE'] = 'E'
    elif fixed_row['GAS_DECLINE_TYPE'] == 'HYP':
        fixed_row['GAS_DECLINE_TYPE'] = 'H'
    
    # Ensure decline rates are valid
    if pd.isna(fixed_row['OIL_USER_DECLINE']) and pd.isna(fixed_row['OIL_EMPIRICAL_DECLINE']):
        fixed_row['OIL_EMPIRICAL_DECLINE'] = 0.06  # Default value
    
    if pd.isna(fixed_row['GAS_USER_DECLINE']) and pd.isna(fixed_row['GAS_EMPIRICAL_DECLINE']):
        fixed_row['GAS_EMPIRICAL_DECLINE'] = 0.06  # Default value
    
    # Ensure b-factors are valid for hyperbolic decline
    if fixed_row['OIL_DECLINE_TYPE'] == 'H':
        if pd.isna(fixed_row['OIL_USER_B_FACTOR']) and pd.isna(fixed_row['OIL_CALC_B_FACTOR']):
            fixed_row['OIL_CALC_B_FACTOR'] = 1.0  # Default value
    
    if fixed_row['GAS_DECLINE_TYPE'] == 'H':
        if pd.isna(fixed_row['GAS_USER_B_FACTOR']) and pd.isna(fixed_row['GAS_CALC_B_FACTOR']):
            fixed_row['GAS_CALC_B_FACTOR'] = 1.0  # Default value
    
    # Ensure minimum values are valid
    if pd.isna(fixed_row['OIL_Q_MIN']) or fixed_row['OIL_Q_MIN'] <= 0:
        fixed_row['OIL_Q_MIN'] = 1.0  # Default value
    
    if pd.isna(fixed_row['GAS_Q_MIN']) or fixed_row['GAS_Q_MIN'] <= 0:
        fixed_row['GAS_Q_MIN'] = 10.0  # Default value
    
    if pd.isna(fixed_row['OIL_D_MIN']) or fixed_row['OIL_D_MIN'] <= 0:
        fixed_row['OIL_D_MIN'] = 0.06  # Default value
    
    if pd.isna(fixed_row['GAS_D_MIN']) or fixed_row['GAS_D_MIN'] <= 0:
        fixed_row['GAS_D_MIN'] = 0.06  # Default value
    
    return fixed_row

# Function to get the next well in the list
def get_next_well(current_api, wells_data):
    
    # Get the list of active well APIs
    active_well_apis = wells_data['API_UWI'].tolist()

    # Find current index in the active wells list
    current_idx = active_well_apis.index(current_api)
    # Get next well index, wrapping around to the beginning if needed
    next_idx = (current_idx + 1) % len(active_well_apis)
    # Return the next well API
    return active_well_apis[next_idx]

# Function to refresh the wells data from database
def refresh_wells_data():
    """Refresh the wells data from the database"""
    fresh_data = get_fast_edit_wells()
    st.session_state['wells_data'] = fresh_data
    return fresh_data

# Function to get last production dates for oil and gas
def get_last_production_dates(api_uwi, production_data):
    """Get the last production dates for oil and gas"""
    if production_data.empty:
        return None, None
    
    try:
        # Ensure LIQUIDSPROD_BBL and GASPROD_MCF are numeric
        production_data['LIQUIDSPROD_BBL'] = pd.to_numeric(production_data['LIQUIDSPROD_BBL'], errors='coerce').fillna(0)
        production_data['GASPROD_MCF'] = pd.to_numeric(production_data['GASPROD_MCF'], errors='coerce').fillna(0)
        
        # Find the last non-zero oil production
        non_zero_oil = production_data[production_data['LIQUIDSPROD_BBL'] > 0]
        last_oil_date = None
        if not non_zero_oil.empty:
            # Sort by date to get the most recent
            non_zero_oil = non_zero_oil.sort_values('PRODUCINGMONTH')
            last_oil_date = non_zero_oil['PRODUCINGMONTH'].iloc[-1]
        
        # Find the last non-zero gas production
        non_zero_gas = production_data[production_data['GASPROD_MCF'] > 0]
        last_gas_date = None
        if not non_zero_gas.empty:
            # Sort by date to get the most recent
            non_zero_gas = non_zero_gas.sort_values('PRODUCINGMONTH')
            last_gas_date = non_zero_gas['PRODUCINGMONTH'].iloc[-1]
        
        return last_oil_date, last_gas_date
    except Exception as e:
        print(f"Error in get_last_production_dates: {e}")
        return None, None

#############################################
# SECTION 5: FORECAST FUNCTIONS
#############################################

# Updated SingleWellForecast function with custom forecast start dates
def SingleWellForecast(well_api, econ_input_df, raw_prod_data_df):
    """
    Generates a production forecast for a single well using either exponential or 
    hyperbolic decline based on decline type specified. Uses FCST_START_OIL and 
    FCST_START_GAS dates to determine when to start the respective forecasts.
    
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
        # Get the last production date
        last_prod_date = well_production['PRODUCINGMONTH'].max()
        
        # Get forecast start dates from ECON_INPUT
        oil_fcst_start = pd.to_datetime(well_row['FCST_START_OIL']) if pd.notna(well_row['FCST_START_OIL']) else None
        gas_fcst_start = pd.to_datetime(well_row['FCST_START_GAS']) if pd.notna(well_row['FCST_START_GAS']) else None
        
        # If forecast start dates are not provided, use last production date + 1 month as default
        default_fcst_start = (pd.to_datetime(last_prod_date) + pd.DateOffset(months=1)).replace(day=1)
        
        # Use the specified dates or the default if not specified
        oil_fcst_start = oil_fcst_start if oil_fcst_start is not None else default_fcst_start
        gas_fcst_start = gas_fcst_start if gas_fcst_start is not None else default_fcst_start
        
        # Ensure forecast start dates are on the first day of the month
        oil_fcst_start = oil_fcst_start.replace(day=1)
        gas_fcst_start = gas_fcst_start.replace(day=1)
        
        # Find the earliest forecast start date to initialize the forecast period
        earliest_fcst_start = min(oil_fcst_start, gas_fcst_start)
        
        # Initialize forecast period - use longer period to ensure we capture all forecast data
        max_months = 600
        dates = pd.date_range(start=earliest_fcst_start, periods=max_months, freq='MS')
        
        # Initialize an empty DataFrame for this well's forecasts
        well_fcst = pd.DataFrame({'PRODUCINGMONTH': dates})
        has_forecast = False
        
        # Generate gas forecast
        try:
            # Determine gas initial rate (qi) - use user input if available, otherwise calculated
            gas_qi = float(well_row['GAS_USER_QI']) if pd.notna(well_row['GAS_USER_QI']) else float(well_row['GAS_CALC_QI'])
            gas_qf = float(well_row['GAS_Q_MIN'])
            
            if gas_qi > gas_qf:
                # Use the gas forecast start date to determine the reference production data
                last_gas_prod = well_production[well_production['PRODUCINGMONTH'] <= gas_fcst_start]['GASPROD_MCF'].tail(3)
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
                    # Calculate the index offset for gas forecast start
                    gas_offset = (gas_fcst_start - earliest_fcst_start).days // 30
                    gas_offset = max(0, gas_offset)  # Ensure non-negative
                    
                    # Limit to available array length
                    data_length = min(len(gas_rates), len(well_fcst) - gas_offset)
                    
                    # Insert gas forecast at the correct position
                    well_fcst.loc[gas_offset:gas_offset+data_length-1, 'GasFcst_MCF'] = gas_rates[:data_length]
                    well_fcst.loc[gas_offset:gas_offset+data_length-1, 'Gas_Decline_Type'] = gas_decline_types[:data_length]
                    
                    # Add gas forecast start date for reference
                    well_fcst['GasFcst_Start_Date'] = gas_fcst_start
                    
                    has_forecast = True
                    
        except Exception as e:
            print(f"Error processing gas forecast for well {well_api}: {str(e)}")
        
        # Generate oil forecast
        try:
            # Determine oil initial rate (qi) - use user input if available, otherwise calculated
            oil_qi = float(well_row['OIL_USER_QI']) if pd.notna(well_row['OIL_USER_QI']) else float(well_row['OIL_CALC_QI'])
            oil_qf = float(well_row['OIL_Q_MIN'])
            
            if oil_qi > oil_qf:
                # Use the oil forecast start date to determine the reference production data
                last_oil_prod = well_production[well_production['PRODUCINGMONTH'] <= oil_fcst_start]['LIQUIDSPROD_BBL'].tail(3)
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
                    # Calculate the index offset for oil forecast start
                    oil_offset = (oil_fcst_start - earliest_fcst_start).days // 30
                    oil_offset = max(0, oil_offset)  # Ensure non-negative
                    
                    # Limit to available array length
                    data_length = min(len(oil_rates), len(well_fcst) - oil_offset)
                    
                    # Insert oil forecast at the correct position
                    well_fcst.loc[oil_offset:oil_offset+data_length-1, 'OilFcst_BBL'] = oil_rates[:data_length]
                    well_fcst.loc[oil_offset:oil_offset+data_length-1, 'Oil_Decline_Type'] = oil_decline_types[:data_length]
                    
                    # Add oil forecast start date for reference
                    well_fcst['OilFcst_Start_Date'] = oil_fcst_start
                    
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
        # Handle potential duplicate dates between historical and forecast data
        final_df = pd.merge(aggregated_data, well_fcst, on='PRODUCINGMONTH', how='outer')
    else:
        final_df = aggregated_data.copy()
        for col in ['OilFcst_BBL', 'GasFcst_MCF']:
            final_df[col] = np.nan
    
    # Sort by date for cumulative calculations
    final_df = final_df.sort_values('PRODUCINGMONTH')
    
    # Calculate forecast cumulatives correctly handling custom forecast start dates
    
    # For gas forecast cumulatives
    if 'GasFcst_MCF' in final_df.columns and final_df['GasFcst_MCF'].notna().any():
        # Find the gas forecast start date
        gas_fcst_start_date = final_df['GasFcst_Start_Date'].iloc[0] if 'GasFcst_Start_Date' in final_df.columns else None
        
        if gas_fcst_start_date is not None:
            # Get the last actual production before forecast start
            gas_hist_before_fcst = final_df[final_df['PRODUCINGMONTH'] < gas_fcst_start_date]
            last_cum_gas = 0 if gas_hist_before_fcst.empty else gas_hist_before_fcst['CumGas_MCF'].iloc[-1]
            
            # Calculate cumulative from forecast start date
            fcst_mask = final_df['PRODUCINGMONTH'] >= gas_fcst_start_date
            final_df.loc[fcst_mask, 'GasFcstCum_MCF'] = final_df.loc[fcst_mask, 'GasFcst_MCF'].fillna(0).cumsum() + last_cum_gas
        else:
            # Fallback to previous method if forecast start date isn't available
            gas_idx = final_df['CumGas_MCF'].last_valid_index()
            if gas_idx is not None:
                last_cum_gas = final_df.loc[gas_idx, 'CumGas_MCF']
                mask = final_df.index > gas_idx
                final_df.loc[mask, 'GasFcstCum_MCF'] = last_cum_gas + final_df.loc[mask, 'GasFcst_MCF'].fillna(0).cumsum()
    
    # For oil forecast cumulatives
    if 'OilFcst_BBL' in final_df.columns and final_df['OilFcst_BBL'].notna().any():
        # Find the oil forecast start date
        oil_fcst_start_date = final_df['OilFcst_Start_Date'].iloc[0] if 'OilFcst_Start_Date' in final_df.columns else None
        
        if oil_fcst_start_date is not None:
            # Get the last actual production before forecast start
            oil_hist_before_fcst = final_df[final_df['PRODUCINGMONTH'] < oil_fcst_start_date]
            last_cum_oil = 0 if oil_hist_before_fcst.empty else oil_hist_before_fcst['CumLiquids_BBL'].iloc[-1]
            
            # Calculate cumulative from forecast start date
            fcst_mask = final_df['PRODUCINGMONTH'] >= oil_fcst_start_date
            final_df.loc[fcst_mask, 'OilFcstCum_BBL'] = final_df.loc[fcst_mask, 'OilFcst_BBL'].fillna(0).cumsum() + last_cum_oil
        else:
            # Fallback to previous method if forecast start date isn't available
            oil_idx = final_df['CumLiquids_BBL'].last_valid_index()
            if oil_idx is not None:
                last_cum_oil = final_df.loc[oil_idx, 'CumLiquids_BBL']
                mask = final_df.index > oil_idx
                final_df.loc[mask, 'OilFcstCum_BBL'] = last_cum_oil + final_df.loc[mask, 'OilFcst_BBL'].fillna(0).cumsum()
    
    # Create blended columns - properly handling custom forecast start dates
    
    # For oil blend, use historical data before FCST_START_OIL and forecast data after
    oil_fcst_start_date = final_df['OilFcst_Start_Date'].iloc[0] if 'OilFcst_Start_Date' in final_df.columns else None
    if oil_fcst_start_date is not None:
        # Use historical production before forecast start, and forecast after
        final_df['Oil_Blend'] = np.nan
        before_fcst_mask = final_df['PRODUCINGMONTH'] < oil_fcst_start_date
        after_fcst_mask = final_df['PRODUCINGMONTH'] >= oil_fcst_start_date
        
        final_df.loc[before_fcst_mask, 'Oil_Blend'] = final_df.loc[before_fcst_mask, 'LIQUIDSPROD_BBL']
        final_df.loc[after_fcst_mask, 'Oil_Blend'] = final_df.loc[after_fcst_mask, 'OilFcst_BBL']
    else:
        # Fallback to the original blend approach
        final_df['Oil_Blend'] = final_df['LIQUIDSPROD_BBL'].fillna(final_df['OilFcst_BBL'])
    
    # For gas blend, use historical data before FCST_START_GAS and forecast data after
    gas_fcst_start_date = final_df['GasFcst_Start_Date'].iloc[0] if 'GasFcst_Start_Date' in final_df.columns else None
    if gas_fcst_start_date is not None:
        # Use historical production before forecast start, and forecast after
        final_df['Gas_Blend'] = np.nan
        before_fcst_mask = final_df['PRODUCINGMONTH'] < gas_fcst_start_date
        after_fcst_mask = final_df['PRODUCINGMONTH'] >= gas_fcst_start_date
        
        final_df.loc[before_fcst_mask, 'Gas_Blend'] = final_df.loc[before_fcst_mask, 'GASPROD_MCF']
        final_df.loc[after_fcst_mask, 'Gas_Blend'] = final_df.loc[after_fcst_mask, 'GasFcst_MCF']
    else:
        # Fallback to the original blend approach
        final_df['Gas_Blend'] = final_df['GASPROD_MCF'].fillna(final_df['GasFcst_MCF'])
    
    # Replace zeros with NaN for numeric columns
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
    final_df[numeric_cols] = final_df[numeric_cols].replace(0, np.nan)
    
    # Filter to end of 2050
    final_df = final_df[final_df['PRODUCINGMONTH'] <= '2050-12-31']
    
    # Add End of Month date column
    final_df['EOM_Date'] = final_df['PRODUCINGMONTH'].dt.to_period('M').dt.to_timestamp('M')
    
    # Set column order
    col_order = ['PRODUCINGMONTH', 'EOM_Date',
                'LIQUIDSPROD_BBL', 'GASPROD_MCF', 'WATERPROD_BBL',
                'CumLiquids_BBL', 'CumGas_MCF', 'CumWater_BBL',
                'OilFcst_BBL', 'GasFcst_MCF', 'OilFcstCum_BBL', 'GasFcstCum_MCF',
                'Oil_Blend', 'Gas_Blend',
                'Oil_Decline_Type', 'Gas_Decline_Type',
                'OilFcst_Start_Date', 'GasFcst_Start_Date']
    
    # Ensure all columns exist and add any additional columns at the end
    existing_cols = [col for col in col_order if col in final_df.columns]
    additional_cols = [col for col in final_df.columns if col not in col_order]
    final_df = final_df[existing_cols + additional_cols]
    
    return final_df

#############################################
# SECTION 6: MAIN APP INTERFACE
#############################################

def generate_forecast(api_uwi, well_data, production_data):
    """Generate forecast for a well"""
    try:
        # Get the well row
        filtered_data = well_data[well_data['API_UWI'] == api_uwi]
        
        if filtered_data.empty:
            st.error(f"Well API {api_uwi} not found in the dataset")
            return None
            
        well_row = filtered_data.iloc[0]
        
        # Fix common data issues
        fixed_well_row = fix_forecast_data_issues(well_row)
        
        # Create a temporary dataframe with just this well's data
        temp_data = pd.DataFrame([fixed_well_row])
        
        # Call the SingleWellForecast function
        forecast_df = SingleWellForecast(
            api_uwi,  # API_UWI of selected well
            temp_data,  # ECON_INPUT data with fixes
            production_data  # Production data for the well
        )
        
        return forecast_df
    except Exception as e:
        st.error(f"Error generating forecast: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# Load the data if not already in session state
if st.session_state['wells_data'] is None:
    wells_data = get_fast_edit_wells()
    st.session_state['wells_data'] = wells_data
else:
    wells_data = st.session_state['wells_data']

# Main layout title
st.title("Fast Edit Wells")

#############################################
# SECTION 7: SIDEBAR
#############################################

st.sidebar.title("Fast Edit Wells")
st.sidebar.header("Wells for Fast Edit")

# Display number of wells in sidebar
st.sidebar.subheader(f"{len(wells_data)} Wells to Process")

# Show progress in sidebar
if 'wells_processed' in st.session_state:
    processed_count = len(st.session_state['wells_processed'])
    if len(wells_data) > 0:
        progress = processed_count / len(wells_data) if len(wells_data) > 0 else 0
        st.sidebar.progress(progress)
        st.sidebar.text(f"Processed: {processed_count}/{len(wells_data)} ({progress*100:.1f}%)")

# Refresh wells list button
if st.sidebar.button("Refresh Wells List"):
    wells_data = refresh_wells_data()
    # Force rerun to update UI
    st.rerun()

# Well list with selection via selectbox
st.sidebar.subheader("Select a Well")

# Create list of well options with formatting for selectbox
well_options = {}
for _, well in wells_data.iterrows():
    api = well['API_UWI']
    well_name = well['WELLNAME']
    trajectory = well['TRAJECTORY'] if not pd.isna(well['TRAJECTORY']) else "Unknown"
    is_processed = api in st.session_state['wells_processed']
    processed_mark = "✓ " if is_processed else ""
    well_options[api] = f"{processed_mark}{well_name} ({api}) | {trajectory}"

# Convert to list for selectbox
well_apis = list(well_options.keys())
well_display_names = list(well_options.values())

# Use selectbox for well selection if wells exist
if well_apis:  # Only show if there are wells
    selected_index = 0
    if st.session_state['selected_well'] in well_apis:
        selected_index = well_apis.index(st.session_state['selected_well'])
    elif st.session_state['selected_well'] is not None and well_apis:
        # If current selected well is not in the active list but we have other wells
        st.session_state['selected_well'] = well_apis[0]  # Reset to first available well
        selected_index = 0
        
    if well_display_names:  # Make sure we have display names
        selected_display_name = st.sidebar.selectbox(
            "Choose a well:",
            well_display_names,
            index=selected_index,
            label_visibility="collapsed"
        )
        
        # Get the API from the selected display name
        selected_index = well_display_names.index(selected_display_name)
        selected_api = well_apis[selected_index]
        
        # Update the session state if changed
        if selected_api != st.session_state['selected_well']:
            st.session_state['selected_well'] = selected_api
            st.rerun()
else:
    st.sidebar.info("No wells available for editing")
    st.session_state['selected_well'] = None  # Clear selected well if no wells available
    
# Display list of Fast Edit wells in sidebar with active one highlighted
st.sidebar.subheader("Wells List")
st.sidebar.markdown('<div class="well-list-container">', unsafe_allow_html=True)

# Sort wells by name for the list
sorted_wells = wells_data.sort_values('WELLNAME')

# Display each well with active one highlighted
for _, well in sorted_wells.iterrows():
    api = well['API_UWI']
    well_name = well['WELLNAME']
    trajectory = well['TRAJECTORY'] if not pd.isna(well['TRAJECTORY']) else "Unknown"
    is_active = (api == st.session_state['selected_well'])
    is_processed = api in st.session_state['wells_processed']
    
    # Determine CSS classes
    css_class = "selected-well" if is_active else "well-item"
    if is_processed:
        css_class += " processed-well"
    
    # Display the well with appropriate styling
    status_mark = "✓ " if is_processed else ""
    
    st.sidebar.markdown(
        f'<div class="{css_class}">{status_mark}{well_name} ({api})<br/>{trajectory}</div>',
        unsafe_allow_html=True
    )

st.sidebar.markdown('</div>', unsafe_allow_html=True)

#############################################
# SECTION 8: MAIN CONTENT
#############################################

# Create two columns for main content layout: plot on left, inputs on right
if 'selected_well' in st.session_state and st.session_state['selected_well']:
    api_uwi = st.session_state['selected_well']
    
    # Check if the well exists in the dataset
    well_exists = False
    well_name = ""
    well_row = None
    
    filtered_wells = wells_data[wells_data['API_UWI'] == api_uwi]
    if not filtered_wells.empty:
        well_exists = True
        well_row = filtered_wells.iloc[0]
        well_name = well_row['WELLNAME']
    
    if not well_exists:
        st.warning(f"Well API {api_uwi} not found in the dataset. It may have been removed. Please select another well.")
        # Refresh data and redirect to first available well
        fresh_wells = refresh_wells_data()
        if not fresh_wells.empty and 'API_UWI' in fresh_wells.columns:
            if not fresh_wells.empty:
                st.session_state['selected_well'] = fresh_wells['API_UWI'].iloc[0]
                st.rerun()
        # If no wells available, clear selection
        st.session_state['selected_well'] = None
    else:
        st.header(f"{well_name} ({api_uwi})")
        
        # Get production data
        production_data = get_production_data(api_uwi)
        
        # Generate forecast if not already in session state
        if api_uwi not in st.session_state['forecast_data']:
            with st.spinner(f"Generating forecast for {well_name}..."):
                forecast_df = generate_forecast(api_uwi, wells_data, production_data)
                if forecast_df is not None:
                    st.session_state['forecast_data'][api_uwi] = forecast_df
        
        # Create two columns for main content: plot on left, inputs on right
        plot_col, input_col = st.columns([2, 1])
        
        #############################################
        # SECTION 9: LEFT COLUMN - VISUALIZATION
        #############################################
        
        with plot_col:
            st.markdown('<div class="section-header">Production and Forecast (10-Year View)</div>', unsafe_allow_html=True)
            
            # Check if we have both production data and forecast
            if not production_data.empty and api_uwi in st.session_state['forecast_data']:
                # Process production data
                production_data['PRODUCINGMONTH'] = pd.to_datetime(production_data['PRODUCINGMONTH'])
                
                # Filter to last 5 years of history
                end_date = production_data['PRODUCINGMONTH'].max()
                start_date = end_date - pd.DateOffset(years=5)
                filtered_prod = production_data[production_data['PRODUCINGMONTH'] >= start_date]
                
                # Handle empty filtered data
                if filtered_prod.empty:
                    filtered_prod = production_data  # Use all data if filter results in empty set
                
                # Get forecast data - this is the FULL forecast to economic limit
                forecast_df = st.session_state['forecast_data'][api_uwi]
                
                # Create a copy of the full forecast for statistics
                full_forecast_df = forecast_df.copy()
                
                # Get forecast start dates for visualization
                oil_fcst_start_date = None
                gas_fcst_start_date = None
                
                if 'OilFcst_Start_Date' in full_forecast_df.columns:
                    oil_fcst_start_values = full_forecast_df['OilFcst_Start_Date'].dropna().unique()
                    if len(oil_fcst_start_values) > 0:
                        oil_fcst_start_date = pd.to_datetime(oil_fcst_start_values[0])
                
                if 'GasFcst_Start_Date' in full_forecast_df.columns:
                    gas_fcst_start_values = full_forecast_df['GasFcst_Start_Date'].dropna().unique()
                    if len(gas_fcst_start_values) > 0:
                        gas_fcst_start_date = pd.to_datetime(gas_fcst_start_values[0])
                
                # Limit ONLY PLOT data to 10 years from last production date
                if 'PRODUCINGMONTH' in forecast_df.columns:
                    max_forecast_date = end_date + pd.DateOffset(years=10)
                    plot_forecast_df = forecast_df[forecast_df['PRODUCINGMONTH'] <= max_forecast_date]
                else:
                    plot_forecast_df = forecast_df.copy()
                
                # Create visualization dataframe
                chart_data = pd.DataFrame()
                chart_data['Production Month'] = filtered_prod['PRODUCINGMONTH']
                chart_data['Oil (BBL)'] = filtered_prod['LIQUIDSPROD_BBL']
                chart_data['Gas (MCF)'] = filtered_prod['GASPROD_MCF']
                
                # Add forecast data - only use the 10-year limited data for the plot
                forecast_chart = pd.DataFrame()
                if 'PRODUCINGMONTH' in plot_forecast_df.columns:
                    forecast_chart['Production Month'] = plot_forecast_df['PRODUCINGMONTH']
                    if 'OilFcst_BBL' in plot_forecast_df.columns:
                        forecast_chart['Oil Forecast (BBL)'] = plot_forecast_df['OilFcst_BBL']
                    if 'GasFcst_MCF' in plot_forecast_df.columns:
                        forecast_chart['Gas Forecast (MCF)'] = plot_forecast_df['GasFcst_MCF']
                
                # Melt data for chart
                melted_hist = chart_data.melt(
                    id_vars=['Production Month'],
                    value_vars=['Oil (BBL)', 'Gas (MCF)'],
                    var_name='Production Type',
                    value_name='Volume'
                )
                
                # Melt forecast data
                if not forecast_chart.empty:
                    forecast_vars = []
                    if 'Oil Forecast (BBL)' in forecast_chart.columns:
                        forecast_vars.append('Oil Forecast (BBL)')
                    if 'Gas Forecast (MCF)' in forecast_chart.columns:
                        forecast_vars.append('Gas Forecast (MCF)')
                    
                    if forecast_vars:
                        melted_forecast = forecast_chart.melt(
                            id_vars=['Production Month'],
                            value_vars=forecast_vars,
                            var_name='Production Type',
                            value_name='Volume'
                        )
                        
                        # Combine historical and forecast data
                        melted_data = pd.concat([melted_hist, melted_forecast], ignore_index=True)
                    else:
                        melted_data = melted_hist
                else:
                    melted_data = melted_hist
                
                # Filter out zero values for log scale
                melted_data = melted_data[melted_data['Volume'] > 0]
                
                # Create color mapping
                color_mapping = {
                    'Oil (BBL)': '#1e8f4e',        # Green for oil
                    'Gas (MCF)': '#d62728',         # Red for gas
                    'Oil Forecast (BBL)': '#1e8f4e',  # Same green for oil forecast
                    'Gas Forecast (MCF)': '#d62728'   # Same red for gas forecast
                }
                
                # Create chart
                chart = alt.Chart(melted_data).encode(
                    x=alt.X('Production Month:T', title='Production Month'),
                    y=alt.Y('Volume:Q', scale=alt.Scale(type='log'), title='Production Volume (Log Scale)'),
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
                
                # Create vertical rules for forecast start dates
                oil_rule = None
                gas_rule = None
                
                if oil_fcst_start_date is not None:
                    oil_rule = alt.Chart(pd.DataFrame({'date': [oil_fcst_start_date]})).mark_rule(
                        color='green',
                        strokeDash=[6, 4],
                        strokeWidth=1.5
                    ).encode(
                        x='date:T',
                        tooltip=[alt.Tooltip('date:T', title='Oil Forecast Start')]
                    )
                
                if gas_fcst_start_date is not None:
                    gas_rule = alt.Chart(pd.DataFrame({'date': [gas_fcst_start_date]})).mark_rule(
                        color='red',
                        strokeDash=[6, 4],
                        strokeWidth=1.5
                    ).encode(
                        x='date:T',
                        tooltip=[alt.Tooltip('date:T', title='Gas Forecast Start')]
                    )
                
                # Combine the charts
                final_chart = (historical_chart + forecast_chart)
                
                # Add vertical rules for forecast start dates if they exist
                if oil_rule is not None:
                    final_chart += oil_rule
                if gas_rule is not None:
                    final_chart += gas_rule
                
                # Set chart properties
                final_chart = final_chart.properties(
                    title=f"Production History and 10-Year Forecast for {well_name}",
                    height=500  # Make chart taller for better visibility
                ).interactive()
                
                # Display the chart
                st.altair_chart(final_chart, use_container_width=True)
                
                # Add statistics below the chart
                st.markdown('<div class="section-header">Production Statistics (Based on Full Economic Forecast)</div>', unsafe_allow_html=True)
                
                stat_col1, stat_col2 = st.columns(2)
                
                # Oil statistics - USE FULL FORECAST DATA for statistics
                with stat_col1:
                    st.markdown("#### Oil Statistics")
                    st.metric("Total Historical Oil (BBL)", f"{filtered_prod['LIQUIDSPROD_BBL'].sum():,.0f}")
                    
                    if 'OilFcst_BBL' in full_forecast_df.columns:
                        oil_forecast_total = full_forecast_df['OilFcst_BBL'].sum() 
                        oil_eur = filtered_prod['LIQUIDSPROD_BBL'].sum() + oil_forecast_total
                        st.metric("Forecast Oil (BBL)", f"{oil_forecast_total:,.0f}")
                        st.metric("Oil EUR (BBL)", f"{oil_eur:,.0f}")
                    
                    if oil_fcst_start_date is not None:
                        st.metric("Oil Forecast Start Date", f"{oil_fcst_start_date.strftime('%Y-%m-%d')}")
                
                # Gas statistics - USE FULL FORECAST DATA for statistics
                with stat_col2:
                    st.markdown("#### Gas Statistics")
                    st.metric("Total Historical Gas (MCF)", f"{filtered_prod['GASPROD_MCF'].sum():,.0f}")
                    
                    if 'GasFcst_MCF' in full_forecast_df.columns:
                        gas_forecast_total = full_forecast_df['GasFcst_MCF'].sum()
                        gas_eur = filtered_prod['GASPROD_MCF'].sum() + gas_forecast_total
                        st.metric("Forecast Gas (MCF)", f"{gas_forecast_total:,.0f}")
                        st.metric("Gas EUR (MCF)", f"{gas_eur:,.0f}")
                    
                    if gas_fcst_start_date is not None:
                        st.metric("Gas Forecast Start Date", f"{gas_fcst_start_date.strftime('%Y-%m-%d')}")
                
                # Add information about forecast periods
                st.info("The chart displays a 10-year forecast view for better visualization, while the statistics include the full economic forecast to Qmin.")
                
            else:
                if production_data.empty:
                    st.warning(f"No production data available for {well_name} ({api_uwi})")
                elif api_uwi not in st.session_state['forecast_data']:
                    st.info(f"Generating forecast for {well_name}...")
                    # Force rerun to generate forecast
                    st.rerun()
        
        #############################################
        # SECTION 10: RIGHT COLUMN - INPUTS
        #############################################
        
        with input_col:
            st.markdown('<div class="section-header">Update Forecast Parameters</div>', unsafe_allow_html=True)
            
            # Define a callback to update forecast when parameters change
            def update_forecast():
                # Use session state to track forecast update status
                st.session_state['forecast_updating'] = True
                st.session_state['update_timestamp'] = pd.Timestamp.now()
                
                # Create a temporary dataframe with modified well data
                temp_well_row = well_row.copy()
                
                # Update with new parameter values
                temp_well_row['OIL_USER_QI'] = st.session_state.oil_qi
                temp_well_row['OIL_USER_DECLINE'] = st.session_state.oil_decline
                temp_well_row['OIL_USER_B_FACTOR'] = st.session_state.oil_b_factor
                temp_well_row['GAS_USER_QI'] = st.session_state.gas_qi
                temp_well_row['GAS_USER_DECLINE'] = st.session_state.gas_decline
                temp_well_row['GAS_USER_B_FACTOR'] = st.session_state.gas_b_factor
                
                # Update forecast start dates
                if 'oil_fcst_start' in st.session_state:
                    temp_well_row['FCST_START_OIL'] = st.session_state.oil_fcst_start
                if 'gas_fcst_start' in st.session_state:
                    temp_well_row['FCST_START_GAS'] = st.session_state.gas_fcst_start
                
                # Create a temporary dataframe with just this well's data
                temp_data = pd.DataFrame([temp_well_row])
                
                # Generate updated forecast
                updated_forecast = generate_forecast(api_uwi, temp_data, production_data)
                
                if updated_forecast is not None:
                    st.session_state['forecast_data'][api_uwi] = updated_forecast
                    st.session_state['forecast_update_success'] = True
                else:
                    st.session_state['forecast_update_success'] = False
                    
                st.session_state['forecast_updating'] = False
        
            # Status indicator for forecast updates
            status_container = st.container()
            
            # Show status message if an update just happened
            if 'forecast_updating' in st.session_state:
                if st.session_state.get('forecast_updating', False):
                    status_container.info("Updating forecast...")
                elif 'update_timestamp' in st.session_state:
                    # Show status briefly based on success/failure
                    time_since_update = pd.Timestamp.now() - st.session_state['update_timestamp']
                    # Only show for a few script executions
                    if time_since_update.total_seconds() < 10:  # arbitrary threshold
                        if st.session_state.get('forecast_update_success', False):
                            status_container.success("Forecast updated successfully")
                        else:
                            status_container.error("Failed to update forecast")
            
            # Oil parameters section
            st.markdown('<div class="section-header">Oil Parameters</div>', unsafe_allow_html=True)
            
            # Get current values for oil parameters
            oil_qi = well_row['OIL_USER_QI'] if not pd.isna(well_row['OIL_USER_QI']) else well_row['OIL_CALC_QI']
            oil_decline = well_row['OIL_USER_DECLINE'] if not pd.isna(well_row['OIL_USER_DECLINE']) else well_row['OIL_EMPIRICAL_DECLINE']
            oil_b_factor = well_row['OIL_USER_B_FACTOR'] if not pd.isna(well_row['OIL_USER_B_FACTOR']) else well_row['OIL_CALC_B_FACTOR']
            
            # Get oil forecast start date
            oil_fcst_start = well_row['FCST_START_OIL'] if 'FCST_START_OIL' in well_row and pd.notna(well_row['FCST_START_OIL']) else None
            if oil_fcst_start is not None:
                try:
                    oil_fcst_start = pd.to_datetime(oil_fcst_start)
                except:
                    oil_fcst_start = None
            
            # Initialize session state for oil parameters if needed
            if 'oil_qi' not in st.session_state:
                st.session_state.oil_qi = float(oil_qi) if not pd.isna(oil_qi) else 0.0
            if 'oil_decline' not in st.session_state:
                st.session_state.oil_decline = float(oil_decline) if not pd.isna(oil_decline) else 0.06
            if 'oil_b_factor' not in st.session_state:
                st.session_state.oil_b_factor = float(oil_b_factor) if not pd.isna(oil_b_factor) else 1.0
            
            # Get last production dates for default forecast start dates
            last_oil_date, last_gas_date = get_last_production_dates(api_uwi, production_data)
            default_oil_fcst_start = last_oil_date + pd.DateOffset(months=1) if last_oil_date is not None else pd.Timestamp.now()
            
            # Oil parameters with keyboard arrow adjustment by 1 unit
            st.number_input("Oil User Qi", 
                        value=float(oil_qi) if not pd.isna(oil_qi) else 0.0,
                        format="%.2f",
                        key="oil_qi",
                        step=1.0,  # Set step size to 1.0 for arrow key increments/decrements
                        on_change=update_forecast)
            
            st.number_input("Oil User Decline", 
                        value=float(oil_decline) if not pd.isna(oil_decline) else 0.06, 
                        format="%.6f",
                        key="oil_decline",
                        step=0.01,  # Smaller step size for decline rate
                        on_change=update_forecast)
            
            st.number_input("Oil User B Factor", 
                        value=float(oil_b_factor) if not pd.isna(oil_b_factor) else 1.0,
                        format="%.4f",
                        key="oil_b_factor",
                        step=0.1,  # Step size for b-factor
                        on_change=update_forecast)
            
            # # Add oil forecast start date picker
            # Generate list of datetime.date objects in 1-month increments
            oil_fcst_date_slider = []
            current = start_date
            while current <= end_date + pd.DateOffset(years=5):
                oil_fcst_date_slider.append(current.date())  # convert to datetime.date
                current += relativedelta(months=1)
            # Streamlit select_slider with datetime.date output
            oil_fcst_start_date = st.select_slider(
                "Oil Forecast Start Date",
                options=oil_fcst_date_slider,
                # value=end_date.date(),
                value=oil_fcst_start.date() if oil_fcst_start is not None else default_oil_fcst_start.date(),
                key= "oil_fcst_start",
                format_func=lambda d: d.strftime("%b %Y"),
                help="Date to start oil forecast (first day of selected month)",
                on_change=update_forecast
            )

            st.markdown("---")
            
            # Gas parameters section
            st.markdown('<div class="section-header">Gas Parameters</div>', unsafe_allow_html=True)
            
            # Get current values for gas parameters
            gas_qi = well_row['GAS_USER_QI'] if not pd.isna(well_row['GAS_USER_QI']) else well_row['GAS_CALC_QI'] 
            gas_decline = well_row['GAS_USER_DECLINE'] if not pd.isna(well_row['GAS_USER_DECLINE']) else well_row['GAS_EMPIRICAL_DECLINE']
            gas_b_factor = well_row['GAS_USER_B_FACTOR'] if not pd.isna(well_row['GAS_USER_B_FACTOR']) else well_row['GAS_CALC_B_FACTOR']
            
            # Get gas forecast start date
            gas_fcst_start = well_row['FCST_START_GAS'] if 'FCST_START_GAS' in well_row and pd.notna(well_row['FCST_START_GAS']) else None
            if gas_fcst_start is not None:
                try:
                    gas_fcst_start = pd.to_datetime(gas_fcst_start)
                except:
                    gas_fcst_start = None
            
            # Default gas forecast start date if needed
            default_gas_fcst_start = last_gas_date + pd.DateOffset(months=1) if last_gas_date is not None else pd.Timestamp.now()
            
            # Initialize session state for gas parameters if needed
            if 'gas_qi' not in st.session_state:
                st.session_state.gas_qi = float(gas_qi) if not pd.isna(gas_qi) else 0.0
            if 'gas_decline' not in st.session_state:
                st.session_state.gas_decline = float(gas_decline) if not pd.isna(gas_decline) else 0.06
            if 'gas_b_factor' not in st.session_state:
                st.session_state.gas_b_factor = float(gas_b_factor) if not pd.isna(gas_b_factor) else 1.0
            
            # Gas parameters with keyboard arrow adjustment by 1 unit
            st.number_input("Gas User Qi", 
                        value=float(gas_qi) if not pd.isna(gas_qi) else 0.0,
                        format="%.2f",
                        key="gas_qi",
                        step=1.0,  # Set step size to 1.0 for arrow key increments/decrements
                        on_change=update_forecast)
            
            st.number_input("Gas User Decline", 
                        value=float(gas_decline) if not pd.isna(gas_decline) else 0.06,
                        format="%.6f",
                        key="gas_decline",
                        step=0.01,  # Smaller step size for decline rate
                        on_change=update_forecast)
            
            st.number_input("Gas User B Factor", 
                        value=float(gas_b_factor) if not pd.isna(gas_b_factor) else 1.0,
                        format="%.4f",
                        key="gas_b_factor",
                        step=0.1,  # Step size for b-factor
                        on_change=update_forecast)
            
            # Add gas forecast start date picker
            # Generate list of datetime.date objects in 1-month increments
            gas_fcst_date_slider = []
            current = start_date
            while current <= end_date + pd.DateOffset(years=5):
                gas_fcst_date_slider.append(current.date())  # convert to datetime.date
                current += relativedelta(months=1)
            # Streamlit select_slider with datetime.date output
            gas_fcst_start_date = st.select_slider(
                "Gas Forecast Start Date",
                options=gas_fcst_date_slider,
                # value=end_date.date(),
                value=gas_fcst_start.date() if gas_fcst_start is not None else default_gas_fcst_start.date(),
                key= "gas_fcst_start",
                format_func=lambda d: d.strftime("%b %Y"),
                help="Date to start gas forecast (first day of selected month)",
                on_change=update_forecast
            )
            
            st.markdown("---")
            
            # Save buttons in separate form
            st.markdown('<div class="section-header">Save Changes</div>', unsafe_allow_html=True)
            
            with st.form(key="save_form"):
                button_col1, button_col2 = st.columns(2)
                
                with button_col1:
                    remove_from_fast_edit = st.form_submit_button("Remove from Fast Edit", use_container_width=True)
                
                with button_col2:
                    save_and_next = st.form_submit_button("Save & Next Well", use_container_width=True)
                
                # Handle form submission
                if remove_from_fast_edit or save_and_next:
                    # Create update dictionary
                    update_values = {
                        'OIL_USER_QI': st.session_state.oil_qi,
                        'OIL_USER_DECLINE': st.session_state.oil_decline,
                        'OIL_USER_B_FACTOR': st.session_state.oil_b_factor,
                        'GAS_USER_QI': st.session_state.gas_qi,
                        'GAS_USER_DECLINE': st.session_state.gas_decline,
                        'GAS_USER_B_FACTOR': st.session_state.gas_b_factor,
                        'FCST_START_OIL': st.session_state.oil_fcst_start.strftime('%Y-%m-%d'),
                        'FCST_START_GAS': st.session_state.gas_fcst_start.strftime('%Y-%m-%d')
                    }
                    
                    # Add FAST_EDIT=0 if removing from fast edit
                    if remove_from_fast_edit:
                        update_values['FAST_EDIT'] = 0
                    
                    # Update database
                    success, message = update_well_parameters(api_uwi, update_values)
                    
                    if success:
                        # Add to processed wells list if not already there
                        if api_uwi not in st.session_state['wells_processed']:
                            st.session_state['wells_processed'].append(api_uwi)
                        
                        # Show success message
                        st.success(message)
                        
                        # Refresh the wells data
                        fresh_wells_data = refresh_wells_data()
                        # Get the next well API
                        next_well = get_next_well(api_uwi, wells_data)

                        # Move to next well
                        if next_well:
                            st.write('next well present')
                            st.session_state['selected_well'] = next_well
                            st.write(st.session_state['selected_well'])
                            # Force page refresh
                            st.rerun()
                        else:
                            st.info("No more wells to process!")
                    else:
                        st.error(message)
            
            # Well statistics section
            if not production_data.empty:
                st.markdown('<div class="section-header">Well Statistics</div>', unsafe_allow_html=True)
                st.metric("Total Oil (BBL)", f"{production_data['LIQUIDSPROD_BBL'].sum():,.0f}")
                st.metric("Total Gas (MCF)", f"{production_data['GASPROD_MCF'].sum():,.0f}")
                
                # Calculate overall last production date
                if 'PRODUCINGMONTH' in production_data.columns:
                    last_prod_date = production_data['PRODUCINGMONTH'].max()
                    st.write(f"**Last Production:** {pd.to_datetime(last_prod_date).strftime('%Y-%m-%d')}")
                
                # Show last production dates by product
                if last_oil_date is not None:
                    st.write(f"**Last Oil Production:** {last_oil_date.strftime('%Y-%m-%d')}")
                if last_gas_date is not None:
                    st.write(f"**Last Gas Production:** {last_gas_date.strftime('%Y-%m-%d')}")
else:
    st.info("Please select a well from the sidebar to begin editing.")

#############################################
# SECTION 11: FOOTER
#############################################

# Add a footer
st.markdown("---")
st.caption("Fast Edit Well Application - Streamlined interface for rapid well parameter updates with custom forecast start dates")