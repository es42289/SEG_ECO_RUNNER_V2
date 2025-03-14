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

# Set Streamlit Page Configuration
st.set_page_config(page_title="Table Updater", layout="wide")

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

##########################################
# SECTION 8: WELL FILTERING
##########################################
with st.expander("Well Filtering", expanded=True):
    st.subheader("Filter Wells for Updates")
    
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
        
        # Create columns for filters to make more compact
        filter_col1, filter_col2 = st.columns(2)
        
        # Apply filters
        with filter_col1:
            for col in filter_columns[:3]:  # First half of the filters
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
        
        with filter_col2:
            for col in filter_columns[3:]:  # Second half of the filters
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
        slider_col1, slider_col2 = st.columns(2)
        
        with slider_col1:
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
        
        with slider_col2:
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
        
        # Display filtered well count and table
        st.write(f"Filtered Wells: {len(filtered_wells)}")
        
        # Add button to use these filtered wells for operations
        if not filtered_wells.empty:
            if st.button("Use Filtered Wells for Updating"):
                st.session_state['filtered_wells'] = filtered_wells
                st.session_state['selected_wells'] = filtered_wells["API_UWI"].tolist()
                st.success(f"Selected {len(filtered_wells)} wells for operations. You can now use these wells in the sections below.")
                # Force refresh to show filtered data
                st.rerun()
            
            # Show a sample of filtered wells
            st.subheader("Filtered Wells Sample")
            display_cols = ["API_UWI", "WELLNAME", "ENVOPERATOR", "COUNTY", "CUMOIL_BBL", "CUMGAS_MCF"]
            display_cols = [col for col in display_cols if col in filtered_wells.columns]
            st.dataframe(filtered_wells[display_cols].head(15))
            
            # Add visualization of top operators using Altair
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
            
            # Map visualization - fixed version
            st.subheader("Well Map Visualization")
            
            if 'LATITUDE' in filtered_wells.columns and 'LONGITUDE' in filtered_wells.columns:
                # Create a copy of the dataframe with required columns
                map_data = filtered_wells.copy()
                
                # Drop any rows with missing coordinates
                map_data = map_data.dropna(subset=['LATITUDE', 'LONGITUDE'])
                
                if not map_data.empty:
                    # Create two columns for map controls
                    map_col1, map_col2 = st.columns(2)
                    
                    # Available color fields - use the same filters available in the app
                    color_fields = [("None (Default)", None)]  # Add default option first
                    if "CUMOIL_BBL" in map_data.columns:
                        color_fields.append(("Oil Production", "CUMOIL_BBL"))
                    if "CUMGAS_MCF" in map_data.columns:
                        color_fields.append(("Gas Production", "CUMGAS_MCF"))
                    if "TOTALPRODUCINGMONTHS" in map_data.columns:
                        color_fields.append(("Production Months", "TOTALPRODUCINGMONTHS"))
                    if "ENVOPERATOR" in map_data.columns:
                        color_fields.append(("Operator", "ENVOPERATOR"))
                    if "TRAJECTORY" in map_data.columns:
                        color_fields.append(("Trajectory", "TRAJECTORY"))
                    if "ENVWELLSTATUS" in map_data.columns:
                        color_fields.append(("Well Status", "ENVWELLSTATUS"))
                    
                    with map_col1:
                        # Default to Oil Production if available, otherwise first option
                        default_index = 0
                        for i, (label, field) in enumerate(color_fields):
                            if field == "CUMOIL_BBL":
                                default_index = i
                                break
                        
                        # Select field for coloring with a unique key
                        color_option = st.selectbox(
                            "Color wells by:",
                            options=[cf[0] for cf in color_fields],
                            index=default_index,
                            key="map_color_select"
                        )
                        
                        # Get the corresponding column name
                        color_field = next((cf[1] for cf in color_fields if cf[0] == color_option), None)
                    
                    with map_col2:
                        # Size options - add "Uniform Size" as the first option
                        size_options = [("Uniform Size", None)]
                        if "CUMOIL_BBL" in map_data.columns:
                            size_options.append(("Oil Production", "CUMOIL_BBL"))
                        if "CUMGAS_MCF" in map_data.columns:
                            size_options.append(("Gas Production", "CUMGAS_MCF"))
                        if "TOTALPRODUCINGMONTHS" in map_data.columns:
                            size_options.append(("Production Months", "TOTALPRODUCINGMONTHS"))
                        
                        # Select field for sizing with a unique key
                        size_option = st.selectbox(
                            "Size wells by:",
                            options=[sf[0] for sf in size_options],
                            index=0,  # Default to Uniform Size
                            key="map_size_select"
                        )
                        
                        # Get the corresponding column name
                        size_field = next((sf[1] for sf in size_options if sf[0] == size_option), None)
                    
                    # Create a dataframe for the map
                    simple_map_data = map_data[['LATITUDE', 'LONGITUDE']].copy()
                    simple_map_data.columns = ['latitude', 'longitude']
                    
                    # Add color field if selected
                    if color_field and color_field in map_data.columns:
                        # For categorical fields like ENVOPERATOR, create a color mapping
                        if map_data[color_field].dtype == 'object' or map_data[color_field].dtype.name == 'category':
                            # Get unique values and sort them for consistency
                            unique_values = sorted(map_data[color_field].dropna().unique())
                            
                            # Create a color mapping (1-10 scale for categorical values)
                            color_mapping = {val: (i % 10) + 1 for i, val in enumerate(unique_values)}
                            
                            # Apply mapping
                            simple_map_data['color'] = map_data[color_field].map(color_mapping)
                            
                            # Show a legend
                            st.write("Color Legend:")
                            # Create a more compact legend with multiple columns
                            for i in range(0, len(unique_values), 3):
                                # Take up to 3 items at a time
                                items = unique_values[i:i+3]
                                cols = st.columns(3)
                                for j, val in enumerate(items):
                                    if val in color_mapping:
                                        color_value = color_mapping[val]
                                        cols[j].markdown(f"**{val}**: Color {color_value}")
                        else:
                            # For numeric fields, normalize to 1-10 range
                            min_val = map_data[color_field].min()
                            max_val = map_data[color_field].max()
                            
                            if min_val != max_val:  # Avoid division by zero
                                simple_map_data['color'] = 1 + 9 * (map_data[color_field] - min_val) / (max_val - min_val)
                            else:
                                simple_map_data['color'] = 5  # Default mid-range value
                            
                            # Show min/max values
                            st.write(f"Color range: Min {min_val:,.2f} to Max {max_val:,.2f}")
                    
                    # Add size field if selected
                    if size_field and size_field in map_data.columns:
                        # Normalize size to 5-15 range
                        min_val = map_data[size_field].min()
                        max_val = map_data[size_field].max()
                        
                        if min_val != max_val:  # Avoid division by zero
                            simple_map_data['size'] = 5 + 10 * (map_data[size_field] - min_val) / (max_val - min_val)
                        else:
                            simple_map_data['size'] = 8  # Default mid-range size
                    else:
                        # Fixed size for all points
                        simple_map_data['size'] = 8
                    
                    # Display the map
                    st.map(simple_map_data)
                    
                    # Add information about the map
                    st.info(f"Map shows {len(simple_map_data)} wells. " + 
                            (f"Wells are colored by {color_option}" if color_field else "Wells use default coloring") + 
                            " and " + 
                            (f"sized by {size_option}." if size_field else "have uniform size."))
                    
                    # Add a note about map refreshing if needed
                    st.caption("If the map doesn't update immediately when changing options, please click outside the dropdown after making your selection.")
                else:
                    st.warning("Cannot display map: No valid coordinate data available")
            else:
                st.warning("Cannot display map: Latitude or longitude data is missing")
    else:
        st.warning("Could not load well data for filtering.")
        
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
                st.markdown("---")
                st.subheader("Well Information")
                if 'ENVOPERATOR' in record:
                    st.markdown(f"**Operator:** {record['ENVOPERATOR']}")
                if 'WELLNAME' in record:
                    st.markdown(f"**Well Name:** {record['WELLNAME']}")
                if 'ENVWELLSTATUS' in record:
                    st.markdown(f"**Status:** {record['ENVWELLSTATUS']}")
                if 'FIRSTPRODDATE' in record:
                    st.markdown(f"**First Production:** {record['FIRSTPRODDATE']}")
                
                # Add the update form directly in the left column
                st.markdown("---")
                
                # Update form
                with st.form("update_form"):
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
                        chart_data['Oil (BBL)'] = production_data['LIQUIDSPROD_BBL']
                    
                    if show_gas:
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
                        
                        # Create scale based on log toggle
                        if log_scale:
                            y_scale = alt.Scale(type='log', domainMin=1)
                            y_title = 'Production Volume (Log Scale)'
                        else:
                            y_scale = alt.Scale(zero=True)
                            y_title = 'Production Volume'
                        
                        # Create color scale
                        color_mapping = {'Oil (BBL)': '#1f77b4', 'Gas (MCF)': '#ff7f0e'}
                        
                        # Create the chart
                        chart = alt.Chart(melted_data).mark_line(point=True).encode(
                            x=alt.X('Production Month:T', title='Production Month'),
                            y=alt.Y('Volume:Q', 
                                  scale=y_scale, 
                                  title=y_title),
                            color=alt.Color('Production Type:N', 
                                            scale=alt.Scale(domain=list(color_mapping.keys()), 
                                                            range=list(color_mapping.values())),
                                            legend=alt.Legend(title='Production Type')),
                            tooltip=['Production Month', 'Production Type', 'Volume']
                        ).properties(
                            title=f'Production History for {selected_key}',
                            height=400
                        ).interactive()
                        
                        # Display the chart
                        st.altair_chart(chart, use_container_width=True)
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