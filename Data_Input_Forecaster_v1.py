import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
# from snowflake.snowpark.context import get_active_session
import snowflake.connector

# Set Streamlit Page Configuration
st.set_page_config(page_title="Table Updater", layout="wide")

# Sidebar Inputs
st.sidebar.title("Table Updater")
table_name = st.sidebar.text_input("Enter table name", "ECON_INPUT")
primary_key = st.sidebar.text_input("Primary Key Column", "API_UWI")

# # Database Connection (Uses Snowpark Session)
# @st.cache_resource
# def get_snowflake_session():
#     return get_active_session()

# conn = get_snowflake_session()

##eliis snwoflake connection
# Set up Snowflake connection
conn = snowflake.connector.connect(
    user="ELII",
    password="Elii123456789!",
    account="CMZNSCB-MU47932",
    warehouse="COMPUTE_WH",
    database="WELLS",
    schema="MINERALS"
)

# Get Table Columns
@st.cache_data(ttl=600)
def get_table_columns(table):
    query = f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table}' ORDER BY ORDINAL_POSITION"
    # df = pd.DataFrame(conn.sql(query).collect())
    df = pd.read_sql(query, conn)
    return df.set_index("COLUMN_NAME")["DATA_TYPE"].to_dict()

# Fetch Table Data
@st.cache_data(ttl=60)
def get_table_data(table, where_clause=""):
    query = f"SELECT * FROM {table} {f'WHERE {where_clause}' if where_clause else ''} LIMIT 1000"
    # return pd.DataFrame(conn.sql(query).collect())
    return pd.read_sql(query, conn)

# Fetch Well Data for Filtering and Mapping
@st.cache_data(ttl=600)
def get_well_data():
    query = """
    SELECT API_UWI, WELLNAME, STATEPROVINCE, COUNTRY, COUNTY, FIRSTPRODDATE, LATITUDE, LONGITUDE,
           ENVOPERATOR, LEASE, ENVWELLSTATUS, ENVINTERVAL, TRAJECTORY, CUMGAS_MCF, CUMOIL_BBL, TOTALPRODUCINGMONTHS
    FROM wells.minerals.vw_well_input
    """
    try:
        # df = pd.DataFrame(conn.sql(query).collect())
        df = pd.read_sql(query, conn)
        if df.empty:
            st.warning("No well data retrieved from database.")
            return None
        # Fill NaNs in the CUMOIL_BBL and CUMGAS_MCF columns with 0
        df["CUMOIL_BBL"] = df["CUMOIL_BBL"].fillna(0)
        df["CUMGAS_MCF"] = df["CUMGAS_MCF"].fillna(0)
        return df
    except Exception as e:
        st.error(f"Error fetching well data: {e}")
        return None

# Fetch Production Data
@st.cache_data(ttl=600)
def get_production_data(api_uwi):
    query = f"""
    SELECT API_UWI, ProducingMonth, LIQUIDSPROD_BBL, GASPROD_MCF
    FROM wells.minerals.raw_prod_data
    WHERE API_UWI = '{api_uwi}'
    ORDER BY ProducingMonth;
    """
    try:
        # result = conn.sql(query).collect()
        result = pd.read_sql(query, conn)
        # Debug information to understand what's being returned
        if len(result) == 0:
            print(f"No production data found for API_UWI: {api_uwi}")
        else:
            print(f"Found {len(result)} production records for API_UWI: {api_uwi}")
        return pd.DataFrame(result)
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
    if 'ProducingMonth' in production_df.columns:
        production_df = production_df.sort_values('ProducingMonth')
    
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

# ----- WELL FILTERING SECTION (From Multi-Page App) -----
with st.expander("Well Filtering and Map", expanded=True):
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
        
        # Create a two-column layout: filters on left, map on right
        filters_column, map_column = st.columns([1, 1])
        
        # Initialize filtered_wells to be the original well data
        filtered_wells = well_data.copy()
        
        # ----- LEFT COLUMN: FILTERS -----
        with filters_column:
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
            
            # Map visualization options 
            map_options_col1, map_options_col2 = st.columns(2)
            
            with map_options_col1:
                # Color wells by attribute
                color_by = st.selectbox(
                    "Color wells by:",
                    options=["Operator", "Oil Production", "Gas Production", "Status", "Trajectory", "Interval"],
                    index=0  # Default to Operator
                )
            
            with map_options_col2:
                # Map style options
                map_style_options = {
                    "Streets": "mapbox://styles/mapbox/streets-v11",
                    "Satellite": "mapbox://styles/mapbox/satellite-v9",
                    "Light": "mapbox://styles/mapbox/light-v10",
                    "Dark": "mapbox://styles/mapbox/dark-v10"
                }
                selected_map_style = st.selectbox(
                    "Map Style:",
                    options=list(map_style_options.keys()),
                    index=3  # Default to Dark style
                )
                map_style = map_style_options[selected_map_style]
            
            # Display filtered well count
            st.write(f"Filtered Wells: {len(filtered_wells)}")
            
            # Add button to use these filtered wells for operations
            if not filtered_wells.empty:
                if st.button("Use Filtered Wells for Updating"):
                    st.session_state['filtered_wells'] = filtered_wells
                    st.session_state['selected_wells'] = filtered_wells["API_UWI"].tolist()
                    st.success(f"Selected {len(filtered_wells)} wells for operations. You can now use these wells in the sections below.")
                    # Force refresh to show filtered data
                    st.rerun()
        
        # ----- RIGHT COLUMN: MAP -----
        with map_column:
            st.subheader("Well Map")
            
            # Initialize map_df outside the condition
            map_df = pd.DataFrame()
            
            # Map visualization of filtered wells
            if not filtered_wells.empty:
                # Prepare data for PyDeck Map
                map_df = filtered_wells.copy()
                map_df = map_df.rename(columns={"LATITUDE": "lat", "LONGITUDE": "lon"})
                map_df = map_df.dropna(subset=["lat", "lon"])
                
                if not map_df.empty:
                    # Convert columns to native Python types
                    map_df["lat"] = map_df["lat"].astype(float)
                    map_df["lon"] = map_df["lon"].astype(float)
                    map_df["CUMOIL_BBL"] = map_df["CUMOIL_BBL"].astype(float)
                    map_df["CUMGAS_MCF"] = map_df["CUMGAS_MCF"].astype(float)
                    map_df["API_UWI"] = map_df["API_UWI"].astype(str)
                    
                    # Define function to assign colors based on selected attribute
                    def get_color_mapping(df, attribute):
                        if attribute == "Oil Production":
                            # Oil production colors
                            def oil_color(oil):
                                if oil >= 50000:
                                    return [0, 255, 0, 140]   # Green for high production
                                elif oil >= 10000:
                                    return [255, 165, 0, 140] # Orange for medium production
                                else:
                                    return [255, 0, 0, 140]   # Red for low production
                            
                            df["color"] = df["CUMOIL_BBL"].apply(oil_color)
                        elif attribute == "Gas Production":
                            # Gas production colors
                            def gas_color(gas):
                                if gas >= 500000:
                                    return [0, 0, 255, 140]   # Blue for high production
                                elif gas >= 100000:
                                    return [75, 0, 130, 140]  # Indigo for medium production
                                else:
                                    return [148, 0, 211, 140] # Violet for low production
                            
                            df["color"] = df["CUMGAS_MCF"].apply(gas_color)
                        else:
                            # For categorical attributes (Operator, Status, Trajectory, Interval)
                            if attribute == "Operator":
                                column = "ENVOPERATOR"
                            elif attribute == "Status":
                                column = "ENVWELLSTATUS"
                            elif attribute == "Trajectory":
                                column = "TRAJECTORY" 
                            elif attribute == "Interval":
                                column = "ENVINTERVAL"
                            
                            # Get unique values (limit to top 10 for readability)
                            unique_values = df[column].dropna().unique()
                            if len(unique_values) > 10:
                                # Get the top 10 most common values
                                top_values = df[column].value_counts().head(10).index.tolist()
                                unique_values = top_values
                            
                            # Create color mapping
                            import random
                            color_map = {}
                            
                            for i, value in enumerate(unique_values):
                                # Generate distinct colors
                                if i % 7 == 0:
                                    color = [255, 0, 0, 140]  # Red
                                elif i % 7 == 1:
                                    color = [0, 255, 0, 140]  # Green
                                elif i % 7 == 2:
                                    color = [0, 0, 255, 140]  # Blue
                                elif i % 7 == 3:
                                    color = [255, 255, 0, 140]  # Yellow
                                elif i % 7 == 4:
                                    color = [0, 255, 255, 140]  # Cyan
                                elif i % 7 == 5:
                                    color = [255, 0, 255, 140]  # Magenta
                                else:
                                    color = [255, 165, 0, 140]  # Orange
                                    
                                color_map[value] = color
                            
                            # Apply color mapping
                            df["color"] = df[column].map(lambda x: color_map.get(x, [100, 100, 100, 140]))
                        
                        return df
                    
                    # Apply color mapping
                    map_df = get_color_mapping(map_df, color_by)
                    
                    # Create a DataFrame for the color channels and convert to native int
                    color_df = pd.DataFrame(map_df["color"].tolist(), index=map_df.index, columns=["r", "g", "b", "a"])
                    map_df = map_df.join(color_df)
                    map_df = map_df.drop(columns=["color"])
                    map_df["r"] = map_df["r"].astype(int)
                    map_df["g"] = map_df["g"].astype(int)
                    map_df["b"] = map_df["b"].astype(int)
                    map_df["a"] = map_df["a"].astype(int)
                    
                    # Keep only required columns for visualization
                    map_df = map_df[["WELLNAME", "lat", "lon", "r", "g", "b", "a", "CUMOIL_BBL", "CUMGAS_MCF", 
                                  "API_UWI", "ENVOPERATOR", "ENVWELLSTATUS", "TRAJECTORY", "ENVINTERVAL"]]
                    
                    # Calculate view state with auto-zoom based on data bounds
                    if not map_df.empty:
                        # Get the bounds of the data
                        min_lat = map_df["lat"].min()
                        max_lat = map_df["lat"].max()
                        min_lon = map_df["lon"].min()
                        max_lon = map_df["lon"].max()
                        
                        # Calculate center point
                        view_lat = (min_lat + max_lat) / 2
                        view_lon = (min_lon + max_lon) / 2
                        
                        # Calculate appropriate zoom level
                        lat_range = max_lat - min_lat
                        lon_range = max_lon - min_lon
                        
                        # Adjust zoom based on the data spread
                        max_range = max(lat_range, lon_range)
                        if max_range > 5:
                            zoom = 4
                        elif max_range > 2:
                            zoom = 5
                        elif max_range > 1:
                            zoom = 6
                        elif max_range > 0.5:
                            zoom = 7
                        elif max_range > 0.1:
                            zoom = 8
                        else:
                            zoom = 9
                    else:
                        # Default view if no data
                        view_lat, view_lon = 40.0, -98.0  # Center of the US
                        zoom = 4
                    
                    view_state = {
                        "latitude": view_lat,
                        "longitude": view_lon,
                        "zoom": zoom,
                        "pitch": 0,
                    }
                    
                    # Define the well layer for the map
                    well_layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=map_df,
                        get_position=["lon", "lat"],
                        get_color=["r", "g", "b", "a"],
                        get_radius=500,
                        pickable=True,
                    )
                    
                    # Create the deck with tooltip
                    tooltip_html = (
                        "<b>Well:</b> {WELLNAME}<br/>"
                        "<b>Operator:</b> {ENVOPERATOR}<br/>"
                        "<b>Status:</b> {ENVWELLSTATUS}<br/>"
                        "<b>Trajectory:</b> {TRAJECTORY}<br/>"
                        "<b>Interval:</b> {ENVINTERVAL}<br/>"
                        "<b>Oil:</b> {CUMOIL_BBL} BBL<br/>"
                        "<b>Gas:</b> {CUMGAS_MCF} MCF<br/>"
                        "<b>API:</b> {API_UWI}"
                    )
                    
                    deck = pdk.Deck(
                        layers=[well_layer],
                        initial_view_state=view_state,
                        tooltip={"html": tooltip_html},
                        map_style=map_style,
                        height=600,  # Set a fixed height for the map
                    )
                    
                    # Display the map with full height in its column
                    st.pydeck_chart(deck, use_container_width=True)
                else:
                    st.warning("No wells with location data match the selected filters.")
            else:
                st.warning("No wells match the selected filters.")
    else:
        st.warning("Could not load well data for filtering.")

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
            if primary_key in data.columns:
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
                st.warning(f"Primary key '{primary_key}' not found in the table.")
else:
    data = pd.DataFrame()
    st.warning("Please enter a table name")

# Main content with expandable sections
st.title("Well Data Updating")

# ----- SECTION 1: SINGLE WELL UPDATING -----
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
                        pass
                        # # Apply decline rates if checkbox is selected and values exist in session state
                        # if apply_calc_decline and selected_key in st.session_state['calculated_declines']:
                        #     oil_decline, gas_decline = st.session_state['calculated_declines'][selected_key]
                        #     update_values["OIL_EMPIRICAL_DECLINE"] = str(oil_decline)
                        #     update_values["GAS_EMPIRICAL_DECLINE"] = str(gas_decline)

                        # # Handle DATE fields (convert empty strings to NULL)
                        # update_values = {
                        #     col: f"'{value}'" if value else "NULL"
                        #     if table_columns.get(col, "") != "DATE" else "NULL" if value == "" else f"'{value}'"
                        #     for col, value in update_values.items()
                        # }

                        # set_clause = ", ".join([f"{col} = {value}" for col, value in update_values.items()])
                        # sql = f"UPDATE {table_name} SET {set_clause} WHERE {primary_key} = '{selected_key}'"

                        # try:
                        #     conn.sql(sql).collect()
                        #     st.success(f"Record {primary_key} = {selected_key} updated successfully!")
                        #     st.cache_data.clear()
                        #     st.rerun()
                        # except Exception as e:
                        #     st.error(f"Error updating record: {e}")
            
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
                    
                    # Create a figure with dual y-axes for oil and gas
                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    
                    # Set Seaborn style
                    sns.set_style("whitegrid")
                    
                    # Set y-axis to log scale if requested
                    if log_scale:
                        ax1.set_yscale('log')
                    
                    # Plot oil production on the left y-axis if requested
                    if show_oil:
                        ax1.set_xlabel('Production Month')
                        ax1.set_ylabel('Oil Production (BBL)', color='tab:blue')
                        oil_line = ax1.plot(production_data['PRODUCINGMONTH'], production_data['LIQUIDSPROD_BBL'], 
                                    color='tab:blue', marker='o', linestyle='-', label='Oil (BBL)')
                        ax1.tick_params(axis='y', labelcolor='tab:blue')
                        ax1.grid(True, alpha=0.3)
                    
                    # Create a second y-axis for gas production if requested
                    if show_gas:
                        if show_oil:
                            # If oil is shown, use secondary y-axis
                            ax2 = ax1.twinx()
                            ax2.set_ylabel('Gas Production (MCF)', color='tab:red')
                            gas_line = ax2.plot(production_data['PRODUCINGMONTH'], production_data['GASPROD_MCF'], 
                                    color='tab:red', marker='x', linestyle='-', label='Gas (MCF)')
                            ax2.tick_params(axis='y', labelcolor='tab:red')
                            # Set log scale for gas axis if needed
                            if log_scale:
                                ax2.set_yscale('log')
                        else:
                            # If only gas is shown, use primary axis
                            ax1.set_xlabel('Production Month')
                            ax1.set_ylabel('Gas Production (MCF)', color='tab:red')
                            gas_line = ax1.plot(production_data['PRODUCINGMONTH'], production_data['GASPROD_MCF'], 
                                    color='tab:red', marker='x', linestyle='-', label='Gas (MCF)')
                            ax1.tick_params(axis='y', labelcolor='tab:red')
                    
                    # Add a title with well information
                    plt.title(f'Production History for {selected_key}', fontsize=14)
                    
                    # Add legend if both oil and gas are shown
                    if show_oil and show_gas:
                        lines1, labels1 = ax1.get_legend_handles_labels()
                        lines2, labels2 = ax2.get_legend_handles_labels()
                        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                    
                    # Format the plot
                    plt.tight_layout()
                    
                    # Rotate x-axis labels for better readability
                    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
                    
                    # Display the plot in Streamlit
                    st.pyplot(fig)
                    
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
        
# ----- SECTION 2: BULK DATA UPDATING -----
# with st.expander("Bulk Data Updating", expanded=False):
#     st.subheader("Update Multiple Records")
    
#     if not data.empty:
#         with st.form("bulk_update_form"):
#             st.subheader("Bulk Update Fields")
            
#             # Get ordered fields that actually exist in the table
#             existing_oil_fields = get_ordered_fields(oil_fields, table_columns)
#             existing_gas_fields = get_ordered_fields(gas_fields, table_columns)
            
#             # Create two columns for oil and gas parameters
#             oil_col, gas_col = st.columns(2)
            
#             # Oil parameters column
#             with oil_col:
#                 st.markdown("### Oil Parameters")
#                 oil_bulk_values = {}
                
#                 for field, data_type in existing_oil_fields:
#                     oil_bulk_values[field] = st.text_input(f"New Value for {field}")
            
#             # Gas parameters column
#             with gas_col:
#                 st.markdown("### Gas Parameters")
#                 gas_bulk_values = {}
                
#                 for field, data_type in existing_gas_fields:
#                     gas_bulk_values[field] = st.text_input(f"New Value for {field}")
            
#             # Other fields (not oil or gas specific)
#             st.markdown("### Other Parameters")
#             other_bulk_values = {}
#             for col in table_columns:
#                 if (col != primary_key and 
#                     col.lower() != 'lease' and
#                     col not in oil_bulk_values and 
#                     col not in gas_bulk_values and
#                     not (col.startswith("OIL_") or col.startswith("GAS_"))):
#                     other_bulk_values[col] = st.text_input(f"New Value for {col}")
            
#             # Combine all bulk values
#             bulk_values = {**oil_bulk_values, **gas_bulk_values, **other_bulk_values}
            
#             # Option to apply calculated decline rates
#             apply_calc_decline = st.checkbox("Apply calculated decline rates (if available)", key="bulk_apply_decline")
            
#             submit_bulk = st.form_submit_button("Execute Bulk Update")

#         if submit_bulk:
#             update_errors = []
#             progress_bar = st.progress(0)
#             status_text = st.empty()
#             success_count = 0
            
#             # Loop through each record in the filtered table data
#             for i, (index, row) in enumerate(data.iterrows()):
#                 api_uwi = row[primary_key]
#                 status_text.text(f"Updating {i+1}/{len(data)}: {api_uwi}")
                
#                 # Copy bulk user inputs
#                 record_bulk_values = bulk_values.copy()
                
#                 # Apply decline rates if checkbox is selected and values exist in session state
#                 if apply_calc_decline and api_uwi in st.session_state['calculated_declines']:
#                     oil_decline, gas_decline = st.session_state['calculated_declines'][api_uwi]
#                     record_bulk_values["OIL_EMPIRICAL_DECLINE"] = str(oil_decline)
#                     record_bulk_values["GAS_EMPIRICAL_DECLINE"] = str(gas_decline)
                
#                 # Auto-populate last production dates if those fields exist
#                 if ('LAST_OIL_DATE' in table_columns or 'LAST_GAS_DATE' in table_columns or 
#                     'LAST_PROD_DATE' in table_columns):
#                     last_oil_date, last_gas_date = get_last_production_dates(api_uwi)
                    
#                     # Update individual date fields if they exist
#                     if last_oil_date is not None and 'LAST_OIL_DATE' in table_columns:
#                         record_bulk_values['LAST_OIL_DATE'] = str(last_oil_date)
#                     if last_gas_date is not None and 'LAST_GAS_DATE' in table_columns:
#                         record_bulk_values['LAST_GAS_DATE'] = str(last_gas_date)
                    
#                     # Determine and update the most recent production date for LAST_PROD_DATE
#                     if 'LAST_PROD_DATE' in table_columns:
#                         most_recent_date = None
#                         if last_oil_date is not None and last_gas_date is not None:
#                             most_recent_date = max(last_oil_date, last_gas_date)
#                         elif last_oil_date is not None:
#                             most_recent_date = last_oil_date
#                         elif last_gas_date is not None:
#                             most_recent_date = last_gas_date
                        
#                         if most_recent_date is not None:
#                             record_bulk_values['LAST_PROD_DATE'] = str(most_recent_date)
                
#                 # Auto-populate Qi values if those fields exist
#                 if 'OIL_CALC_QI' in table_columns or 'GAS_CALC_QI' in table_columns:
#                     oil_qi, gas_qi = calculate_qi_values(api_uwi)
#                     if oil_qi > 0 and 'OIL_CALC_QI' in table_columns:
#                         record_bulk_values['OIL_CALC_QI'] = str(oil_qi)
#                     if gas_qi > 0 and 'GAS_CALC_QI' in table_columns:
#                         record_bulk_values['GAS_CALC_QI'] = str(gas_qi)

#                 # Only update if there are values to set
#                 if any(v for v in record_bulk_values.values()):
#                     # Handle DATE fields and empty values (convert empty strings to NULL)
#                     record_bulk_values = {
#                         col: f"'{value}'" if value else "NULL"
#                         if table_columns.get(col, "") != "DATE" else "NULL" if value == "" else f"'{value}'"
#                         for col, value in record_bulk_values.items() if value
#                     }

#                     if record_bulk_values:
#                         set_clause = ", ".join([f"{col} = {value}" for col, value in record_bulk_values.items()])
#                         sql = f"UPDATE {table_name} SET {set_clause} WHERE {primary_key} = '{api_uwi}'"
#                         try:
#                             conn.sql(sql).collect()
    #                         success_count += 1
    #                     except Exception as e:
    #                         update_errors.append(f"Error updating record {api_uwi}: {e}")
                
    #             # Update progress
    #             progress_bar.progress((i + 1) / len(data))

    #         if update_errors:
    #             st.error("Some updates failed:\n" + "\n".join(update_errors))
            
    #         status_text.text(f"Update complete! Successfully updated {success_count} records.")
    #         if success_count > 0:
    #             st.success(f"Successfully updated {success_count} records")
    #             st.cache_data.clear()
    # else:
    #     st.warning("No data available for bulk update")

# ----- SECTION 3: BULK DECLINE CALCULATION -----
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
                
                # Get the greater of the two dates (Last Oil Date or Last Gas Date) for LAST_PROD_DATE
                if "Last Production Dates" in additional_calcs:
                    last_oil_date, last_gas_date = get_last_production_dates(api_uwi)
                    
                    # Determine the most recent production date (greater of the two)
                    most_recent_date = None
                    if last_oil_date is not None and last_gas_date is not None:
                        most_recent_date = max(last_oil_date, last_gas_date)
                    elif last_oil_date is not None:
                        most_recent_date = last_oil_date
                    elif last_gas_date is not None:
                        most_recent_date = last_gas_date
                    
                    # Add to result dictionary for display
                    result["Last Production Date"] = most_recent_date
                    
                    # Store for database update
                    if most_recent_date is not None:
                        calc_values[api_uwi]["LAST_PROD_DATE"] = most_recent_date
                
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
                result_fields.extend(["Last Oil Date", "Last Gas Date"])
            if "Initial Rate (Qi) Values" in additional_calcs:
                result_fields.extend(["Oil Initial Rate (Qi)", "Gas Initial Rate (Qi)"])
                
            # Create dataframe with only the calculated fields
            display_fields = [field for field in result_fields if any(field in result for result in calculation_results)]
            decline_df = pd.DataFrame(calculation_results)[display_fields]
            st.dataframe(decline_df)
            
            # Option to update database with calculated values
            if st.button("Apply Calculated Rates to Database"):
                pass
                # progress_bar = st.progress(0)
                # status_text = st.empty()
                # update_errors = []
                # success_count = 0
                
                # for i, api_uwi in enumerate(calc_values.keys()):
                #     status_text.text(f"Updating {i+1}/{len(calc_values)}: {api_uwi}")
                    
                #     # Get the calculated values for this well
                #     well_values = calc_values[api_uwi]
                    
                #     # Skip if no values to update
                #     if not well_values:
                #         continue
                    
                #     # Create SET clause with proper formatting for each data type
                #     set_parts = []
                    
                #     for col, value in well_values.items():
                #         # Check if column exists in table
                #         if col in table_columns:
                #             # Format based on data type
                #             if col.startswith("LAST_") and table_columns[col] == "DATE":
                #                 # Date format
                #                 set_parts.append(f"{col} = '{value}'")
                #             elif isinstance(value, (int, float)):
                #                 # Numeric format
                #                 set_parts.append(f"{col} = {value}")
                #             else:
                #                 # String format
                #                 set_parts.append(f"{col} = '{value}'")
                    
                #     # Only proceed if we have values to update
                #     if set_parts:
                #         set_clause = ", ".join(set_parts)
                        
                #         # Build and execute SQL update
                #         sql = f"""
                #         UPDATE {table_name} 
                #         SET {set_clause}
                #         WHERE {primary_key} = '{api_uwi}'
                #         """
                        
                #         try:
                #             conn.sql(sql).collect()
                #             success_count += 1
                #         except Exception as e:
                #             update_errors.append(f"Error updating record {api_uwi}: {e}")
                    
                #     # Update progress
                #     progress_bar.progress((i + 1) / len(calc_values))
                
                # if update_errors:
                #     st.error("Some updates failed:\n" + "\n".join(update_errors))
                
                # status_text.text(f"Update complete! Successfully updated {success_count} records.")
                # if success_count > 0:
                #     st.success(f"Successfully updated {success_count} records with calculated values")
                #     st.cache_data.clear()
    else:
        st.warning("No data available for bulk calculation")