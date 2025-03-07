import streamlit as st
import pandas as pd
import pydeck as pdk
import altair as alt
import json

# Set page configuration to use wide layout with light theme
st.set_page_config(
    page_title="Well Data Visualization",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for sidebar file-like navigation
st.markdown("""
<style>
    /* File-like navigation styling */
    .file-nav {
        padding: 0;
        margin: 10px 0;
    }
    
    .file-nav-item {
        display: flex;
        align-items: center;
        padding: 8px 12px;
        margin: 2px 0;
        border-radius: 4px;
        background-color: #f8f9fa;
        transition: background-color 0.2s;
        cursor: pointer;
        text-decoration: none;
        color: #333;
    }
    
    .file-nav-item:hover {
        background-color: #e9ecef;
    }
    
    .file-nav-item.active {
        background-color: #dde5f9;
        border-left: 3px solid #4c8bf5;
    }
    
    .file-nav-icon {
        margin-right: 10px;
        font-size: 1.1rem;
        width: 20px;
        text-align: center;
    }
    
    /* Style for radio buttons */
    .stRadio label {
        font-weight: bold;
        padding: 5px 10px;
        border-radius: 5px;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = None
if 'filtered_secondary_df' not in st.session_state:
    st.session_state.filtered_secondary_df = None
if 'selected_well' not in st.session_state:
    st.session_state.selected_well = None

# Establish Snowflake Connection
conn = st.connection("snowflake")

def get_well_data():
    query = """
    SELECT API_UWI, WELLNAME, STATEPROVINCE, COUNTRY, COUNTY, FIRSTPRODDATE, LATITUDE, LONGITUDE,
           ENVOPERATOR, LEASE, ENVWELLSTATUS, ENVINTERVAL, TRAJECTORY, CUMGAS_MCF, CUMOIL_BBL, TOTALPRODUCINGMONTHS
    FROM wells.minerals.vw_well_input
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            df = cur.fetch_pandas_all()
        if df.empty:
            st.warning("No data retrieved from Snowflake.")
            return None
        df.columns = df.columns.str.upper()
        # Fill NaNs in the CUMOIL_BBL and CUMGAS_MCF columns with 0
        df["CUMOIL_BBL"] = df["CUMOIL_BBL"].fillna(0)
        df["CUMGAS_MCF"] = df["CUMGAS_MCF"].fillna(0)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def get_secondary_table_data():
    query = """
    SELECT API_UWI, GASPROD_MCF, LIQUIDSPROD_BBL, PRODUCINGMONTH, 
           TOTALPRODMONTHS, WATERPROD_BBL, WELLNAME
    FROM wells.minerals.VW_PROD_INPUT
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            df = cur.fetch_pandas_all()
        if df.empty:
            st.warning("No production data retrieved from Snowflake.")
            return None
        df.columns = df.columns.str.upper()
        # Fill NaNs with 0 for numeric columns
        numeric_cols = ["GASPROD_MCF", "LIQUIDSPROD_BBL", "TOTALPRODMONTHS", "WATERPROD_BBL"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Convert PRODUCINGMONTH to datetime for proper time series handling
        if "PRODUCINGMONTH" in df.columns:
            df["PRODUCINGMONTH"] = pd.to_datetime(df["PRODUCINGMONTH"])
            
        return df
    except Exception as e:
        st.error(f"Error fetching production data: {e}")
        return None

# Function to switch pages
def switch_page(page_num):
    st.session_state.page = page_num

# App title with page indicator - standard style
st.title(f"⛽ Well Data Visualization - Page {st.session_state.page} of 2")

# Navigation in sidebar as file-like interface
st.sidebar.header("Pages")

# Create file-like navigation
page1_active = "active" if st.session_state.page == 1 else ""
st.sidebar.markdown(f"""
<div class="file-nav">
    <div class="file-nav-item {page1_active}" onclick="document.querySelector('button[data-testid=\"baseButton-secondary\"]:first-of-type').click()">
        <span class="file-nav-icon">📊</span> Visualizations
    </div>
</div>
""", unsafe_allow_html=True)

# Hidden button that will be clicked via the custom navigation
if st.sidebar.button("Page 1: Visualizations", key="nav_p1", help="", disabled=st.session_state.page == 1):
    switch_page(1)

page2_active = "active" if st.session_state.page == 2 else ""
st.sidebar.markdown(f"""
<div class="file-nav">
    <div class="file-nav-item {page2_active}" onclick="document.querySelector('button[data-testid=\"baseButton-secondary\"]:nth-of-type(2)').click()">
        <span class="file-nav-icon">📋</span> Data Tables
    </div>
</div>
""", unsafe_allow_html=True)

# Hidden button that will be clicked via the custom navigation
if st.sidebar.button("Page 2: Tables", key="nav_p2", help="", disabled=st.session_state.page == 2):
    switch_page(2)

# Load Data
if st.session_state.filtered_df is None:
    df = get_well_data()
    secondary_df = get_secondary_table_data()
    st.session_state.filtered_df = df
    st.session_state.filtered_secondary_df = secondary_df
else:
    # Use the data from session state
    df = st.session_state.filtered_df
    secondary_df = st.session_state.filtered_secondary_df

# ----------------- Sidebar: Dynamic Multi-select Filters -----------------
# Show filters on both pages
st.sidebar.header("Filters")

# Define filter columns in the specified order
filter_columns = [
    "TRAJECTORY", 
    "COUNTY", 
    "ENVWELLSTATUS", 
    "ENVOPERATOR", 
    "WELLNAME", 
    "API_UWI"
]

# Initialize filtered_df to be the original dataframe
if df is not None:
    filtered_df = df.copy()
else:
    filtered_df = pd.DataFrame()

# Apply filters in order
for col in filter_columns:
    if df is not None and col in df.columns:
        # Get unique values for the current filter, based on data filtered so far
        unique_values = sorted(filtered_df[col].dropna().unique().tolist())
        
        # Create multiselect widget
        selected_values = st.sidebar.multiselect(
            f"Select {col}:",
            options=unique_values,
            default=[]
        )
        
        # Apply filter if values are selected
        if selected_values:
            filtered_df = filtered_df[filtered_df[col].isin(selected_values)]

# Add oil production range slider
if df is not None and "CUMOIL_BBL" in df.columns:
    max_oil_value = int(df["CUMOIL_BBL"].max()) if not pd.isna(df["CUMOIL_BBL"].max()) else 0
    oil_range = st.sidebar.slider(
        "Total Oil Production (BBL)", 
        min_value=0, 
        max_value=max_oil_value, 
        value=(0, max_oil_value)
    )
    filtered_df = filtered_df[(filtered_df["CUMOIL_BBL"] >= oil_range[0]) & (filtered_df["CUMOIL_BBL"] <= oil_range[1])]

# Add gas production range slider
if df is not None and "CUMGAS_MCF" in df.columns:
    max_gas_value = int(df["CUMGAS_MCF"].max()) if not pd.isna(df["CUMGAS_MCF"].max()) else 0
    gas_range = st.sidebar.slider(
        "Total Gas Production (MCF)", 
        min_value=0, 
        max_value=max_gas_value, 
        value=(0, max_gas_value)
    )
    filtered_df = filtered_df[(filtered_df["CUMGAS_MCF"] >= gas_range[0]) & (filtered_df["CUMGAS_MCF"] <= gas_range[1])]

# Add total producing months range slider
if df is not None and "TOTALPRODUCINGMONTHS" in df.columns:
    max_months = int(df["TOTALPRODUCINGMONTHS"].max()) if not pd.isna(df["TOTALPRODUCINGMONTHS"].max()) else 0
    months_range = st.sidebar.slider(
        "Total Producing Months", 
        min_value=0, 
        max_value=max_months, 
        value=(0, max_months)
    )
    filtered_df = filtered_df[(filtered_df["TOTALPRODUCINGMONTHS"] >= months_range[0]) & (filtered_df["TOTALPRODUCINGMONTHS"] <= months_range[1])]

# Filter the secondary table based on the API_UWIs in the filtered well data
if secondary_df is not None and not filtered_df.empty:
    filtered_secondary_df = secondary_df[secondary_df["API_UWI"].isin(filtered_df["API_UWI"])]
else:
    filtered_secondary_df = pd.DataFrame()

# Update the session state with the latest filtered data
st.session_state.filtered_df = filtered_df
st.session_state.filtered_secondary_df = filtered_secondary_df

# Add map settings to sidebar on Page 1
if st.session_state.page == 1:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Map Settings")
    color_by = st.sidebar.selectbox(
        "Color wells by:",
        options=["Operator", "Oil Production", "Gas Production", "Status", "Trajectory", "Interval"],
        index=0  # Default to Operator
    )

    # Define map style options
    map_style_options = {
        "Streets": "mapbox://styles/mapbox/streets-v11",
        "Satellite": "mapbox://styles/mapbox/satellite-v9",
        "Light": "mapbox://styles/mapbox/light-v10",
        "Dark": "mapbox://styles/mapbox/dark-v10",
        "Outdoors": "mapbox://styles/mapbox/outdoors-v11",
        "Navigation Day": "mapbox://styles/mapbox/navigation-day-v1",
        "Navigation Night": "mapbox://styles/mapbox/navigation-night-v1"
    }

    # Let user select map style
    selected_map_style = st.sidebar.selectbox(
        "Map Style:",
        options=list(map_style_options.keys()),
        index=3  # Default to Dark style
    )
    
    map_style = map_style_options[selected_map_style]
    
    # Use dark mode for map if available (works with mapbox dark style)
    if selected_map_style == "Dark":
        map_style = "mapbox://styles/mapbox/dark-v10"  # Ensure dark style is used

# ----------------- Page 1: Visualizations -----------------
if st.session_state.page == 1:
    # Create a 3-column layout for Page 1 - narrower left for bar charts, wider center for map, small right for legend
    page1_col1, page1_col2, page1_col3 = st.columns([3, 6, 2])
    
    # ----------------- Left Column: Stacked Bar Charts -----------------
    with page1_col1:
        st.subheader("Well Count by Category")

        if not filtered_df.empty:
            # 1. Chart for Operators
            operator_counts = filtered_df["ENVOPERATOR"].value_counts().reset_index()
            operator_counts.columns = ["Operator", "Count"]
            # Limit to top 20 operators
            if len(operator_counts) > 20:
                operator_counts = operator_counts.head(20)
                title_operator = "Top 20 Operators"
            else:
                title_operator = "Operators"

            chart_operator = alt.Chart(operator_counts).mark_bar().encode(
                x=alt.X("Count:Q", title="Count"),
                y=alt.Y("Operator:N", title="Operator", sort="-x"),
                tooltip=["Operator", "Count"]
            ).properties(title=title_operator, height=200)
            
            # 2. Chart for Interval
            interval_counts = filtered_df["ENVINTERVAL"].value_counts().reset_index()
            interval_counts.columns = ["Interval", "Count"]
            # Limit to top 20
            if len(interval_counts) > 20:
                interval_counts = interval_counts.head(20)
                title_interval = "Top 20 Intervals"
            else:
                title_interval = "Intervals"

            chart_interval = alt.Chart(interval_counts).mark_bar().encode(
                x=alt.X("Count:Q", title="Count"),
                y=alt.Y("Interval:N", title="Interval", sort="-x"),
                tooltip=["Interval", "Count"]
            ).properties(title=title_interval, height=200)
            
            # 3. Chart for Status
            status_counts = filtered_df["ENVWELLSTATUS"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            # Limit to top 20
            if len(status_counts) > 20:
                status_counts = status_counts.head(20)
                title_status = "Top 20 Statuses"
            else:
                title_status = "Statuses"

            chart_status = alt.Chart(status_counts).mark_bar().encode(
                x=alt.X("Count:Q", title="Count"),
                y=alt.Y("Status:N", title="Status", sort="-x"),
                tooltip=["Status", "Count"]
            ).properties(title=title_status, height=200)
            
            # 4. Chart for Trajectory
            trajectory_counts = filtered_df["TRAJECTORY"].value_counts().reset_index()
            trajectory_counts.columns = ["Trajectory", "Count"]
            # Limit to top 20
            if len(trajectory_counts) > 20:
                trajectory_counts = trajectory_counts.head(20)
                title_trajectory = "Top 20 Trajectories"
            else:
                title_trajectory = "Trajectories"

            chart_trajectory = alt.Chart(trajectory_counts).mark_bar().encode(
                x=alt.X("Count:Q", title="Count"),
                y=alt.Y("Trajectory:N", title="Trajectory", sort="-x"),
                tooltip=["Trajectory", "Count"]
            ).properties(title=title_trajectory, height=200)
            
            # Display charts stacked in a single column
            st.altair_chart(chart_operator, use_container_width=True)
            st.altair_chart(chart_interval, use_container_width=True)
            st.altair_chart(chart_status, use_container_width=True)
            st.altair_chart(chart_trajectory, use_container_width=True)
        else:
            st.warning("No data to display. Please adjust your filters.")
    
# ----------------- Center Column: Map Visualization -----------------
    with page1_col2:
        st.subheader("Well Location")

        if not filtered_df.empty:
            # Prepare data for PyDeck Map Visualization
            map_df = filtered_df.copy()
            map_df = map_df.rename(columns={"LATITUDE": "lat", "LONGITUDE": "lon"})
            map_df = map_df.dropna(subset=["lat", "lon"])

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
                    legend_items = [
                        {"label": "High Production (≥ 50,000 BBL)", "color": "🟢 Green"},
                        {"label": "Medium Production (10,000 - 50,000 BBL)", "color": "🟠 Orange"},
                        {"label": "Low Production (< 10,000 BBL)", "color": "🔴 Red"}
                    ]
                    
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
                    legend_items = [
                        {"label": "High Production (≥ 500,000 MCF)", "color": "🔵 Blue"},
                        {"label": "Medium Production (100,000 - 500,000 MCF)", "color": "🟣 Indigo"},
                        {"label": "Low Production (< 100,000 MCF)", "color": "🟪 Violet"}
                    ]
                    
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
                    legend_items = []
                    
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
                        
                        # Convert RGB to color name for legend
                        color_names = ["🔴 Red", "🟢 Green", "🔵 Blue", "🟡 Yellow", "🔵 Cyan", "🟣 Magenta", "🟠 Orange"]
                        legend_items.append({"label": str(value), "color": color_names[i % 7]})
                    
                    # Apply color mapping
                    df["color"] = df[column].map(lambda x: color_map.get(x, [100, 100, 100, 140]))
                
                return df, legend_items

            # Apply color mapping
            map_df, legend_items = get_color_mapping(map_df, color_by)

            # Create a DataFrame for the color channels and convert to native int
            color_df = pd.DataFrame(map_df["color"].tolist(), index=map_df.index, columns=["r", "g", "b", "a"])
            map_df = map_df.join(color_df)
            map_df = map_df.drop(columns=["color"])
            map_df["r"] = map_df["r"].astype(int)
            map_df["g"] = map_df["g"].astype(int)
            map_df["b"] = map_df["b"].astype(int)
            map_df["a"] = map_df["a"].astype(int)

            # Keep only required columns for visualization
            map_df = map_df[["WELLNAME", "lat", "lon", "r", "g", "b", "a", "CUMOIL_BBL", "CUMGAS_MCF", "API_UWI", "ENVOPERATOR", 
                            "ENVWELLSTATUS", "TRAJECTORY", "ENVINTERVAL"]]

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
                # This is a simplified approach, more sophisticated methods could be used
                lat_range = max_lat - min_lat
                lon_range = max_lon - min_lon
                
                # Adjust zoom based on the data spread (larger spread = lower zoom)
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

            # Set up custom CSS to control the map height
            st.markdown("""
            <style>
            .stPydeck > iframe {
                height: 800px !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Define the well layer for the map
            well_layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position=["lon", "lat"],
                get_color=["r", "g", "b", "a"],
                get_radius=500,
                pickable=True,
            )

            # Create the deck with updated tooltip including all relevant attributes
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
                height=800,  # Increase map height to 800px
            )

            # Display the map in the center column
            # We'll use an alternative method to control height by creating a container first
            map_container = st.container()
            with map_container:
                st.pydeck_chart(deck, use_container_width=True)
        else:
            st.warning("No well location data available with the current filters.")
            
    # ----------------- Right Column: Legend -----------------
    with page1_col3:
        st.subheader("Map Legend")
        
        if not filtered_df.empty:
            # Display a standalone legend (not in an expander)
            st.markdown(f"**{color_by} Legend**")
            for item in legend_items:
                st.write(f"{item['color']} {item['label']}")

    # Add a button at the bottom to navigate to page 2
    st.markdown("---")
    st.markdown("""
    <style>
        .nav-button {
            background-color: #4e8cff;
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            text-align: center;
            margin: 10px 0;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .nav-button:hover {
            background-color: #3a7ce0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    if st.button("View Well Information and Production Tables →", use_container_width=True, 
                key="nav_to_page2"):
        switch_page(2)
        
# ----------------- Page 2: Tables -----------------
elif st.session_state.page == 2:
    # Create a dropdown for well selection at the top
    if filtered_df is not None and not filtered_df.empty:
        unique_wells = filtered_df[["API_UWI", "WELLNAME"]].drop_duplicates()
        well_options = [f"{row['WELLNAME']} (API: {row['API_UWI']})" for _, row in unique_wells.iterrows()]
        well_options.insert(0, "All Wells")  # Add "All Wells" option at the beginning

        # Create a selectbox for well selection
        selected_well_option = st.selectbox("Select a well to filter production data:", well_options)

        # Extract API_UWI from selection if a specific well was chosen
        if selected_well_option != "All Wells":
            # Extract API_UWI from the string format "WELLNAME (API: API_UWI)"
            selected_well = selected_well_option.split("API: ")[1].rstrip(")")
            st.session_state.selected_well = selected_well
        else:
            st.session_state.selected_well = None

    # Create 3 columns for the three tables
    table_col1, table_col2, table_col3 = st.columns(3)

    # Create styled headers for tables
    def styled_subheader(title):
        return st.markdown(f"""
        <h3 style='color: #4e8cff; text-align: center; font-size: 1.2rem; 
        padding: 5px; border-bottom: 1px solid #4e8cff; margin-bottom: 10px;'>
            {title}
        </h3>
        """, unsafe_allow_html=True)
    
    # Well Information Table
    with table_col1:
        styled_subheader("Well Information")
        if filtered_df is not None and not filtered_df.empty:
            # If a specific well is selected, filter to just that well
            if st.session_state.selected_well:
                well_info_df = filtered_df[filtered_df["API_UWI"] == st.session_state.selected_well]
            else:
                well_info_df = filtered_df
            
            # Apply light mode styling to dataframe
            st.dataframe(well_info_df, use_container_width=True, height=300)
        else:
            st.markdown("""
            <div style="background-color: #2d3250; color: #ffa07a; padding: 10px; 
                        border-radius: 5px; text-align: center;">
                No well data available with the current filters.
            </div>
            """, unsafe_allow_html=True)

    # Monthly Production Summary Table
    with table_col2:
        styled_subheader("Monthly Production Summary")
        if filtered_secondary_df is not None and not filtered_secondary_df.empty:
            # Filter data to only include dates after January 1, 2018
            filtered_prod_df = filtered_secondary_df[filtered_secondary_df['PRODUCINGMONTH'] >= pd.Timestamp('2018-01-01')]

            # If a specific well is selected, filter the production data
            if st.session_state.selected_well:
                filtered_prod_df = filtered_prod_df[filtered_prod_df['API_UWI'] == st.session_state.selected_well]

            if not filtered_prod_df.empty:
                # Group production data by month and summarize
                monthly_summary = filtered_prod_df.groupby('PRODUCINGMONTH').agg({
                    'API_UWI': 'nunique',
                    'GASPROD_MCF': 'sum',
                    'LIQUIDSPROD_BBL': 'sum',
                    'WATERPROD_BBL': 'sum'
                }).reset_index()
                
                # Rename columns for better readability
                monthly_summary.columns = ['Production Month', 'Well Count', 'Gas (MCF)', 'Oil (BBL)', 'Water (BBL)']
                
                # Sort by month
                monthly_summary = monthly_summary.sort_values('Production Month')
                
                # Display the summary table
                st.dataframe(monthly_summary, use_container_width=True, height=300)
            else:
                st.warning("No production data available for the selected period and filters.")
        else:
            st.warning("No production data available with the current filters.")

    # Well-specific Production Data Table
    with table_col3:
        styled_subheader("Well Production Details")
        if filtered_secondary_df is not None and not filtered_secondary_df.empty:
            if st.session_state.selected_well:
                well_filtered_df = filtered_secondary_df[filtered_secondary_df['API_UWI'] == st.session_state.selected_well]
                if not well_filtered_df.empty:
                    # Show the most recent 30 months of data
                    well_filtered_df = well_filtered_df.sort_values('PRODUCINGMONTH', ascending=False).head(30)
                    
                    # Display with standard styling
                    st.dataframe(well_filtered_df, use_container_width=True, height=300)
                else:
                    st.info("No production data available for the selected well.")
            else:
                st.info("Select a specific well to view detailed production data.")
        else:
            st.markdown("""
            <div style="background-color: #2d3250; color: #ffa07a; padding: 10px; 
                        border-radius: 5px; text-align: center;">
                No production data available with the current filters.
            </div>
            """, unsafe_allow_html=True)

    # ----------------- Production Chart -----------------
    st.header("Monthly Production Chart")

    if filtered_secondary_df is not None and not filtered_secondary_df.empty:
        # Filter data to only include dates after January 1, 2018
        filtered_prod_df = filtered_secondary_df[filtered_secondary_df['PRODUCINGMONTH'] >= pd.Timestamp('2018-01-01')]
        
        # If a specific well is selected, filter the production data
        if st.session_state.selected_well:
            filtered_prod_df = filtered_prod_df[filtered_prod_df['API_UWI'] == st.session_state.selected_well]
            title_prefix = f"Well {st.session_state.selected_well} - "
        else:
            title_prefix = "All Wells - "
        
        if not filtered_prod_df.empty:
            # Group production data by month
            monthly_summary = filtered_prod_df.groupby('PRODUCINGMONTH').agg({
                'API_UWI': 'nunique',
                'GASPROD_MCF': 'sum',
                'LIQUIDSPROD_BBL': 'sum',
                'WATERPROD_BBL': 'sum'
            }).reset_index()
            
            # Rename columns for better readability
            monthly_summary.columns = ['Production Month', 'Well Count', 'Gas (MCF)', 'Oil (BBL)', 'Water (BBL)']
            
            # Sort by month
            monthly_summary = monthly_summary.sort_values('Production Month')
            
            # Add a radio button to toggle between log and linear scale
            scale_type = st.radio("Chart Scale:", ["Log Scale", "Linear Scale"], horizontal=True)
            
            # Prepare data for plotting
            plot_data = monthly_summary.melt(
                id_vars=['Production Month', 'Well Count'],
                value_vars=['Gas (MCF)', 'Oil (BBL)', 'Water (BBL)'],
                var_name='Production Type',
                value_name='Volume'
            )
            
            # Filter out any zero values to prevent log scale issues
            if scale_type == "Log Scale":
                plot_data = plot_data[plot_data['Volume'] > 0]
            
            # Create a time series plot with dynamic scale and well count on secondary axis
            # Base chart for production volumes - with dynamic scale type
            if scale_type == "Linear Scale":
                # Linear scale
                y_scale = alt.Scale(zero=True)
                y_title = 'Production Volume'
            else:
                # Log scale with minimum domain value - avoid zeros
                y_scale = alt.Scale(type='log')
                y_title = 'Production Volume (Log Scale)'
            
            # Create color mapping
            color_mapping = {'Gas (MCF)': 'red', 'Oil (BBL)': 'green', 'Water (BBL)': 'blue'}
            
            volume_chart = alt.Chart(plot_data).mark_line(point=True).encode(
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
                height=500,
                title=f"{title_prefix}Production Volume Over Time"
            )

            # Create a chart for well count
            well_count_chart = alt.Chart(monthly_summary).mark_line(
                color='black', 
                strokeDash=[5, 5],
                strokeWidth=2
            ).encode(
                x=alt.X('Production Month:T'),
                y=alt.Y('Well Count:Q', 
                      title='Well Count',
                      axis=alt.Axis(titleColor='black')),
                tooltip=['Production Month', 'Well Count']
            )

            # Add points to well count line
            well_count_points = alt.Chart(monthly_summary).mark_circle(
                color='black',
                size=50
            ).encode(
                x=alt.X('Production Month:T'),
                y=alt.Y('Well Count:Q'),
                tooltip=['Production Month', 'Well Count']
            )
            
            # Layer the charts together
            combined_chart = alt.layer(
                volume_chart,
                well_count_chart + well_count_points
            ).resolve_scale(
                y='independent'  # Use independent y scales
            ).interactive()

            # Display the combined chart
            st.altair_chart(combined_chart, use_container_width=True)
        else:
            st.warning("No production data available for the selected time period.")
    else:
        st.warning("No production data available with the current filters.")
            
    # Add a button at the bottom to navigate back to page 1
    st.markdown("---")
    if st.button("← Back to Visualizations", use_container_width=True, 
                key="nav_to_page1"):
        switch_page(1)