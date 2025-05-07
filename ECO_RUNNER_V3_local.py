# =============================================================================
# SECTION 1: APPLICATION SETUP AND INITIALIZATION
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import snowflake.snowpark.functions as F
from snowflake.snowpark import Session
import io
import base64
import requests
from bs4 import BeautifulSoup
from html.parser import HTMLParser
## delete below
import snowflake.connector
from snowflake.snowpark.session import Session

# Configure page
st.set_page_config(
    page_title="Oil & Gas Well Forecasting",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #1E3A8A;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def get_snowflake_session():
    connection_parameters = {'user':"ELII",
        'password':"Elii123456789!",
        'account':"CMZNSCB-MU47932",
        'warehouse':"COMPUTE_WH",
        'database':"WELLS",
        'schema':"MINERALS"}
    return Session.builder.configs(connection_parameters).create()

# Function to create a downloadable link for dataframes
def get_csv_download_link(df, filename="data.csv", link_text="Download CSV"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# =============================================================================
# SECTION 2: DATA ACCESS FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600)
def get_well_data():
   """Get well data from Snowflake tables"""
   # Get well data from ECON_INPUT and VW_WELL_INPUT tables
   # Updated column names to match actual Snowflake table structure
   query = """
   SELECT 
       e.API_UWI,
       w.WELLNAME,
       w.ENVBASIN,
       e.OIL_USER_QI,
       e.GAS_USER_QI,
       e.OIL_DECLINE_TYPE,
       e.GAS_DECLINE_TYPE,
       e.OIL_CALC_B_FACTOR,
       e.GAS_CALC_B_FACTOR,
       e.OIL_USER_DECLINE,
       e.GAS_USER_DECLINE,
       e.OIL_D_MIN,
       e.GAS_D_MIN,
       e.OIL_FCST_YRS,
       e.GAS_FCST_YRS,
       e.OIL_Q_MIN,
       e.GAS_Q_MIN
   FROM 
       ECON_INPUT e
   JOIN 
       VW_WELL_INPUT w ON e.API_UWI = w.API_UWI
   """
   session = get_snowflake_session()
   return session.sql(query).to_pandas()

@st.cache_data(ttl=3600)
def get_production_history():
   """Get production history from Snowflake tables"""
   # Get production history from VW_PROD_INPUT table
   query = """
   SELECT 
       API_UWI,
       PRODUCINGMONTH,
       LIQUIDSPROD_BBL,
       GASPROD_MCF,
       WATERPROD_BBL,
       TOTALPRODMONTHS
   FROM 
       VW_PROD_INPUT
   """
   session = get_snowflake_session()
   return session.sql(query).to_pandas()

@st.cache_data(ttl=3600)
def get_economic_scenarios():
   """Get economic scenarios from Snowflake tables"""
   # Get economic scenarios from ECON_SCENARIOS table
   # Updated column name to match the actual Snowflake table
   query = """
   SELECT 
       ECON_SCENARIO,
       BASIN,
       COUNTY,
       OIL_DIFF_PCT,
       OIL_DIFF_AMT,
       GAS_DIFF_PCT,
       GAS_DIFF_AMT,
       NGL_DIFF_PCT,
       NGL_DIFF_AMT,
       NGL_YIELD,
       GAS_SHRINK,
       OIL_GPT_DEDUCT,
       GAS_GPT_DEDUCT,
       NGL_GPT_DEDUCT,
       OIL_OPT_DEDUCT,
       GAS_OPT_DEDUCT,
       NGL_OPT_DEDUCT,
       OIL_TAX,
       GAS_TAX,
       NGL_TAX,
       AD_VAL_TAX
   FROM 
       ECON_SCENARIOS
   """
   session = get_snowflake_session()
   return session.sql(query).to_pandas()

@st.cache_data(ttl=3600)
def get_price_deck():
   """Get price deck data from Snowflake tables"""
   # Get price deck from PRICE_DECK table - updated to use PRICE_DECK_NAME field
   query = """
   SELECT 
       MONTH_DATE,
       OIL,
       GAS,
       PRICE_DECK_NAME
   FROM 
       PRICE_DECK
   """
   session = get_snowflake_session()
   return session.sql(query).to_pandas()

# ## html parser for gathering strip prices
# class TableParser(HTMLParser):
#     def __init__(self):
#         super().__init__()
#         self.in_table = False
#         self.in_row = False
#         self.in_cell = False
#         self.headers = []
#         self.rows = []
#         self.current_row = []
#         self.current_cell = ''

#     def handle_starttag(self, tag, attrs):
#         if tag == 'table':
#             self.in_table = True
#         elif tag == 'tr' and self.in_table:
#             self.in_row = True
#             self.current_row = []
#         elif tag in ('td', 'th') and self.in_row:
#             self.in_cell = True
#             self.current_cell = ''

#     def handle_endtag(self, tag):
#         if tag == 'table':
#             self.in_table = False
#         elif tag == 'tr' and self.in_table:
#             if self.current_row:
#                 if not self.headers:
#                     self.headers = self.current_row
#                 else:
#                     self.rows.append(self.current_row)
#             self.in_row = False
#         elif tag in ('td', 'th') and self.in_row:
#             self.in_cell = False
#             self.current_row.append(self.current_cell.strip())

#     def handle_data(self, data):
#         if self.in_cell:
#             self.current_cell += data
            
# @st.cache_data()
# def get_live_strip(url):
    # Set headers to mimic a browser visit
    # headers = {
    #     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    #                   "AppleWebKit/537.36 (KHTML, like Gecko) "
    #                   "Chrome/113.0.0.0 Safari/537.36"
    # }
    # # Send a GET request to the URL
    # response = requests.get(url, headers=headers)

    # # Check if the request was successful
    # if response.status_code == 200:
    #     parser = TableParser()
    #     parser.feed(response.text)

    #     # Build DataFrame
    #     if parser.headers and parser.rows:
    #         df = pd.DataFrame(parser.rows, columns=parser.headers)

    #         if 'Crude' in df.iloc[0]['Settlement Date']:
    #             commodity = 'OIL'
    #         else:
    #             commodity = 'GAS'

    #         df['Settlement Date'] = df['Settlement Date'].str.replace('Crude Oil ', '').str.replace('Natural Gas ', '')
    #         df['Settlement Date'] = pd.to_datetime(["1 " + d for d in df['Settlement Date']], format="%d %b %y")
    #         df['Settlement Date'] = (df['Settlement Date'] - pd.Timedelta(days=1))
    #         df['Price'] = df['Price'].str.replace(',', '').astype('float')
    #         df.sort_values(by='Settlement Date', ascending=True, inplace=True)
    #         df = df.set_index('Settlement Date')
    #         df['Price'] = df['Price'].interpolate(method='linear')
    #         df = df.reset_index().rename(columns={'index': 'Settlement Date'}).drop(columns=['Change', 'Change %']).rename(
    #             columns={'Settlement Date': 'MONTH_DATE',
    #                      'Price': f'{commodity}',
    #                      'Contract Name': 'PRICE_DECK_NAME'})
    #         return df
    #     else:
    #         print("Futures data table not found on the page.")

    # return pd.DataFrame()  # In case of error, return empty df


def get_live_strip(url):
    # Set headers to mimic a browser visit
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/113.0.0.0 Safari/537.36"
    }
    # Send a GET request to the URL
    response = requests.get(url, headers=headers)
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find the table containing the futures data
        table = soup.find('table')
        # Check if the table was found
        if table:
            # Extract table headers
            headers = [header.text.strip() for header in table.find_all('th')]
            # Extract table rows
            rows = []
            for row in table.find_all('tr')[1:]:  # Skip header row
                cells = row.find_all('td')
                if len(cells) == len(headers):
                    row_data = [cell.text.strip() for cell in cells]
                    rows.append(row_data)
            # Create a DataFrame from the extracted data
            df = pd.DataFrame(rows, columns=headers)
            if 'Crude' in df.iloc[0]['Settlement Date']:
                commodity = 'OIL'
            else:
                commodity = 'GAS'
            df['Settlement Date'] = df['Settlement Date'].str.replace('Crude Oil ', '').str.replace('Natural Gas ', '')
            df['Settlement Date'] = pd.to_datetime(["1 " + d for d in df['Settlement Date']], format="%d %b %y")
            df['Settlement Date'] = (df['Settlement Date'] - pd.Timedelta(days=1))
            df['Price'] = df['Price'].astype('float')
            df.sort_values(by='Settlement Date', ascending=True, inplace=True)
            df = df.set_index('Settlement Date')
            df['Price'] = df['Price'].interpolate(method='linear')
            df = df.reset_index().rename(columns={'index': 'Settlement Date'}).drop(columns=['Change','Change %']).rename(columns = {'Settlement Date':'MONTH_DATE',
                                                                                                                                      'Price':f'{commodity}',
                                                                                                                                      'Contract Name':'PRICE_DECK_NAME'})
        else:
            print("Futures data table not found on the page.")
    return df

@st.cache_data(ttl=3600)
def get_blended_price_deck(active_price_deck_name):
    session = get_snowflake_session()
    hist_query = """
        SELECT 
            MONTH_DATE, 
            OIL, 
            GAS,
            PRICE_DECK_NAME
        FROM 
            PRICE_DECK 
        WHERE 
            PRICE_DECK_NAME = 'HIST'
        ORDER BY 
            MONTH_DATE
        """    
    hist_data = session.sql(hist_query).to_pandas()
    if active_price_deck_name == 'Live Strip':
        oil_url = "https://finance.yahoo.com/quote/CL%3DF/futures/"
        gas_url = 'https://finance.yahoo.com/quote/NG=F/futures/'

        oil_df = get_live_strip(oil_url)
        gas_df = get_live_strip(gas_url)

        df = pd.merge(oil_df, gas_df, on = 'MONTH_DATE', how = 'outer')
        df = df.drop(columns = ['PRICE_DECK_NAME_x','PRICE_DECK_NAME_y'])
        df['PRICE_DECK_NAME'] = f'CME Strip {datetime.now().strftime("%Y%m%d%H%M")}'
    else:
        # Get the active price deck
        active_query = f"""
        SELECT 
            MONTH_DATE, 
            OIL, 
            GAS,
            PRICE_DECK_NAME
        FROM 
            PRICE_DECK 
        WHERE 
            PRICE_DECK_NAME = '{active_price_deck_name}'
        ORDER BY 
            MONTH_DATE
        """
        df = session.sql(active_query).to_pandas()

    combined_data = pd.concat([df, hist_data], axis = 0, ignore_index=True)
    combined_data['MONTH_DATE'] = pd.to_datetime(combined_data['MONTH_DATE'])
    combined_data = combined_data.drop_duplicates(subset = ['MONTH_DATE'])
    combined_data.sort_values(by='MONTH_DATE', ascending=True, inplace=True)
    combined_data = combined_data.set_index('MONTH_DATE')
    all_months = pd.date_range(start=combined_data.index.min(), end=pd.to_datetime('2075-01-01'), freq='ME')
    combined_data = combined_data.reindex(all_months)
    combined_data['OIL'] = combined_data['OIL'].interpolate(method='linear')
    combined_data['GAS'] = combined_data['GAS'].interpolate(method='linear')
    combined_data = combined_data.reset_index() 
    combined_data = combined_data.rename(columns={'index':'MONTH_DATE'})
    combined_data['PRICE_DECK_NAME'] = combined_data['PRICE_DECK_NAME'].fillna('Interpolated')
    return combined_data

@st.cache_data(ttl=3600)
def get_historical_cashflow(api_uwi, price_deck, years=2):
   """Get historical cash flow data for the past specified years"""
   # This is a new function to retrieve historical cash flow data
   # In a real implementation, this would query your database
   # For now, we'll estimate based on historical production and price data
   
   try:
       # Get historical production
       prod_history = get_production_history()
       well_history = prod_history[prod_history['API_UWI'] == api_uwi].copy()
       
       if well_history.empty:
           return pd.DataFrame()  # Return empty dataframe if no history
           
       # Convert date and sort
       well_history['PRODUCINGMONTH'] = pd.to_datetime(well_history['PRODUCINGMONTH'])
       well_history = well_history.sort_values('PRODUCINGMONTH')
       
       # Filter for last X years
       cutoff_date = datetime.now() - timedelta(days=365*years)
       well_history = well_history[well_history['PRODUCINGMONTH'] >= cutoff_date]
       
       if well_history.empty:
           return pd.DataFrame()  # Return empty dataframe if no recent history
           
       # Get price data for estimation
       price_data = get_price_deck()
    #    price_data = get_blended_price_deck(price_deck)
       price_data['MONTH_DATE'] = pd.to_datetime(price_data['MONTH_DATE'])
       
       # Create a simple price lookup by month
       price_lookup = {}
       for _, row in price_data.iterrows():
           month_key = row['MONTH_DATE'].strftime('%Y-%m')
           price_lookup[month_key] = {'OIL': row['OIL'], 'GAS': row['GAS']}
       
       # Create dataframe with revenue and cash flow estimates
       hist_cf = well_history.copy()
       hist_cf['Date'] = hist_cf['PRODUCINGMONTH']
       hist_cf['Oil_Blend'] = hist_cf['LIQUIDSPROD_BBL']
       hist_cf['Gas_Blend'] = hist_cf['GASPROD_MCF']
       
       # Apply default prices if not in lookup
       hist_cf['Oil_Price'] = 50.0  # Default price
       hist_cf['Gas_Price'] = 3.0   # Default price
       
       # Update with actual prices where available
       for idx, row in hist_cf.iterrows():
           month_key = row['Date'].strftime('%Y-%m')
           if month_key in price_lookup:
               hist_cf.at[idx, 'Oil_Price'] = price_lookup[month_key]['OIL']
               hist_cf.at[idx, 'Gas_Price'] = price_lookup[month_key]['GAS']
       
       # Estimate revenues and cash flow (simplified)
       hist_cf['Gross Revenue'] = (hist_cf['Oil_Blend'] * hist_cf['Oil_Price'] + 
                                  hist_cf['Gas_Blend'] * hist_cf['Gas_Price'] * 0.9)  # Assume 90% gas shrink
       
       # Apply simplified tax/cost assumptions (this would be more complex in real implementation)
       operating_cost_pct = 0.3  # Simplified assumption: 30% of revenue goes to operating costs
       tax_pct = 0.08          # Simplified assumption: 8% for taxes
       
       hist_cf['Net Cash Flow'] = hist_cf['Gross Revenue'] * (1 - operating_cost_pct - tax_pct)
       
       # Add Data_Source column to match forecast data
       hist_cf['Data_Source'] = 'Historical'
       
       # Calculate cumulatives
       hist_cf['Cum_Revenue'] = hist_cf['Gross Revenue'].cumsum()
       hist_cf['Cum_NCF'] = hist_cf['Net Cash Flow'].cumsum()
       
       return hist_cf
   
   except Exception as e:
       st.error(f"Error retrieving historical cash flow: {str(e)}")
       return pd.DataFrame()  # Return empty dataframe on error


# =============================================================================
# SECTION 3A: CALCULATION FUNCTIONS - DECLINE CURVES
# =============================================================================

@st.cache_data(ttl=3600)
def calculate_exponential_decline(qi, decline_rate, months):
    """Calculate production using simple exponential decline"""
    try:
        qi = float(qi)
        decline_rate = float(decline_rate)
        monthly_decline = 1 - np.exp(-decline_rate/12)
        
        # Use Python list comprehension instead of numpy operations
        rates = [qi * ((1 - monthly_decline) ** i) for i in range(months)]
        return rates
    except Exception as e:
        st.error(f"Error in exponential decline calculation: {str(e)}")
        return [qi * (0.95 ** i) for i in range(months)]  # Fallback to 5% monthly decline

@st.cache_data(ttl=3600)
def calculate_hyperbolic_decline(qi, b_factor, initial_decline, terminal_decline, months, qf=0):
    """
    Calculate production using hyperbolic decline with terminal shift to exponential.
    Returns rates and decline type indicators ('H' for hyperbolic, 'E' for exponential).
    """
    try:
        # Convert inputs to float to ensure correct calculation
        qi = float(qi)
        b_factor = float(b_factor) if b_factor is not None else 0.5
        initial_decline = float(initial_decline)  # Annual decline rate
        terminal_decline = float(terminal_decline) if terminal_decline is not None else 0.05  # Annual terminal decline rate
        qf = float(qf) if qf is not None else 0
        
        # Validate inputs to prevent calculation errors
        if b_factor <= 0:
            st.warning(f"Invalid b-factor {b_factor}, using default 0.5")
            b_factor = 0.5
        
        if initial_decline <= 0:
            st.warning(f"Invalid initial decline {initial_decline}, using default 0.1")
            initial_decline = 0.1
            
        if terminal_decline <= 0:
            st.warning(f"Invalid terminal decline {terminal_decline}, using default 0.05")
            terminal_decline = 0.05
        
        rates = []
        decline_types = []
        
        # Get monthly equivalent of terminal decline rate
        monthly_terminal = 1 - np.exp(-terminal_decline/12)
        
        # Initialize time counter
        t = 0
        
        while len(rates) < months:
            # Calculate current annual decline rate at time t
            current_decline = initial_decline / (1 + b_factor * initial_decline * t/12)
            
            # Check if we've reached terminal decline
            if current_decline <= terminal_decline:
                # Calculate rate at transition point
                transition_rate = qi / ((1 + b_factor * initial_decline * t/12) ** (1/b_factor))
                
                # Add the transition point as hyperbolic
                rates.append(transition_rate)
                decline_types.append('H')
                
                # Switch to exponential decline for remaining months
                current_rate = transition_rate
                remaining_months = months - len(rates)
                
                for i in range(remaining_months):
                    current_rate *= (1 - monthly_terminal)
                    if current_rate <= qf:
                        break
                    rates.append(current_rate)
                    decline_types.append('E')
                
                break
            
            # Still in hyperbolic decline - use formula
            current_rate = qi / ((1 + b_factor * initial_decline * t/12) ** (1/b_factor))
            
            if current_rate <= qf:
                break
                
            rates.append(current_rate)
            decline_types.append('H')
            t += 1
        
        return rates, decline_types
        
    except Exception as e:
        st.error(f"Error in hyperbolic decline calculation: {str(e)}")
        # Fallback to exponential decline if hyperbolic fails
        rates = calculate_exponential_decline(qi, initial_decline, months)
        return rates, ['E'] * len(rates)

# =============================================================================
# SECTION 3B: CALCULATION FUNCTIONS - DATABASE OPERATIONS
# =============================================================================

def save_forecast_results(forecast_df, run_info):
    session = get_snowflake_session()
    """Save forecast summary results to ECON_RESULTS table"""
    if not session:
        st.warning("Not connected to Snowflake - results not saved")
        return False
    
    try:
        # Get the effective date from run_info
        effective_date_pd = pd.to_datetime(run_info.get('effective_date'))
        
        # Filter to only include cash flows on or after effective date for PV calculations
        if effective_date_pd is not None:
            future_cashflows = forecast_df[forecast_df['Date'] >= effective_date_pd]
        else:
            future_cashflows = forecast_df

        # Create summary data for ECON_RESULTS table - updated column names to match schema
        summary_data = {
            'API_UWI': run_info['api'],
            'ECORUN_USER': 'ECO_RUNNER_main',
            'ECORUN_DEAL': run_info['deal'],
            'ECORUN_ID': run_info['id'],
            'ECORUN_DATE': datetime.now(),
            'ECORUN_SCENARIO': run_info['scenario'],
            'TOTAL_OIL_BLEND': forecast_df['Oil_Blend'].sum().round(2),  # Total from all production
            'TOTAL_GAS_BLEND': forecast_df['Gas_Blend'].sum().round(2),  # Total from all production
            'AVG_OILPRICE': forecast_df[forecast_df['Date']>=pd.to_datetime(effective_date)]['OIL'].mean(),
            'AVG_GASPRICE': forecast_df[forecast_df['Date']>=pd.to_datetime(effective_date)]['GAS'].mean(),
            'AVG_NGLPRICE': forecast_df[forecast_df['Date']>=pd.to_datetime(effective_date)]['NGL'].mean(),
            'PRICE_DECK_NAME': run_info['price_deck'],  # Changed from PRICE_DECK to PRICE_DECK_NAME
            'TOTAL_REVENUE': forecast_df['Gross Revenue'].sum().round(2),  # Total from all production    
            'TOTAL_GPT': forecast_df['GPT'].sum().round(2),              # Total from all production
            'TOTAL_SEVTAX': forecast_df['Sev Tax'].sum().round(2),       # Total from all production
            'TOTAL_ADVALTAX': forecast_df['AdVal Tax'].sum().round(2),   # Total from all production
            'PV0': future_cashflows['Net Cash Flow'].sum().round(2),  # Only future cash flows from effective date
            'TOTAL_NGL_BLEND': forecast_df['NGL'].sum().round(2),        # Total from all production
            'TOTAL_NET_CASH_FLOW': forecast_df['Net Cash Flow'].sum().round(2),  # Total from all production
        }
        # Calculate PV values for specified discount rates using only future cash flows
        pv_rates = [8, 10, 12, 14, 16, 18, 20, 22, 24]
        for rate in pv_rates:
            # Use the Years column which is already calculated relative to effective date
            discount_factor = (1 + rate/100) ** -future_cashflows['Years']
            summary_data[f'PV{rate}'] = (future_cashflows['Net Cash Flow'] * discount_factor).sum().round(2)
        
        # Convert to DataFrame
        summary_df = pd.DataFrame([summary_data])
        ## reorder to fit db table
        summary_df = summary_df[['API_UWI', 'ECORUN_USER', 'ECORUN_DEAL', 'ECORUN_ID', 'ECORUN_DATE',
       'ECORUN_SCENARIO', 'TOTAL_OIL_BLEND', 'TOTAL_GAS_BLEND', 'AVG_OILPRICE',
       'AVG_GASPRICE', 'AVG_NGLPRICE', 'TOTAL_REVENUE', 'TOTAL_GPT',
       'TOTAL_SEVTAX', 'TOTAL_ADVALTAX', 'PV0', 'PV8', 'PV10', 'PV12', 'PV14',
       'PV16', 'PV18', 'PV20', 'PV22', 'PV24', 'TOTAL_NGL_BLEND',
       'TOTAL_NET_CASH_FLOW', 'PRICE_DECK_NAME']]
        # Save to Snowflake
        session.create_dataframe(summary_df).write.mode("append").save_as_table("ECON_RESULTS")
        
        return True
    
    except Exception as e:
        st.error(f"Error saving forecast results: {str(e)}")
        return False
        
# =============================================================================
# SECTION 4: MAIN APPLICATION UI STRUCTURE
# =============================================================================

# Title and description
st.markdown('<p class="main-header">Oil & Gas Well Production Forecasting</p>', unsafe_allow_html=True)
st.markdown("""
This app helps forecast oil and gas well production using decline curve analysis methods.
You can analyze individual wells or upload data for batch processing.
""")

# Create tabs for different functionality
tab1, tab2, tab3, tab4 = st.tabs(["Single Well Forecast", "Multi-Well Analysis", "Data Import", "Documentation"])

# =============================================================================
# SECTION 5A: SINGLE WELL FORECAST - INPUTS
# =============================================================================
session = get_snowflake_session()
with tab1:
    st.markdown('<p class="sub-header">Single Well Forecast</p>', unsafe_allow_html=True)
    
    # Add run identification inputs
    st.subheader("Run Identification")
    col_run1, col_run2, col_run3 = st.columns(3)
    with col_run1:
        ecorun_deal = st.text_input("Deal Name", "Base Case")
    with col_run2:
        ecorun_id = st.text_input("Run ID", "RUN001")
    with col_run3:
        # Load economic scenarios for selection
        scenarios = get_economic_scenarios()
        scenario_names = scenarios['ECON_SCENARIO'].tolist() if not scenarios.empty else ["Base Case"]
        ecorun_scenario = st.selectbox("Economic Scenario", scenario_names)
    
    # Add forecast date settings
    st.subheader("Forecast Settings")
    col_date1, col_date2, col_date3 = st.columns(3)
    with col_date1:
        effective_date = st.date_input("Effective Date (Forecast Start)", datetime.now().date().replace(day=1))
    with col_date2:
        max_fcst_months = st.slider("Forecast Months", min_value=12, max_value=600, value=360, step=12)
    with col_date3:
        # Load price decks for selection - use PRICE_DECK_NAME 
        if session:
            price_decks = session.sql("SELECT DISTINCT PRICE_DECK_NAME FROM PRICE_DECK").to_pandas()
            price_deck_options = price_decks['PRICE_DECK_NAME'].tolist() if not price_decks.empty else ["Base Deck"]
            price_deck_options.append('Live Strip')
        else:
            price_deck_options = ["Base Deck", "High Deck", "Low Deck"]
        price_deck_options.sort()
        price_deck_options.remove('HIST')
        price_deck = st.selectbox("Price Deck", price_deck_options)
    
    # Well selection or manual parameter entry
    st.subheader("Well Parameters")
    param_source = st.radio("Parameter Source", ["Select Existing Well", "Enter Manually"])
    
    if param_source == "Select Existing Well":
        # Load well data
        wells_df = get_well_data()
        
        
        # Allow filtering by basin
        if not wells_df.empty:
            basin_options = ["All"] + sorted(wells_df['ENVBASIN'].unique().tolist())
            selected_basin = st.selectbox("Filter by Basin", basin_options)
            
            if selected_basin != "All":
                wells_df = wells_df[wells_df['ENVBASIN'] == selected_basin]
            
            # Select well
            well_options = wells_df['API_UWI'].tolist()
            selected_api = st.selectbox("Select Well", well_options)
            
            # Get selected well data
            well_data = wells_df[wells_df['API_UWI'] == selected_api].iloc[0]
            
            # Display well info
            well_name = well_data['WELLNAME']
            st.write(f"**Selected Well:** {well_name} ({selected_api}) in {well_data['ENVBASIN']} Basin")
            
            # Use well data for parameters - directly use database column names
            oil_qi = well_data['OIL_USER_QI']
            gas_qi = well_data['GAS_USER_QI']
            oil_decline_type = well_data['OIL_DECLINE_TYPE']
            gas_decline_type = well_data['GAS_DECLINE_TYPE']
            # Adjusted from aliased to direct column names
            oil_decline = well_data['OIL_USER_DECLINE']
            gas_decline = well_data['GAS_USER_DECLINE']
            oil_b_factor = well_data['OIL_CALC_B_FACTOR'] if oil_decline_type == 'H' else 0
            gas_b_factor = well_data['GAS_CALC_B_FACTOR'] if gas_decline_type == 'H' else 0
            oil_terminal_decline = well_data['OIL_D_MIN'] if oil_decline_type == 'H' else 0
            gas_terminal_decline = well_data['GAS_D_MIN'] if gas_decline_type == 'H' else 0
            oil_qf = well_data['OIL_Q_MIN']
            gas_qf = well_data['GAS_Q_MIN']
            oil_yrs = well_data['OIL_FCST_YRS']
            gas_yrs = well_data['GAS_FCST_YRS']
            
            # Show production history if available
            prod_df = get_production_history()
            well_hist = prod_df[prod_df['API_UWI'] == selected_api]
            
            if not well_hist.empty:
                st.write("**Recent Production History:**")
                recent_hist = well_hist.sort_values('PRODUCINGMONTH', ascending=False).head(6)
                recent_hist = recent_hist[['PRODUCINGMONTH', 'LIQUIDSPROD_BBL', 'GASPROD_MCF']]
                st.dataframe(recent_hist)
        else:
            st.error("No well data available. Please use manual entry.")
            param_source = "Enter Manually"
    
    if param_source == "Enter Manually":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Oil Parameters")
            oil_qi = st.number_input("Initial Oil Rate (BBL/day)", min_value=0.0, value=100.0, step=10.0)
            oil_decline_type = st.selectbox("Oil Decline Type", ["E", "H"], index=1)
            oil_decline = st.number_input("Oil Decline Rate (annual)", min_value=0.01, max_value=0.99, value=0.3, step=0.05)
            
            if oil_decline_type == "H":
                oil_b_factor = st.number_input("Oil B-Factor", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
                oil_terminal_decline = st.number_input("Oil Terminal Decline Rate (annual)", min_value=0.01, max_value=0.5, value=0.05, step=0.01)
            else:
                oil_b_factor = 0
                oil_terminal_decline = 0
                
            oil_qf = st.number_input("Oil Economic Limit (BBL/day)", min_value=0.0, value=5.0, step=1.0)
            oil_yrs = st.number_input("Oil Years Remaining (Exponential)", min_value=1.0, value=30.0, step=1.0)
            selected_api = "MANUAL"
            well_name = "Manual Entry"
        
        with col2:
            st.subheader("Gas Parameters")
            gas_qi = st.number_input("Initial Gas Rate (MCF/day)", min_value=0.0, value=500.0, step=50.0)
            gas_decline_type = st.selectbox("Gas Decline Type", ["E", "H"], index=1)
            gas_decline = st.number_input("Gas Decline Rate (annual)", min_value=0.01, max_value=0.99, value=0.4, step=0.05)
            
            if gas_decline_type == "H":
                gas_b_factor = st.number_input("Gas B-Factor", min_value=0.1, max_value=2.0, value=0.8, step=0.1)
                gas_terminal_decline = st.number_input("Gas Terminal Decline Rate (annual)", min_value=0.01, max_value=0.5, value=0.08, step=0.01)
            else:
                gas_b_factor = 0
                gas_terminal_decline = 0
                
            gas_qf = st.number_input("Gas Economic Limit (MCF/day)", min_value=0.0, value=50.0, step=10.0)
            gas_yrs = st.number_input("Gas Years Remaining (Exponential)", min_value=1.0, value=30.0, step=1.0)
    
    # Get economic parameters based on selected scenario
    if not scenarios.empty:
        
        scenario_data = scenarios[scenarios['ECON_SCENARIO'] == ecorun_scenario].iloc[0]
        
        # Using direct column names based on the ECON_SCENARIOS table structure
        # Explicitly convert to float to avoid non-numeric issues
        # Added NaN checks to use default values when NaN is detected
        oil_basis_pct = float(scenario_data['OIL_DIFF_PCT']) if not pd.isna(scenario_data['OIL_DIFF_PCT']) else 1.0
        oil_basis_amt = float(scenario_data['OIL_DIFF_AMT']) if not pd.isna(scenario_data['OIL_DIFF_AMT']) else 0.0
        gas_basis_pct = float(scenario_data['GAS_DIFF_PCT']) if not pd.isna(scenario_data['GAS_DIFF_PCT']) else 1.0
        gas_basis_amt = float(scenario_data['GAS_DIFF_AMT']) if not pd.isna(scenario_data['GAS_DIFF_AMT']) else 0.0
        ngl_basis_pct = float(scenario_data['NGL_DIFF_PCT']) if not pd.isna(scenario_data['NGL_DIFF_PCT']) else 0.3
        ngl_basis_amt = float(scenario_data['NGL_DIFF_AMT']) if not pd.isna(scenario_data['NGL_DIFF_AMT']) else 0.0
        ngl_yield = float(scenario_data['NGL_YIELD']) if not pd.isna(scenario_data['NGL_YIELD']) else 10.0
        gas_shrink = float(scenario_data['GAS_SHRINK']) if not pd.isna(scenario_data['GAS_SHRINK']) else 0.9
        oil_gpt = float(scenario_data['OIL_GPT_DEDUCT']) if not pd.isna(scenario_data['OIL_GPT_DEDUCT']) else 0.0
        gas_gpt = float(scenario_data['GAS_GPT_DEDUCT']) if not pd.isna(scenario_data['GAS_GPT_DEDUCT']) else 0.0
        ngl_gpt = float(scenario_data['NGL_GPT_DEDUCT']) if not pd.isna(scenario_data['NGL_GPT_DEDUCT']) else 0.0
        oil_tax = float(scenario_data['OIL_TAX']) if not pd.isna(scenario_data['OIL_TAX']) else 0.046
        gas_tax = float(scenario_data['GAS_TAX']) if not pd.isna(scenario_data['GAS_TAX']) else 0.075
        ngl_tax = float(scenario_data['NGL_TAX']) if not pd.isna(scenario_data['NGL_TAX']) else 0.046
        ad_val_tax = float(scenario_data['AD_VAL_TAX']) if not pd.isna(scenario_data['AD_VAL_TAX']) else 0.02
        
    else:
        # Default values if no scenarios are available
        oil_basis_pct = 1.0
        oil_basis_amt = 0.0
        gas_basis_pct = 1.0
        gas_basis_amt = 0.0
        ngl_basis_pct = 0.3
        ngl_basis_amt = 0.0
        ngl_yield = 10.0
        gas_shrink = 0.9
        oil_gpt = 0.0
        gas_gpt = 0.0
        ngl_gpt = 0.0
        oil_tax = 0.046
        gas_tax = 0.075
        ngl_tax = 0.046
        ad_val_tax = 0.02
    
    # Show economic parameters
    with st.expander("Economic Parameters"):
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            st.write("**Price Adjustments:**")
            st.write(f"Oil: {oil_basis_pct:.2f}x + ${oil_basis_amt:.2f}/BBL")
            st.write(f"Gas: {gas_basis_pct:.2f}x + ${gas_basis_amt:.2f}/MCF")
            st.write(f"NGL: {ngl_basis_pct:.2f}x + ${ngl_basis_amt:.2f}/BBL")
        
        with ec2:
            st.write("**Production:**")
            st.write(f"NGL Yield: {ngl_yield:.2f} BBL/MMCF")
            st.write(f"Gas Shrink: {gas_shrink:.2f}")
        
        with ec3:
            st.write("**Taxes:**")
            st.write(f"Oil Sev. Tax: {oil_tax*100:.2f}%")
            st.write(f"Gas Sev. Tax: {gas_tax*100:.2f}%")
            st.write(f"Ad Valorem: {ad_val_tax*100:.2f}%")
            
# =============================================================================
# SECTION 5B: SINGLE WELL FORECAST - CALCULATION & RESULTS
# =============================================================================

    # Create a container for the forecast data so we can access it throughout the tab
    if 'forecast_df' not in st.session_state:
        st.session_state.forecast_df = None
        st.session_state.blended_df = None
        st.session_state.pv_results = None
        st.session_state.well_name = None
        st.session_state.pv_rates = None

    # Forecast button
    if st.button("Generate Forecast", key="forecast_btn"):
        prod_history = get_production_history()
        well_history = prod_history[prod_history['API_UWI'] == selected_api].copy()
        well_history['PRODUCINGMONTH'] = pd.to_datetime(well_history['PRODUCINGMONTH'])
        # Create date range
        # date_range = pd.date_range(start=effective_date, periods=max_fcst_months, freq='M') ##delete
        date_range = pd.date_range(start=well_history['PRODUCINGMONTH'].max(), periods=max_fcst_months, freq='M')
        
        # Calculate oil forecast
        if oil_decline_type == "H":
            oil_rates, oil_decline_types = calculate_hyperbolic_decline(
                qi=oil_qi, 
                b_factor=oil_b_factor, 
                initial_decline=oil_decline, 
                terminal_decline=oil_terminal_decline, 
                months=max_fcst_months,
                qf=oil_qf
            )
        else:
            oil_months = min(int(oil_yrs * 12), max_fcst_months)
            oil_rates = calculate_exponential_decline(oil_qi, oil_decline, oil_months)
            oil_decline_types = ['E'] * len(oil_rates)
            # Pad with zeros if oil forecast is shorter than the requested period
            if len(oil_rates) < max_fcst_months:
                oil_rates = oil_rates + [0] * (max_fcst_months - len(oil_rates))
                oil_decline_types = oil_decline_types + [''] * (max_fcst_months - len(oil_decline_types))
        
        # Calculate gas forecast
        if gas_decline_type == "H":
            gas_rates, gas_decline_types = calculate_hyperbolic_decline(
                qi=gas_qi, 
                b_factor=gas_b_factor, 
                initial_decline=gas_decline, 
                terminal_decline=gas_terminal_decline, 
                months=max_fcst_months,
                qf=gas_qf
            )
        else:
            gas_months = min(int(gas_yrs * 12), max_fcst_months)
            gas_rates = calculate_exponential_decline(gas_qi, gas_decline, gas_months)
            gas_decline_types = ['E'] * len(gas_rates)
            # Pad with zeros if gas forecast is shorter than the requested period
            if len(gas_rates) < max_fcst_months:
                gas_rates = gas_rates + [0] * (max_fcst_months - len(gas_rates))
                gas_decline_types = gas_decline_types + [''] * (max_fcst_months - len(gas_decline_types))
        
        # Create the forecast dataframe
        forecast_df = pd.DataFrame({
            'Date': date_range,
            'Oil_Blend': oil_rates[:len(date_range)],
            'Gas_Blend': gas_rates[:len(date_range)],
            'Oil_Decline_Type': oil_decline_types[:len(date_range)] if len(oil_decline_types) >= len(date_range) else oil_decline_types + [''] * (len(date_range) - len(oil_decline_types)),
            'Gas_Decline_Type': gas_decline_types[:len(date_range)] if len(gas_decline_types) >= len(date_range) else gas_decline_types + [''] * (len(date_range) - len(gas_decline_types))
        })
        
        # Get price data - Updated to use PRICE_DECK_NAME instead of PRICE_DECK_ID
        price_query = f"""
        SELECT 
            MONTH_DATE, 
            OIL, 
            GAS 
        FROM 
            PRICE_DECK 
        WHERE 
            PRICE_DECK_NAME = '{price_deck}'
        ORDER BY 
            MONTH_DATE
        """
        # price_data = session.sql(price_query).to_pandas()
        price_data = get_blended_price_deck(price_deck)
    
        # Check if price data is empty
        if price_data.empty:
            st.error(f"No price data found for price deck '{price_deck}'")
            # Use default values for prices
            st.warning("Using default price values: $50/BBL for oil, $3/MCF for gas")
            forecast_df['OIL'] = 50.0
            forecast_df['GAS'] = 3.0
        else:
            # Merge price data with forecast
            # Convert both date columns to datetime with standardized format
            forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
            price_data['MONTH_DATE'] = pd.to_datetime(price_data['MONTH_DATE'])
            
            # Create standardized date keys for merging (YYYY-MM-01 format)
            forecast_df['Date_Key'] = forecast_df['Date'].dt.strftime('%Y-%m-01')
            price_data['Date_Key'] = price_data['MONTH_DATE'].dt.strftime('%Y-%m-01')
            # Merge on the standardized date key
            forecast_df = forecast_df.merge(price_data, left_on='Date_Key', right_on='Date_Key', how='left')
            forecast_df.drop('Date_Key', axis=1, inplace=True)
            # Fill missing price data with default values if any rows didn't match
            forecast_df['OIL'] = forecast_df['OIL'].fillna(50.0)  # Default oil price
            forecast_df['GAS'] = forecast_df['GAS'].fillna(3.0)   # Default gas price

        # Calculate NGL, though we won't display it in the chart
        forecast_df['NGL'] = ((forecast_df['Gas_Blend']/1000) * ngl_yield).round(2)
        
        # Ensure data types are numeric before calculation
        forecast_df['OIL'] = pd.to_numeric(forecast_df['OIL'], errors='coerce').fillna(50.0)
        forecast_df['GAS'] = pd.to_numeric(forecast_df['GAS'], errors='coerce').fillna(3.0)

        # Verify economic parameters are valid floats and not NaN
        oil_basis_pct = float(oil_basis_pct) if not pd.isna(oil_basis_pct) else 1.0
        oil_basis_amt = float(oil_basis_amt) if not pd.isna(oil_basis_amt) else 0.0
        gas_basis_pct = float(gas_basis_pct) if not pd.isna(gas_basis_pct) else 1.0
        gas_basis_amt = float(gas_basis_amt) if not pd.isna(gas_basis_amt) else 0.0
        ngl_basis_pct = float(ngl_basis_pct) if not pd.isna(ngl_basis_pct) else 0.3
        ngl_basis_amt = float(ngl_basis_amt) if not pd.isna(ngl_basis_amt) else 0.0
        
        # Calculate revenue and cash flow with safeguards against NaN values
        forecast_df['Real Oil'] = ((forecast_df['OIL'] * oil_basis_pct) + oil_basis_amt).round(2)
        forecast_df['Real Gas'] = ((forecast_df['GAS'] * gas_basis_pct) + gas_basis_amt).round(2)
        forecast_df['Real NGL'] = ((forecast_df['OIL'] * ngl_basis_pct) + ngl_basis_amt).round(2)
        
        # Calculate revenue with checks to prevent NaN propagation
        forecast_df['Oil_Revenue'] = forecast_df['Oil_Blend'] * forecast_df['Real Oil'].fillna(forecast_df['OIL'])
        forecast_df['Gas_Revenue'] = forecast_df['Gas_Blend'] * forecast_df['Real Gas'].fillna(forecast_df['GAS']) * gas_shrink
        forecast_df['NGL_Revenue'] = forecast_df['NGL'] * forecast_df['Real NGL'].fillna(forecast_df['OIL'] * 0.3)
        forecast_df['Gross Revenue'] = (forecast_df['Oil_Revenue'] + forecast_df['Gas_Revenue'] + forecast_df['NGL_Revenue']).round(2)
        
        forecast_df['Oil_GPT'] = forecast_df['Oil_Blend'] * oil_gpt
        forecast_df['Gas_GPT'] = forecast_df['Gas_Blend'] * gas_gpt
        forecast_df['NGL_GPT'] = forecast_df['NGL'] * ngl_gpt
        forecast_df['GPT'] = (forecast_df['Oil_GPT'] + forecast_df['Gas_GPT'] + forecast_df['NGL_GPT']).round(2)
        
        forecast_df['Oil_Sev'] = forecast_df['Oil_Revenue'] * oil_tax
        forecast_df['Gas_Sev'] = forecast_df['Gas_Revenue'] * gas_tax
        forecast_df['NGL_Sev'] = forecast_df['NGL_Revenue'] * ngl_tax
        forecast_df['Sev Tax'] = (forecast_df['Oil_Sev'] + forecast_df['Gas_Sev'] + forecast_df['NGL_Sev']).round(2)
        
        forecast_df['AdVal Tax'] = (forecast_df['Gross Revenue'] * ad_val_tax).round(2)
        forecast_df['Net Cash Flow'] = (forecast_df['Gross Revenue'] - forecast_df['GPT'] - forecast_df['Sev Tax'] - forecast_df['AdVal Tax']).round(2)
        
        # Calculate cumulative production and revenue
        forecast_df['Cum_Oil'] = forecast_df['Oil_Blend'].cumsum()
        forecast_df['Cum_Gas'] = forecast_df['Gas_Blend'].cumsum()
        forecast_df['Cum_NGL'] = forecast_df['NGL'].cumsum()
        forecast_df['Cum_Revenue'] = forecast_df['Gross Revenue'].cumsum()
        forecast_df['Cum_NCF'] = forecast_df['Net Cash Flow'].cumsum()
        
        # Calculate years from effective date for PV calculations
        forecast_df['Years'] = ((forecast_df['Date'] - pd.to_datetime(effective_date)).dt.days / 365.25)
        
        # Add 'Year' column for yearly aggregation
        forecast_df['Year'] = forecast_df['Date'].dt.year
        
        # Store the forecast data in session state
        st.session_state.forecast_df = forecast_df
        
        # Blend with historical data if well is selected from database
        if param_source == "Select Existing Well":
            # # Load production history
            # prod_history = get_production_history()
            # well_history = prod_history[prod_history['API_UWI'] == selected_api].copy()##delete
            
            if not well_history.empty:
                # Process historical data
                # well_history['PRODUCINGMONTH'] = pd.to_datetime(well_history['PRODUCINGMONTH'])##delete
                well_history = well_history.sort_values('PRODUCINGMONTH')
                well_history['Date'] = well_history['PRODUCINGMONTH'].apply(lambda x: x + pd.offsets.MonthEnd(0))
                
                # Get last historical date and set up effective date
                last_hist_date = well_history['Date'].max()
                effective_date_pd = pd.to_datetime(effective_date)
                
                # Create a complete timeline - ALL months from earliest history to end of forecast
                all_dates = pd.date_range(
                    start=well_history['Date'].min(),
                    end=forecast_df['Date'].max(),
                    freq='M'
                )
                
                # Create the complete timeline dataframe
                timeline_df = pd.DataFrame({'Date': all_dates})
                
                # Add Data_Source column based on date ranges
                timeline_df['Data_Source'] = 'Forecast'  # Default
                
                # Mark historical dates
                historical_dates = well_history['Date'].unique()
                timeline_df.loc[timeline_df['Date'].isin(historical_dates), 'Data_Source'] = 'Historical'
                
                # Mark gap period (after last historical, before effective date)
                if last_hist_date < effective_date_pd:
                    gap_mask = (timeline_df['Date'] > last_hist_date) & (timeline_df['Date'] < effective_date_pd)
                    timeline_df.loc[gap_mask, 'Data_Source'] = 'Gap'
                
                # Merge the timeline with production data sources
                
                # 1. Create a historical dataframe with required columns
                hist_df = pd.DataFrame()
                hist_df['Date'] = well_history['Date']
                hist_df['Oil_Blend'] = well_history['LIQUIDSPROD_BBL']
                hist_df['Gas_Blend'] = well_history['GASPROD_MCF']
                hist_df['Data_Source'] = 'Historical'
                
                # 2. Create a modified forecast dataframe
                # Make sure forecast dates don't overlap with historical data
                fcst_df = forecast_df.copy()
                
                # 3. CRITICAL: Create entries for gap period
                if last_hist_date < effective_date_pd:
                    # Calculate the months in the gap
                    gap_months = pd.date_range(
                        start=last_hist_date + pd.DateOffset(months=1),
                        end=effective_date_pd,
                        freq='M'
                    )
                    
                    if len(gap_months) > 0:
                        # Create a gap dataframe by using a slice of the forecast that covers this period
                        gap_df = fcst_df[fcst_df['Date'].isin(gap_months)].copy()
                        
                        # If some months are missing from the forecast, we need to create them
                        if len(gap_df) < len(gap_months):
                            missing_dates = [date for date in gap_months if date not in gap_df['Date'].values]
                            
                            # For each missing date, create a new row by interpolating
                            for gap_date in missing_dates:
                                # Get productions from the forecast at effective date as a starting point
                                ref_row = fcst_df[fcst_df['Date'] == effective_date_pd]
                                
                                if not ref_row.empty:
                                    # Copy data from the reference row
                                    new_row = ref_row.copy()
                                    new_row['Date'] = gap_date  # Update date
                                    
                                    # Add to gap_df
                                    gap_df = pd.concat([gap_df, new_row], ignore_index=True)
                        
                        # Mark these as gap period
                        gap_df['Data_Source'] = 'Gap'
                        
                        # Get prices for interpolation
                        # Get historical prices (HIST price deck)
                        hist_query = """
                        SELECT 
                            MONTH_DATE, 
                            OIL, 
                            GAS,
                            PRICE_DECK_NAME
                        FROM 
                            PRICE_DECK 
                        WHERE 
                            PRICE_DECK_NAME = 'HIST'
                        ORDER BY 
                            MONTH_DATE
                        """
                        hist_price_data = session.sql(hist_query).to_pandas()
                        hist_price_data['MONTH_DATE'] = pd.to_datetime(hist_price_data['MONTH_DATE']).apply(lambda x: x + pd.offsets.MonthEnd(0))
                        
                        # Get active price deck
                        active_query = f"""
                        SELECT 
                            MONTH_DATE, 
                            OIL, 
                            GAS,
                            PRICE_DECK_NAME
                        FROM 
                            PRICE_DECK 
                        WHERE 
                            PRICE_DECK_NAME = '{price_deck}'
                        ORDER BY 
                            MONTH_DATE
                        """
                        active_price_data = session.sql(active_query).to_pandas()
                        active_price_data['MONTH_DATE'] = pd.to_datetime(active_price_data['MONTH_DATE']).apply(lambda x: x + pd.offsets.MonthEnd(0))
                        
                        # Get the last historical price
                        hist_prices = hist_price_data[hist_price_data['MONTH_DATE'] <= last_hist_date]
                        if not hist_prices.empty:
                            last_hist_price = hist_prices.iloc[-1]
                            last_hist_oil_price = last_hist_price['OIL']
                            last_hist_gas_price = last_hist_price['GAS']
                        else:
                            last_hist_oil_price = 50.0
                            last_hist_gas_price = 3.0
                        
                        # Get the first active price
                        active_prices = active_price_data[active_price_data['MONTH_DATE'] >= effective_date_pd]
                        if not active_prices.empty:
                            first_active_price = active_prices.iloc[0]
                            first_active_oil_price = first_active_price['OIL']
                            first_active_gas_price = first_active_price['GAS']
                        else:
                            first_active_oil_price = 50.0
                            first_active_gas_price = 3.0
                        
                        # Interpolate prices for each gap month
                        gap_df = gap_df.sort_values('Date').reset_index(drop=True)
                        
                        for idx, row in gap_df.iterrows():
                            gap_date = row['Date']
                            
                            # Calculate position in gap as a factor (0 to 1)
                            total_days = (effective_date_pd - last_hist_date).days
                            position_days = (gap_date - last_hist_date).days
                            position_factor = position_days / total_days if total_days > 0 else 0
                            
                            # Interpolate prices
                            oil_price = last_hist_oil_price + (first_active_oil_price - last_hist_oil_price) * position_factor
                            gas_price = last_hist_gas_price + (first_active_gas_price - last_hist_gas_price) * position_factor
                            
                            # Set prices in the gap dataframe
                            gap_df.loc[idx, 'OIL'] = oil_price
                            gap_df.loc[idx, 'GAS'] = gas_price
                    else:
                        gap_df = pd.DataFrame()  # No gap months
                else:
                    gap_df = pd.DataFrame()  # No gap period
                
                # 4. Now combine all three parts
                # Make sure forecast doesn't include gap period

                fcst_after_effective = fcst_df[fcst_df['Date'] > effective_date_pd].copy()
                ##make sure forecast go backwards - elii
                # fcst_after_effective = fcst_df[fcst_df['Date'] > hist_df['Date'].max()].copy()

                fcst_after_effective['Data_Source'] = 'Forecast'
                # Combine all dataframes
                combined_parts = [hist_df]
                if not gap_df.empty:
                    combined_parts.append(gap_df)
                combined_parts.append(fcst_after_effective)
                
                # Concatenate all parts
                blended_df = pd.concat(combined_parts, ignore_index=True)
                col_order = blended_df.columns
                blended_df = pd.merge(blended_df, 
                                      price_data, 
                                      left_on = 'Date', 
                                      right_on='MONTH_DATE', 
                                      how='left').drop(columns=['MONTH_DATE_x',
                                      'OIL_x',
                                      'GAS_x']).rename(columns = {'MONTH_DATE_y':'MONTH_DATE',
                                                                  'PRICE_DECK_NAME_y':'PRICE_DECK_NAME',
                                                                  'OIL_y':'OIL',
                                                                  'GAS_y':'GAS'})
                blended_df = blended_df[col_order]
                # Sort by date
                blended_df = blended_df.sort_values('Date').reset_index(drop=True)
                
                # Get full price deck for any missing prices
                price_data = get_blended_price_deck(price_deck)
                
                # Ensure all rows have prices
                missing_prices = blended_df[~blended_df['Date'].isin(gap_df['Date'])]
                if 'OIL' not in missing_prices.columns:
                    # Merge with price deck
                    price_columns = ['MONTH_DATE', 'OIL', 'GAS']
                    if not all(col in price_data.columns for col in price_columns):
                        # Create default prices if price deck doesn't have expected columns
                        missing_prices['OIL'] = 50.0
                        missing_prices['GAS'] = 3.0
                    else:
                        # Perform the merge with price deck
                        missing_prices = missing_prices.merge(
                            price_data[price_columns],
                            left_on='Date',
                            right_on='MONTH_DATE',
                            how='left'
                        )
                        
                        # Clean up after merge
                        if 'MONTH_DATE' in missing_prices.columns:
                            missing_prices.drop('MONTH_DATE', axis=1, inplace=True)
                    
                    # Replace in blended_df
                    blended_df = pd.concat([
                        missing_prices,
                        blended_df[blended_df['Date'].isin(gap_df['Date'])],
                    ], ignore_index=True)
                    
                    # Re-sort
                    blended_df = blended_df.sort_values('Date').reset_index(drop=True)
                
                # Fill any missing price data with defaults
                blended_df['OIL'] = blended_df['OIL'].fillna(50.0)
                blended_df['GAS'] = blended_df['GAS'].fillna(3.0)
                
                # Calculate NGL yield for all data
                if 'NGL' not in blended_df.columns:
                    blended_df['NGL'] = ((blended_df['Gas_Blend']/1000) * ngl_yield).round(2)
                
                # Create economic parameters dict
                economic_params = {
                    'oil_basis_pct': oil_basis_pct,
                    'oil_basis_amt': oil_basis_amt,
                    'gas_basis_pct': gas_basis_pct,
                    'gas_basis_amt': gas_basis_amt,
                    'ngl_basis_pct': ngl_basis_pct,
                    'ngl_basis_amt': ngl_basis_amt,
                    'ngl_yield': ngl_yield,
                    'gas_shrink': gas_shrink,
                    'oil_gpt': oil_gpt,
                    'gas_gpt': gas_gpt,
                    'ngl_gpt': ngl_gpt,
                    'oil_tax': oil_tax,
                    'gas_tax': gas_tax,
                    'ngl_tax': ngl_tax,
                    'ad_val_tax': ad_val_tax
                }
                
                # Calculate economics
                blended_df['Real Oil'] = ((blended_df['OIL'] * oil_basis_pct) + oil_basis_amt).round(2)
                blended_df['Real Gas'] = ((blended_df['GAS'] * gas_basis_pct) + gas_basis_amt).round(2)
                blended_df['Real NGL'] = ((blended_df['OIL'] * ngl_basis_pct) + ngl_basis_amt).round(2)
                
                blended_df['Oil_Revenue'] = blended_df['Oil_Blend'] * blended_df['Real Oil']
                blended_df['Gas_Revenue'] = blended_df['Gas_Blend'] * blended_df['Real Gas'] * gas_shrink
                blended_df['NGL_Revenue'] = blended_df['NGL'] * blended_df['Real NGL']
                blended_df['Gross Revenue'] = (blended_df['Oil_Revenue'] + blended_df['Gas_Revenue'] + blended_df['NGL_Revenue']).round(2)
                
                blended_df['Oil_GPT'] = blended_df['Oil_Blend'] * oil_gpt
                blended_df['Gas_GPT'] = blended_df['Gas_Blend'] * gas_gpt
                blended_df['NGL_GPT'] = blended_df['NGL'] * ngl_gpt
                blended_df['GPT'] = (blended_df['Oil_GPT'] + blended_df['Gas_GPT'] + blended_df['NGL_GPT']).round(2)
                
                blended_df['Oil_Sev'] = blended_df['Oil_Revenue'] * oil_tax
                blended_df['Gas_Sev'] = blended_df['Gas_Revenue'] * gas_tax
                blended_df['NGL_Sev'] = blended_df['NGL_Revenue'] * ngl_tax
                blended_df['Sev Tax'] = (blended_df['Oil_Sev'] + blended_df['Gas_Sev'] + blended_df['NGL_Sev']).round(2)
                
                blended_df['AdVal Tax'] = (blended_df['Gross Revenue'] * ad_val_tax).round(2)
                blended_df['Net Cash Flow'] = (blended_df['Gross Revenue'] - blended_df['GPT'] - blended_df['Sev Tax'] - blended_df['AdVal Tax']).round(2)
                
                # Calculate cumulative values
                blended_df['Cum_Oil'] = blended_df['Oil_Blend'].cumsum()
                blended_df['Cum_Gas'] = blended_df['Gas_Blend'].cumsum()
                blended_df['Cum_NGL'] = blended_df['NGL'].cumsum()
                blended_df['Cum_Revenue'] = blended_df['Gross Revenue'].cumsum()
                blended_df['Cum_NCF'] = blended_df['Net Cash Flow'].cumsum()
                
                # Calculate years from effective date for PV calculations
                blended_df['Years'] = ((blended_df['Date'] - pd.to_datetime(effective_date)).dt.days / 365.25)
                
                # Add Year column for aggregation
                blended_df['Year'] = blended_df['Date'].dt.year
                
                # Store the blended dataset
                st.session_state.blended_df = blended_df
            else:
                # No historical data, use the forecast only
                forecast_df['Data_Source'] = 'Forecast'
                st.session_state.blended_df = forecast_df
        else:
            # For manual entry, no historical data is available
            forecast_df['Data_Source'] = 'Forecast'
            st.session_state.blended_df = forecast_df

        # Store the well name in session state
        st.session_state.well_name = well_name
        
        # Calculate PV values and store in session state
        pv_rates = [0, 8, 10, 12, 14, 16, 18, 20, 22, 24]
        st.session_state.pv_rates = pv_rates
        
        pv_results = {}
        effective_date_pd = pd.to_datetime(effective_date)
        # Filter to only include cash flows on or after effective date
        future_cashflows = forecast_df[forecast_df['Date'] >= effective_date_pd]
        
        for rate in pv_rates:
            if rate == 0:
                # For PV0, simply sum the future cash flows
                pv_results[f'PV{rate}'] = future_cashflows['Net Cash Flow'].sum()
            else:
                # Use the Years column which is already relative to effective date
                discount_factor = (1 + rate/100) ** -future_cashflows['Years']
                pv_results[f'PV{rate}'] = (future_cashflows['Net Cash Flow'] * discount_factor).sum()
        
        st.session_state.pv_results = pv_results
                        
# =============================================================================
# SECTION 5C: SINGLE WELL FORECAST - VISUALIZATION & DISPLAY
# =============================================================================

    # Only display results if forecast has been generated
    if st.session_state.forecast_df is not None:
        forecast_df = st.session_state.forecast_df  # Get the stored forecast
        blended_df = st.session_state.blended_df    # Get the blended data
        well_name = st.session_state.well_name      # Get the stored well name
        pv_results = st.session_state.pv_results    # Get the stored PV results
        pv_rates = st.session_state.pv_rates        # Get the stored PV rates
        
        # Save results if requested
        run_info = {
            'deal': ecorun_deal,
            'id': ecorun_id,
            'scenario': ecorun_scenario,
            'price_deck': price_deck,  # This is now PRICE_DECK_NAME instead of PRICE_DECK_ID
            'api': selected_api,
            'effective_date': effective_date  # Add this line to pass the effective date
        }
        
        if st.checkbox("Save Results to Database", value=True):
            save_success = save_forecast_results(forecast_df, run_info)
            if save_success:
                st.success(f"Results for {well_name} saved to ECON_RESULTS table")
        
        # Display metrics
        st.markdown('<p class="sub-header">Forecast Summary</p>', unsafe_allow_html=True)
        
        metric1, metric2, metric3, metric4 = st.columns(4)
        with metric1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Oil (BBL)", f"{forecast_df['Oil_Blend'].sum():,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with metric2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Gas (MCF)", f"{forecast_df['Gas_Blend'].sum():,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with metric3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total NGL (BBL)", f"{forecast_df['NGL'].sum():,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with metric4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Revenue ($)", f"${forecast_df['Gross Revenue'].sum():,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display PV values in two rows
        st.markdown("### Present Value Analysis")
        pv_row1 = st.columns(5)
        pv_row2 = st.columns(5)
        
        for i, rate in enumerate(pv_rates[:5]):
            with pv_row1[i]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(f"PV{rate} ($)", f"${pv_results[f'PV{rate}']:,.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
        for i, rate in enumerate(pv_rates[5:]):
            with pv_row2[i]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(f"PV{rate} ($)", f"${pv_results[f'PV{rate}']:,.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Create production plots
        st.markdown("### Production Forecast")
        
        # Add options for chart customization
        prod_options_col1, prod_options_col2, prod_options_col3 = st.columns(3)
        
        with prod_options_col1:
            # Option to toggle between blended view and forecast-only view
            use_blended_view = st.checkbox("Show Historical Production", value=True, key="use_blended_view")
        
        with prod_options_col2:
            # Calculate the appropriate starting point for the months to display slider
            if use_blended_view and 'Data_Source' in blended_df.columns:
                historical_data = blended_df[blended_df['Data_Source'] == 'Historical']
                if not historical_data.empty:
                    last_hist_date = historical_data['Date'].max()
                    forecast_data = blended_df[blended_df['Date'] > last_hist_date]
                    months_since_last_hist = len(forecast_data)
                else:
                    months_since_last_hist = len(blended_df)
            else:
                months_since_last_hist = len(forecast_df)
            
            # Option to select custom time range, starting from the last historical date
            max_months = months_since_last_hist
            display_months = st.slider(
                "Forecast Months to Display",
                min_value=12,
                max_value=min(360, max_months),
                value=min(120, max_months),
                step=12,
                key="prod_display_months"
            )
            
        with prod_options_col3:
            # Add option for log scale Y-axis (common in oil & gas decline curves)
            use_log_scale = st.checkbox("Log Scale Y-Axis", value=True, key="use_log_scale")
        
        # Choose dataset based on user preference
        if use_blended_view:
            # If using blended view, we need to include all historical data plus the selected number of forecast months
            if 'Data_Source' in blended_df.columns:
                historical_data = blended_df[blended_df['Data_Source'] == 'Historical']
                forecast_data = blended_df[blended_df['Data_Source'] != 'Historical']
                
                if not historical_data.empty and not forecast_data.empty:
                    # Include all historical data plus selected forecast months
                    plot_data = pd.concat([
                        historical_data,
                        forecast_data.head(display_months)
                    ])
                else:
                    plot_data = blended_df.head(display_months)
            else:
                plot_data = blended_df.head(display_months)
        else:
            plot_data = forecast_df.head(display_months)
            
        # Create production plots with color-coded historical vs forecast data
        fig1 = go.Figure()
        # Split data into historical and forecast parts for color coding if using blended view
        if use_blended_view and 'Data_Source' in plot_data.columns:
            historical_data = plot_data[plot_data['Data_Source'] == 'Historical']
            forecast_data = blended_df[blended_df['Data_Source'] != 'Historical']
            
            # Get last historical date for transition line
            if not historical_data.empty:
                last_historical_date = historical_data['Date'].max()
            else:
                last_historical_date = None
            
            # Add oil rate traces with different colors/styles for historical vs forecast
            if not historical_data.empty:
                fig1.add_trace(go.Scatter(
                    x=historical_data['Date'],
                    y=historical_data['Oil_Blend'],
                    mode='lines+markers',
                    name='Oil - Historical',
                    line=dict(color='darkgreen', width=2, dash='solid'),
                    marker=dict(size=6)
                ))
            
            fig1.add_trace(go.Scatter(
                x=forecast_data['Date'],
                y=forecast_data['Oil_Blend'],
                mode='lines',
                name='Oil - Forecast',
                line=dict(color='green', width=2)
            ))
            
            # Add gas rate traces with different colors/styles for historical vs forecast
            if not historical_data.empty:
                fig1.add_trace(go.Scatter(
                    x=historical_data['Date'],
                    y=historical_data['Gas_Blend'],
                    mode='lines+markers',
                    name='Gas - Historical',
                    line=dict(color='darkred', width=2, dash='solid'),
                    marker=dict(size=6),
                    yaxis='y2'
                ))
            
            fig1.add_trace(go.Scatter(
                x=forecast_data['Date'],
                y=forecast_data['Gas_Blend'],
                mode='lines',
                name='Gas - Forecast',
                line=dict(color='red', width=2),
                yaxis='y2'
            ))
            
            # Add vertical line at last historical date (transition point)
            if last_historical_date:
                fig1.add_shape(
                    type="line",
                    x0=last_historical_date,
                    x1=last_historical_date,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(
                        color="gray",
                        width=2,
                        dash="dash",
                    )
                )
                
                # Add the annotation for forecast start
                fig1.add_annotation(
                    x=last_historical_date,
                    y=1,
                    yref="paper",
                    text="Forecast Start",
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom",
                    xshift=5,
                    font=dict(color="gray")
                )
            
            # Add vertical line at effective date (which may differ from forecast start)
            effective_date_pd = pd.to_datetime(effective_date)
            fig1.add_shape(
                type="line",
                x0=effective_date_pd,
                x1=effective_date_pd,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(
                    color="blue",
                    width=2,
                    dash="dash",
                )
            )
            
            # Add the annotation for effective date
            fig1.add_annotation(
                x=effective_date_pd,
                y=0.9,  # Slightly below the forecast start annotation
                yref="paper",
                text="Effective Date",
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                xshift=5,
                font=dict(color="blue")
            )
        else:
            # Standard traces for forecast-only view
            fig1.add_trace(go.Scatter(
                x=plot_data['Date'],
                y=plot_data['Oil_Blend'],
                mode='lines',
                name='Oil Volume (BBL/month)',
                line=dict(color='green', width=2)
            ))
            
            fig1.add_trace(go.Scatter(
                x=plot_data['Date'],
                y=plot_data['Gas_Blend'],
                mode='lines',
                name='Gas Volume (MCF/month)',
                line=dict(color='red', width=2),
                yaxis='y2'
            ))
            
            # Add vertical line at effective date
            effective_date_pd = pd.to_datetime(effective_date)
            fig1.add_shape(
                type="line",
                x0=effective_date_pd,
                x1=effective_date_pd,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(
                    color="blue",
                    width=2,
                    dash="dash",
                )
            )
            
            # Add the annotation for effective date
            fig1.add_annotation(
                x=effective_date_pd,
                y=0.9,
                yref="paper",
                text="Effective Date",
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                xshift=5,
                font=dict(color="blue")
            )
        
        # Set layout with two y-axes only (removed third axis for NGL) and enhanced dynamic adjustment
        fig1.update_layout(
            title=f'Production Forecast - {well_name}',
            xaxis=dict(
                title='Date',
                rangeslider=dict(visible=True),
                type='date',
                autorange=True  # Add autorange for dynamic x-axis adjustment
            ),
            yaxis=dict(
                title='Oil Volume (BBL/month)',
                titlefont=dict(color='green'),
                tickfont=dict(color='green'),
                autorange=True,  # Add autorange for dynamic y-axis adjustment
                type='log' if use_log_scale else 'linear',  # Add log scale option
                fixedrange=False  # Ensure y-axis can be adjusted by zoom/pan
            ),
            yaxis2=dict(
                title='Gas Volume (MCF/month)',
                titlefont=dict(color='red'),
                tickfont=dict(color='red'),
                anchor='x',
                overlaying='y',
                side='right',
                autorange=True,  # Add autorange for dynamic y-axis adjustment
                type='log' if use_log_scale else 'linear',  # Add log scale option
                fixedrange=False  # Ensure y-axis can be adjusted by zoom/pan
            ),
            height=500,
            margin=dict(l=50, r=100, t=80, b=50),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            uirevision='true'  # This helps maintain zoom level on updates
        )
        
        # Configure additional plot settings for proper dynamic scaling
        fig1.update_layout(
            # These settings help ensure axes actually respond to zoom/pan
            modebar_add=["v1hovermode", "toggleSpikelines"],
            hovermode="x unified",
            dragmode="zoom"
        )
        
        st.plotly_chart(fig1, use_container_width=True)

        # Create revenue and cash flow plot with customization options
        st.markdown("### Revenue and Cash Flow")
        
        # Get historical cash flow data for the past 2 years
        hist_cashflow = get_historical_cashflow(selected_api, price_deck, years=2)
        
        # Add options for chart customization
        chart_options_col1, chart_options_col2, chart_options_col3 = st.columns(3)
        
        with chart_options_col1:
            # Option to view by month or by year
            time_period = st.selectbox(
                "Time Aggregation",
                ["Monthly", "Yearly"],
                key="cashflow_time_period"
            )
        
        with chart_options_col2:
            # Option to choose between bars and lines for monthly/yearly values
            chart_type = st.selectbox(
                "Chart Type",
                ["Bar Chart", "Line Chart"],
                key="cashflow_chart_type"
            )
        
        with chart_options_col3:
            # Number of periods to display - Use forecast_df which is already defined above
            if time_period == "Monthly":
                # We already have forecast_df from above, so this is safe
                max_periods = len(forecast_df)
                display_periods = st.slider(
                    "Months to Display",
                    min_value=12,
                    max_value=min(120, max_periods),
                    value=min(60, max_periods),
                    step=12,
                    key="cashflow_display_months"
                )
            else:  # Yearly
                # Calculate number of years in the forecast
                years_in_forecast = forecast_df['Year'].nunique()
                display_periods = st.slider(
                    "Years to Display",
                    min_value=1,
                    max_value=years_in_forecast,
                    value=min(5, years_in_forecast),
                    step=1,
                    key="cashflow_display_years"
                )
        
        # Prepare data based on selected time period
        if time_period == "Yearly":
            # Group forecast data by year
            yearly_data = forecast_df.groupby('Year').agg({
                'Gross Revenue': 'sum',
                'Net Cash Flow': 'sum'
            }).reset_index()
            
            # Group historical data by year if available
            if not hist_cashflow.empty:
                hist_cashflow['Year'] = hist_cashflow['Date'].dt.year
                hist_yearly = hist_cashflow.groupby('Year').agg({
                    'Gross Revenue': 'sum',
                    'Net Cash Flow': 'sum'
                }).reset_index()
                hist_yearly['Data_Source'] = 'Historical'
                
                # Combine historical and forecast yearly data
                yearly_data['Data_Source'] = 'Forecast'
                yearly_data = pd.concat([hist_yearly, yearly_data], ignore_index=True)
                yearly_data = yearly_data.sort_values('Year').reset_index(drop=True)
            
            # Create cumulative net cash flow column
            yearly_data['Cum_NCF'] = yearly_data['Net Cash Flow'].cumsum()
            
            # Limit to selected number of years
            plot_data = yearly_data.head(display_periods)
            x_data = plot_data['Year']
            x_title = 'Year'
        else:  # Monthly
            # Prepare forecast data
            forecast_plot = forecast_df.head(display_periods).copy()
            forecast_plot['Data_Source'] = 'Forecast'
            
            # Combine with historical data if available
            if not hist_cashflow.empty:
                # Concatenate and sort
                plot_data = pd.concat([hist_cashflow, forecast_plot], ignore_index=True)
                plot_data = plot_data.sort_values('Date').reset_index(drop=True)
                
                # Recalculate cumulative values
                plot_data['Cum_NCF'] = plot_data['Net Cash Flow'].cumsum()
                
                # Limit to display periods
                plot_data = plot_data.tail(display_periods)
            else:
                plot_data = forecast_plot
            
            x_data = plot_data['Date']
            x_title = 'Date'
        
        # Create figure
        fig3 = go.Figure()
        
        # Split data into historical and forecast for different styling
        if 'Data_Source' in plot_data.columns and 'Historical' in plot_data['Data_Source'].values:
            hist_data = plot_data[plot_data['Data_Source'] == 'Historical']
            fcst_data = plot_data[plot_data['Data_Source'] == 'Forecast']
            
            # Add traces based on selected chart type
            if chart_type == "Bar Chart":
                # Add historical revenue trace
                if not hist_data.empty:
                    fig3.add_trace(go.Bar(
                        x=hist_data[x_data.name],
                        y=hist_data['Gross Revenue'],
                        name='Revenue - Historical',
                        marker_color='darkblue'
                    ))
                
                # Add forecast revenue trace
                fig3.add_trace(go.Bar(
                    x=fcst_data[x_data.name],
                    y=fcst_data['Gross Revenue'],
                    name='Revenue - Forecast',
                    marker_color='blue'
                ))
                
                # Add historical net cash flow trace
                if not hist_data.empty:
                    fig3.add_trace(go.Bar(
                        x=hist_data[x_data.name],
                        y=hist_data['Net Cash Flow'],
                        name='Net Cash Flow - Historical',
                        marker_color='darkgreen'
                    ))
                
                # Add forecast net cash flow trace
                fig3.add_trace(go.Bar(
                    x=fcst_data[x_data.name],
                    y=fcst_data['Net Cash Flow'],
                    name='Net Cash Flow - Forecast',
                    marker_color='green'
                ))
                
                # Set barmode
                fig3.update_layout(barmode='group')
            else:  # Line Chart
                # Add historical revenue trace
                if not hist_data.empty:
                    fig3.add_trace(go.Scatter(
                        x=hist_data[x_data.name],
                        y=hist_data['Gross Revenue'],
                        mode='lines+markers',
                        name='Revenue - Historical',
                        line=dict(color='darkblue', width=2)
                    ))
                
                # Add forecast revenue trace
                fig3.add_trace(go.Scatter(
                    x=fcst_data[x_data.name],
                    y=fcst_data['Gross Revenue'],
                    mode='lines+markers',
                    name='Revenue - Forecast',
                    line=dict(color='blue', width=2)
                ))
                
                # Add historical net cash flow trace
                if not hist_data.empty:
                    fig3.add_trace(go.Scatter(
                        x=hist_data[x_data.name],
                        y=hist_data['Net Cash Flow'],
                        mode='lines+markers',
                        name='Net Cash Flow - Historical',
                        line=dict(color='darkgreen', width=2)
                    ))
                
                # Add forecast net cash flow trace
                fig3.add_trace(go.Scatter(
                    x=fcst_data[x_data.name],
                    y=fcst_data['Net Cash Flow'],
                    mode='lines+markers',
                    name='Net Cash Flow - Forecast',
                    line=dict(color='green', width=2)
                ))
        else:
            # Add traces based on selected chart type (no historical data)
            if chart_type == "Bar Chart":
                # Add revenue trace
                fig3.add_trace(go.Bar(
                    x=x_data,
                    y=plot_data['Gross Revenue'],
                    name='Revenue',
                    marker_color='blue'
                ))
                
                # Add net cash flow trace
                fig3.add_trace(go.Bar(
                    x=x_data,
                    y=plot_data['Net Cash Flow'],
                    name='Net Cash Flow',
                    marker_color='green'
                ))
                
                # Set barmode
                fig3.update_layout(barmode='group')
            else:  # Line Chart
                # Add revenue trace
                fig3.add_trace(go.Scatter(
                    x=x_data,
                    y=plot_data['Gross Revenue'],
                    mode='lines+markers',
                    name='Revenue',
                    line=dict(color='blue', width=2)
                ))
                
                # Add net cash flow trace
                fig3.add_trace(go.Scatter(
                    x=x_data,
                    y=plot_data['Net Cash Flow'],
                    mode='lines+markers',
                    name='Net Cash Flow',
                    line=dict(color='green', width=2)
                ))
        
        # Add cumulative NCF trace with secondary y-axis
        if time_period == "Yearly":
            cumulative_data = plot_data['Cum_NCF']
        else:
            cumulative_data = plot_data['Cum_NCF'] if 'Cum_NCF' in plot_data.columns else plot_data['Net Cash Flow'].cumsum()
        
        fig3.add_trace(go.Scatter(
            x=x_data,
            y=cumulative_data,
            mode='lines',
            name='Cumulative Net Cash Flow',
            line=dict(color='red', width=2),
            yaxis='y2'
        ))
        
        # Add vertical line at effective date on the cash flow chart as well
        effective_date_pd = pd.to_datetime(effective_date)
        
        # Only add the effective date line if we're in monthly view and the effective date is within the date range
        if time_period == "Monthly":
            min_date = plot_data['Date'].min()
            max_date = plot_data['Date'].max()
            
            if min_date <= effective_date_pd <= max_date:
                fig3.add_shape(
                    type="line",
                    x0=effective_date_pd,
                    x1=effective_date_pd,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(
                        color="blue",
                        width=2,
                        dash="dash",
                    )
                )
                
                fig3.add_annotation(
                    x=effective_date_pd,
                    y=0.9,
                    yref="paper",
                    text="Effective Date",
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom",
                    xshift=5,
                    font=dict(color="blue")
                )
        
        # Set layout with two y-axes and dynamic adjustment
        fig3.update_layout(
            title='Revenue and Cash Flow',
            xaxis=dict(
                title=x_title,
                showgrid=True,
                rangeslider=dict(visible=True),
                type='date' if time_period == "Monthly" else '-',
                autorange=True  # Add autorange for dynamic x-axis adjustment
            ),
            yaxis=dict(
                title='Value ($)',
                titlefont=dict(color='blue'),
                tickfont=dict(color='blue'),
                showgrid=True,
                autorange=True,  # Add autorange for dynamic y-axis adjustment
                fixedrange=False  # Ensure y-axis can be adjusted by zoom/pan
            ),
            yaxis2=dict(
                title='Cumulative NCF ($)',
                titlefont=dict(color='red'),
                tickfont=dict(color='red'),
                anchor='x',
                overlaying='y',
                side='right',
                autorange=True,  # Add autorange for dynamic y-axis adjustment
                fixedrange=False  # Ensure y-axis can be adjusted by zoom/pan
            ),
            height=500,
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            hovermode='x unified',
            uirevision='true'  # This helps maintain zoom level on updates
        )
        
        st.plotly_chart(fig3, use_container_width=True)

    # This section should be at the end of the visualization/display section

# Add download options
st.markdown("### Export Forecast Data")
try:
    st.markdown(get_csv_download_link(forecast_df, f"{selected_api}_forecast.csv", "Download Forecast CSV"), unsafe_allow_html=True)

    # Add option to download blended data if available
    if use_blended_view and 'Data_Source' in blended_df.columns and any(blended_df['Data_Source'] == 'Historical'):
        st.markdown(get_csv_download_link(blended_df, f"{selected_api}_blended.csv", "Download Blended Data CSV"), unsafe_allow_html=True)

    # Option to download cash flow data specifically
    if 'Net Cash Flow' in forecast_df.columns:
        cash_flow_df = forecast_df[['Date', 'Oil_Blend', 'Gas_Blend', 'Gross Revenue', 'Net Cash Flow', 'Cum_NCF']]
        st.markdown(get_csv_download_link(cash_flow_df, f"{selected_api}_cash_flow.csv", "Download Cash Flow CSV"), unsafe_allow_html=True)
        
        # Add option to download blended cash flow data if available
        if use_blended_view and 'Data_Source' in blended_df.columns and any(blended_df['Data_Source'] == 'Historical'):
            blended_cash_flow = blended_df[['Date', 'Data_Source', 'Oil_Blend', 'Gas_Blend', 'Gross Revenue', 'Net Cash Flow', 'Cum_NCF']]
            st.markdown(get_csv_download_link(blended_cash_flow, f"{selected_api}_blended_cash_flow.csv", "Download Blended Cash Flow CSV"), unsafe_allow_html=True)

    # Option to save monthly forecast to Snowflake
    if st.checkbox("Save Monthly Forecast to Snowflake", value=False):
        if session:
            try:
                # Create a clean version for database storage
                monthly_df = forecast_df[['Date', 'Oil_Blend', 'Gas_Blend', 'Gross Revenue', 'Net Cash Flow', 'Years']].copy()
                monthly_df['API_UWI'] = selected_api
                monthly_df['ECORUN_ID'] = ecorun_id
                monthly_df['ECORUN_DATE'] = datetime.now()
                
                # Save to Snowflake
                session.create_dataframe(monthly_df).write.mode("append").save_as_table("ECON_MONTHLY_FORECAST")
                
                st.success("Monthly forecast data saved to Snowflake")
            except Exception as e:
                st.error(f"Error saving monthly forecast: {str(e)}")
        else:
            st.warning("Not connected to Snowflake - monthly forecast not saved")
            
    # =============================================================================
    # SECTION 5D: SINGLE WELL FORECAST - DATA TABLES & EXPORT
    # =============================================================================

        # Only run this section if a forecast exists
        if st.session_state.forecast_df is not None:
            forecast_df = st.session_state.forecast_df  # Get the stored forecast
            selected_api = st.session_state.get('selected_api', 'MANUAL')  # Get selected_api from session state
            
            # Get historical cash flow data for comparison
            hist_cashflow = None
            if param_source == "Select Existing Well":
                hist_cashflow = get_historical_cashflow(selected_api, price_deck, years=2)
            
            # Additional display options for the forecast
            with st.expander("Additional Forecast Options", expanded=False):
                st.markdown("### Additional Analysis")
                
                # Choose options for advanced analysis
                analysis_type = st.selectbox(
                    "Analysis Type",
                    ["Production Summary", "Revenue Breakdown", "NPV Sensitivity"],
                    key="advanced_analysis"
                )
                
                if analysis_type == "Production Summary":
                    # Summary statistics for production
                    st.subheader("Production Summary Statistics")
                    
                    # Calculate monthly averages and other statistics
                    stats_df = pd.DataFrame({
                        'Metric': ['Average Monthly', 'Maximum Monthly', 'Minimum Monthly', '1st Year Total', '5-Year Total'],
                        'Oil (BBL)': [
                            round(forecast_df['Oil_Blend'].mean(), 0),
                            round(forecast_df['Oil_Blend'].max(), 0),
                            round(forecast_df['Oil_Blend'][forecast_df['Oil_Blend'] > 0].min(), 0),
                            round(forecast_df['Oil_Blend'].head(12).sum(), 0),
                            round(forecast_df['Oil_Blend'].head(min(60, len(forecast_df))).sum(), 0)
                        ],
                        'Gas (MCF)': [
                            round(forecast_df['Gas_Blend'].mean(), 0),
                            round(forecast_df['Gas_Blend'].max(), 0),
                            round(forecast_df['Gas_Blend'][forecast_df['Gas_Blend'] > 0].min(), 0),
                            round(forecast_df['Gas_Blend'].head(12).sum(), 0),
                            round(forecast_df['Gas_Blend'].head(min(60, len(forecast_df))).sum(), 0)
                        ]
                    })
                    
                    st.dataframe(stats_df.style.format({
                        'Oil (BBL)': '{:,.0f}',
                        'Gas (MCF)': '{:,.0f}'
                    }))
                    
                    # Decline curve parameters summary
                    st.subheader("Decline Curve Parameters")
                    
                    params_df = pd.DataFrame({
                        'Parameter': ['Initial Rate', 'Decline Type', 'Initial Decline Rate', 'B-Factor', 'Terminal Decline'],
                        'Oil': [
                            f"{oil_qi:.0f} BBL/day",
                            "Hyperbolic" if oil_decline_type == "H" else "Exponential",
                            f"{oil_decline*100:.1f}%",
                            f"{oil_b_factor:.2f}" if oil_decline_type == "H" else "N/A",
                            f"{oil_terminal_decline*100:.1f}%" if oil_decline_type == "H" else "N/A"
                        ],
                        'Gas': [
                            f"{gas_qi:.0f} MCF/day",
                            "Hyperbolic" if gas_decline_type == "H" else "Exponential",
                            f"{gas_decline*100:.1f}%",
                            f"{gas_b_factor:.2f}" if gas_decline_type == "H" else "N/A",
                            f"{gas_terminal_decline*100:.1f}%" if gas_decline_type == "H" else "N/A"
                        ]
                    })
                    
                    st.dataframe(params_df)
                    
                elif analysis_type == "Revenue Breakdown":
                    # Breakdown of revenue components
                    st.subheader("Revenue Components Breakdown")
                    
                    # Calculate totals for each component
                    total_oil_revenue = forecast_df['Oil_Revenue'].sum()
                    total_gas_revenue = forecast_df['Gas_Revenue'].sum()
                    total_expenses = forecast_df['GPT'].sum() + forecast_df['Sev Tax'].sum() + forecast_df['AdVal Tax'].sum()
                    total_net = forecast_df['Net Cash Flow'].sum()
                    
                    # Create a revenue breakdown chart
                    revenue_data = {
                        'Category': ['Oil Revenue', 'Gas Revenue', 'Expenses', 'Net Cash Flow'],
                        'Amount': [total_oil_revenue, total_gas_revenue, -total_expenses, total_net]
                    }
                    
                    revenue_df = pd.DataFrame(revenue_data)
                    
                    # Display breakdown stats
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Oil Revenue %", f"{total_oil_revenue/(total_oil_revenue+total_gas_revenue)*100:.1f}%")
                        st.metric("Gas Revenue %", f"{total_gas_revenue/(total_oil_revenue+total_gas_revenue)*100:.1f}%")
                    
                    with col2:
                        st.metric("Expense Ratio", f"{total_expenses/(total_oil_revenue+total_gas_revenue)*100:.1f}%")
                        st.metric("Net Cash Flow Ratio", f"{total_net/(total_oil_revenue+total_gas_revenue)*100:.1f}%")
                    
                    # Create a waterfall chart
                    waterfall_fig = go.Figure(go.Waterfall(
                        name="Revenue Breakdown",
                        orientation="v",
                        measure=["relative", "relative", "relative", "total"],
                        x=revenue_df['Category'],
                        textposition="outside",
                        text=revenue_df['Amount'].apply(lambda x: f"${x:,.0f}"),
                        y=revenue_df['Amount'],
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                    ))
                    
                    waterfall_fig.update_layout(
                        title="Revenue Breakdown Waterfall",
                        showlegend=False
                    )
                    
                    st.plotly_chart(waterfall_fig, use_container_width=True)
                    
                elif analysis_type == "NPV Sensitivity":
                    # NPV sensitivity analysis
                    st.subheader("NPV Sensitivity Analysis")
                    
                    # Create sensitivity sliders
                    col1, col2 = st.columns(2)
                    with col1:
                        price_factor = st.slider("Price Factor", min_value=0.5, max_value=1.5, value=1.0, step=0.1)
                        decline_factor = st.slider("Decline Rate Factor", min_value=0.8, max_value=1.2, value=1.0, step=0.05)
                    
                    with col2:
                        expense_factor = st.slider("Expense Factor", min_value=0.7, max_value=1.3, value=1.0, step=0.1)
                        discount_rate = st.slider("Discount Rate", min_value=8, max_value=24, value=10, step=2)
                    
                    # Calculate base case NPV
                    base_npv = pv_results[f'PV{discount_rate}']
                    
                    # Calculate simple sensitivities
                    high_price_npv = base_npv * price_factor
                    low_expense_npv = base_npv * (1 + (1-expense_factor)*0.3)  # Simplified sensitivity
                    high_decline_npv = base_npv * (1 - (decline_factor-1)*0.5)  # Simplified sensitivity
                    combined_npv = base_npv * price_factor * (1 + (1-expense_factor)*0.3) * (1 - (decline_factor-1)*0.5)
                    
                    # Create sensitivity results
                    sensitivity_data = {
                        'Scenario': ['Base Case', 'Price Sensitivity', 'Expense Sensitivity', 'Decline Sensitivity', 'Combined Effect'],
                        f'NPV{discount_rate}': [base_npv, high_price_npv, low_expense_npv, high_decline_npv, combined_npv],
                        'Change': [0, high_price_npv - base_npv, low_expense_npv - base_npv, high_decline_npv - base_npv, combined_npv - base_npv],
                        'Change %': [0, (high_price_npv / base_npv - 1) * 100, (low_expense_npv / base_npv - 1) * 100, 
                                (high_decline_npv / base_npv - 1) * 100, (combined_npv / base_npv - 1) * 100]
                    }
except:
    st.warning('Click Generate Forecast button to see results')

# =============================================================================
# SECTION 6: MULTI-WELL ANALYSIS
# =============================================================================

with tab2:
    st.markdown('<p class="sub-header">Multi-Well Analysis</p>', unsafe_allow_html=True)
    
    # Add run identification inputs for batch processing
    st.subheader("Run Identification")
    col_batch1, col_batch2, col_batch3 = st.columns(3)
    with col_batch1:
        batch_deal = st.text_input("Deal Name", "Batch Deal", key="batch_deal")
    with col_batch2:
        batch_id = st.text_input("Run ID", "BATCH001", key="batch_id")
    with col_batch3:
        # Load economic scenarios for batch selection
        batch_scenario = st.selectbox("Economic Scenario", scenario_names, key="batch_scenario")
    
    # Add forecast date settings for batch
    st.subheader("Forecast Settings")
    col_batch_date1, col_batch_date2, col_batch_date3 = st.columns(3)
    with col_batch_date1:
        batch_date = st.date_input("Effective Date", datetime.now().date().replace(day=1), key="batch_date")
    with col_batch_date2:
        batch_months = st.slider("Forecast Months", min_value=12, max_value=600, value=360, step=12, key="batch_months")
    with col_batch_date3:
        st.write(price_deck_options)
        price_deck_options.sort()
        batch_price_deck = st.selectbox("Price Deck", price_deck_options, key="batch_price_deck", index = 3)
    
    # Well selection for batch processing
    st.subheader("Well Selection")
    
    # Load well data
    wells_df = get_well_data()
    
    if not wells_df.empty:
        # Filter options
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            basin_filter = st.multiselect("Filter by Basin", sorted(wells_df['ENVBASIN'].unique().tolist()), 
                                        default=sorted(wells_df['ENVBASIN'].unique().tolist()))
        
        with col_filter2:
            # Custom filter option
            custom_filter = st.text_input("API Filter (comma-separated)", "")
        
        # Apply filters
        if basin_filter:
            filtered_wells = wells_df[wells_df['ENVBASIN'].isin(basin_filter)]
        else:
            filtered_wells = wells_df
            
        # Apply custom API filter if provided
        if custom_filter:
            api_list = [api.strip() for api in custom_filter.split(',')]
            filtered_wells = filtered_wells[filtered_wells['API_UWI'].isin(api_list)]
        
        # Display filtered wells
        st.write(f"**Found {len(filtered_wells)} wells matching criteria**")
        st.dataframe(filtered_wells[['API_UWI', 'WELLNAME', 'ENVBASIN', 'OIL_USER_QI', 'GAS_USER_QI']])
        
        # Select wells to process
        selected_apis = st.multiselect("Select Wells for Processing", 
                                      filtered_wells['API_UWI'].tolist(),
                                      default=filtered_wells['API_UWI'].tolist()[:5])
        
        if st.button("Run Multi-Well Analysis", key="multi_well_btn"):
            if not selected_apis:
                st.error("Please select at least one well to process")
            else:
                # Get economic parameters
                scenario_data = scenarios[scenarios['ECON_SCENARIO'] == batch_scenario].iloc[0]
                
                # Process each well
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                well_results = []
                
                for i, api in enumerate(selected_apis):
                    status_text.text(f"Processing well {i+1} of {len(selected_apis)}: {api}")
                    
                    # Get well data
                    well_data = wells_df[wells_df['API_UWI'] == api].iloc[0]
                    
                    # Process the well - similar to single well code but without visualization
                    # This would call the calculation functions for each well
                    
                    # Create dummy result for now
                    well_result = {
                        'API_UWI': api,
                        'WELLNAME': well_data['WELLNAME'],
                        'ENVBASIN': well_data['ENVBASIN'],
                        'ECORUN_DATE': datetime.now(),
                        'ECORUN_DEAL': batch_deal,
                        'ECORUN_ID': batch_id,
                        'ECORUN_SCENARIO': batch_scenario,
                        'PRICE_DECK': batch_price_deck,
                        'TOTAL_OIL_BLEND': 0,
                        'TOTAL_GAS_BLEND': 0,
                        'TOTAL_NGL_BLEND': 0,
                        'TOTAL_REVENUE': 0,
                        'TOTAL_GPT': 0, 
                        'TOTAL_SEVTAX': 0,
                        'TOTAL_ADVALTAX': 0,
                        'TOTAL_NET_CASH_FLOW': 0
                    }
                    
                    # In a real implementation, this would calculate all values
                    
                    well_results.append(well_result)
                    progress_bar.progress((i+1)/len(selected_apis))
                
                # Create results dataframe
                results_df = pd.DataFrame(well_results)
                
                # Display results
                st.success(f"Processed {len(selected_apis)} wells successfully")
                st.dataframe(results_df)
                
                # Option to save results
                if st.checkbox("Save Batch Results to Snowflake", value=True):
                    if session:
                        try:
                            session.create_dataframe(results_df).write.mode("append").save_as_table("ECON_RESULTS")
                            st.success("Batch results saved to ECON_RESULTS table")
                        except Exception as e:
                            st.error(f"Error saving batch results: {str(e)}")
                    else:
                        st.warning("Not connected to Snowflake - results not saved")
                
                # Download option
                st.markdown(get_csv_download_link(results_df, "batch_results.csv", "Download Batch Results CSV"), unsafe_allow_html=True)
    else:
        st.error("No well data available for batch processing")

# =============================================================================
# SECTION 7: DATA IMPORT & CONNECTIONS
# =============================================================================

with tab3:
    st.markdown('<p class="sub-header">Data Import & Connections</p>', unsafe_allow_html=True)
    
    # Snowflake connection info
    st.subheader("Snowflake Connection")
    if session:
        st.success("Successfully connected to Snowflake")
        
        # Show table information
        table_info = [
            ("ECON_INPUT", "Well parameters including decline curve data"),
            ("VW_WELL_INPUT", "Basic well information"),
            ("VW_PROD_INPUT", "Historical production data"),
            ("ECON_SCENARIOS", "Economic parameters for different scenarios"),
            ("PRICE_DECK", "Price forecasts"),
            ("ECON_RESULTS", "Forecast summary results"),
            ("ECON_MONTHLY_FORECAST", "Detailed monthly forecast data (optional)")
        ]
        
        # Display table info
        st.write("### Available Tables")
        for table, description in table_info:
            st.write(f"**{table}**: {description}")
        
        # Option to view table schemas
        if st.checkbox("View Table Schemas"):
            for table, _ in table_info:
                try:
                    schema = session.sql(f"DESCRIBE TABLE {table}").to_pandas()
                    with st.expander(f"{table} Schema"):
                        st.dataframe(schema)
                except Exception as e:
                    st.warning(f"Cannot retrieve schema for {table}: {str(e)}")
        
        # Add refresh data option
        if st.button("Refresh Data from Snowflake"):
            st.cache_data.clear()
            st.success("Cache cleared. Data will be refreshed on next query.")
    else:
        st.warning("Running in local mode without Snowflake connection")
        st.info("When deployed in Snowflake, this app will connect automatically to your tables")
    
    # Data management options
    st.subheader("Data Management")
    
    data_tabs = st.tabs(["View Results", "Export Data", "Import Data"])
    
    with data_tabs[0]:
        # View recent results
        st.write("### Recent Forecast Results")
        if session:
            try:
                recent_results = session.sql("""
                SELECT 
                    ECORUN_DATE, 
                    ECORUN_DEAL, 
                    ECORUN_ID, 
                    ECORUN_SCENARIO,
                    COUNT(DISTINCT API_UWI) as WELL_COUNT,
                    SUM(TOTAL_OIL_BLEND) as TOTAL_OIL,
                    SUM(TOTAL_GAS_BLEND) as TOTAL_GAS,
                    SUM(PV10) as TOTAL_PV10
                FROM 
                    ECON_RESULTS
                GROUP BY 
                    ECORUN_DATE, ECORUN_DEAL, ECORUN_ID, ECORUN_SCENARIO
                ORDER BY 
                    ECORUN_DATE DESC
                LIMIT 10
                """).to_pandas()
                
                st.dataframe(recent_results.style.format({
                    'TOTAL_OIL': '{:,.0f}',
                    'TOTAL_GAS': '{:,.0f}',
                    'TOTAL_PV10': '${:,.0f}'
                }))
            except Exception as e:
                st.error(f"Error retrieving recent results: {str(e)}")
        else:
            st.info("Connect to Snowflake to view recent results")
    
    with data_tabs[1]:
        # Export data options
        st.write("### Export Data")
        
        export_options = ["Well List", "Production History", "Economic Scenarios", "Price Deck", "Forecast Results"]
        export_choice = st.selectbox("Select Data to Export", export_options)
        
        if st.button("Export Selected Data"):
            if session:
                try:
                    if export_choice == "Well List":
                        export_data = get_well_data()
                    elif export_choice == "Production History":
                        # Limit production history to avoid huge exports
                        export_data = get_production_history().head(10000)
                        st.info("Production history limited to 10,000 records. Use filters for complete data.")
                    elif export_choice == "Economic Scenarios":
                        export_data = get_economic_scenarios()
                    elif export_choice == "Price Deck":
                        export_data = get_price_deck()
                    elif export_choice == "Forecast Results":
                        export_data = session.sql("SELECT * FROM ECON_RESULTS ORDER BY ECORUN_DATE DESC LIMIT 1000").to_pandas()
                    
                    st.dataframe(export_data)
                    st.markdown(get_csv_download_link(export_data, f"{export_choice.lower().replace(' ', '_')}.csv", 
                                                    f"Download {export_choice} CSV"), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error exporting data: {str(e)}")
            else:
                st.warning("Connect to Snowflake to export data")
    
    with data_tabs[2]:
        # Import data options
        st.write("### Import Data")
        
        import_options = ["Well Parameters", "Production History", "Economic Scenarios", "Price Deck"]
        import_choice = st.selectbox("Select Data to Import", import_options)
        
        upload_file = st.file_uploader(f"Upload {import_choice} CSV", type="csv")
        
        if upload_file is not None:
            try:
                import_df = pd.read_csv(upload_file)
                st.write("Preview of data to import:")
                st.dataframe(import_df.head(5))
                
                if st.button("Import Data to Snowflake"):
                    if session:
                        if import_choice == "Well Parameters":
                            target_table = "ECON_INPUT"
                        elif import_choice == "Production History":
                            target_table = "VW_PROD_INPUT"
                        elif import_choice == "Economic Scenarios":
                            target_table = "ECON_SCENARIOS"
                        elif import_choice == "Price Deck":
                            target_table = "PRICE_DECK"
                        
                        # This would need column mapping in a real implementation
                        st.info(f"Data would be imported to {target_table}. In a real implementation, this would include column mapping.")
                    else:
                        st.warning("Connect to Snowflake to import data")
            except Exception as e:
                st.error(f"Error processing import file: {str(e)}")

# =============================================================================
# SECTION 8: DOCUMENTATION
# =============================================================================

with tab4:
    st.markdown('<p class="sub-header">Documentation & Help</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Decline Curve Analysis Methods
    
    This application supports two main types of decline curve analysis:
    
    #### 1. Exponential Decline
    
    The exponential decline model assumes a constant percentage decline rate over time. The production rate at any time is calculated as:
    
    ```
    q(t) = q_i * e^(-d*t)
    ```
    
    Where:
    - q(t) is the production rate at time t
    - q_i is the initial production rate
    - d is the annual decline rate
    - t is time in years
    
    #### 2. Hyperbolic Decline
    
    The hyperbolic decline model is more flexible and often more realistic for unconventional reservoirs. It allows the decline rate to decrease over time. The production rate is calculated as:
    
    ```
    q(t) = q_i / (1 + b * d_i * t)^(1/b)
    ```
    
    Where:
    - q(t) is the production rate at time t
    - q_i is the initial production rate
    - d_i is the initial annual decline rate
    - b is the hyperbolic exponent (b-factor)
    - t is time in years
    
    When the decline rate reaches a specified terminal decline rate, the model switches to exponential decline.
    
    ### Parameter Guide
    
    #### Initial Rate (q_i)
    The starting production rate for oil (BBL/day) or gas (MCF/day). This can be based on recent production history or estimated from similar wells.
    
    #### Decline Rate
    The annual percentage rate at which production decreases, expressed as a decimal (e.g., 0.3 for 30% annual decline).
    
    #### B-Factor
    The hyperbolic exponent that controls the curvature of the hyperbolic decline. Values typically range from 0.1 to 2.0:
    - Lower values (0.1-0.5): Steeper initial decline, quicker approach to terminal decline
    - Higher values (1.0-2.0): More gradual initial decline, common in unconventional reservoirs
    
    #### Terminal Decline Rate
    The minimum annual decline rate that applies after the hyperbolic phase. This creates a "modified hyperbolic" model where production eventually shifts to exponential decline.
    
    #### Economic Limit
    The minimum production rate (oil or gas) at which the well remains economically viable. Production forecasts stop when this limit is reached.
    
    ### Using This Application
    
    1. **Single Well Forecast**:
       - Enter parameters directly in the form
       - Generate production profiles and economic analyses for one well
       - Visualize results and download the forecast
    
    2. **Multi-Well Analysis**:
       - Select multiple wells based on filters
       - Process them simultaneously
       - Compare results across your well portfolio
    
    3. **Data Import**:
       - Connect directly to Snowflake tables
       - Load and manipulate data within the application
       - Export results in various formats
    """)
    
    st.markdown("### Snowflake Table Structure")
    
    with st.expander("View Snowflake Table Mapping"):
        st.markdown("""
        | App Table | Snowflake Table | Description |
        |-----------|-----------------|-------------|
        | Well List | ECON_INPUT, VW_WELL_INPUT | Well parameters and basic information |
        | Production History | VW_PROD_INPUT | Historical production data by month |
        | Forecast Assumptions | ECON_SCENARIOS | Economic parameters for forecast scenarios |
        | Price Deck | PRICE_DECK | Price forecasts by month |
        | Well Forecast Results | ECON_RESULTS | Summary results and PV values |
        | Well Monthly Forecast | ECON_MONTHLY_FORECAST | Detailed monthly forecast (optional) |
        """)
    
    # Example file formats
    with st.expander("Example File Formats"):
        st.code("""# ECON_INPUT table sample
API_UWI,OIL_USER_QI,GAS_USER_QI,OIL_DECLINE_TYPE,GAS_DECLINE_TYPE,OIL_USER_B_FACTOR,GAS_USER_B_FACTOR,OIL_USER_DECLINE,GAS_USER_DECLINE,OIL_D_MIN,GAS_D_MIN,OIL_FCST_YRS,GAS_FCST_YRS,OIL_Q_MIN,GAS_Q_MIN
4200300001,500,2500,H,H,1.2,0.8,0.6,0.7,0.05,0.08,30,30,5,50
4200300002,300,4000,H,H,0.9,0.7,0.5,0.65,0.05,0.08,30,30,5,50
4200300003,800,1200,E,E,,,0.45,0.5,,,10,12,5,50""")
    
    # Helpful tips
    st.markdown("### Helpful Tips")
    
    tips = [
        "**Initial Rates**: For existing wells, the app will consider recent production history when setting initial rates.",
        "**Saving Results**: Results are only saved to Snowflake when you check the 'Save Results' option.",
        "**Monthly Forecast Data**: By default, only summary results are saved. Check the option to save monthly data if needed.",
        "**Batch Processing**: For large numbers of wells, use the filters to select specific subsets to process.",
        "**Data Refresh**: Use the 'Refresh Data' button in the Data Import tab to clear the cache and reload from Snowflake."
    ]
    
    for tip in tips:
        st.markdown(f"- {tip}")
    
    # FAQ section
    st.markdown("### Frequently Asked Questions")
    
    faqs = {
        "What is the difference between Exponential and Hyperbolic decline?": 
            "Exponential decline assumes a constant decline rate, while hyperbolic decline allows the decline rate to decrease over time. Hyperbolic is often more accurate for unconventional reservoirs with steep initial declines that flatten over time.",
        
        "How do I choose the right B-factor?": 
            "B-factors typically range from 0.1 to 2.0. Higher values (>1.0) create a more gradual decline curve, common in unconventional reservoirs. Lower values (<0.5) create steeper initial declines. Analyze historical production to calibrate the B-factor.",
        
        "What is the Terminal Decline rate?": 
            "The Terminal Decline rate is the minimum annual decline rate in a hyperbolic model. When the calculated decline rate reaches this value, the forecast switches from hyperbolic to exponential decline.",
        
        "Why is the monthly forecast data optional?": 
            "Monthly forecast data can be voluminous, especially for many wells over long time periods. To minimize storage requirements, only summary results are saved by default."
    }
    
    for question, answer in faqs.items():
        with st.expander(question):
            st.write(answer)
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    ### Disclaimer
    
    This application is for informational purposes only. The forecasts and economic analyses produced are estimates based on the provided inputs and mathematical models. Actual well performance may vary significantly from these projections. Users should exercise their professional judgment when interpreting results and making business decisions.
    
    The decline curve analysis methods used here are simplified models that may not capture all reservoir characteristics or operational factors that influence actual production.
    
    ### Support and Feedback
    
    For questions, support, or feedback, please contact your Snowflake administrator or the application developer.
    """)

# =============================================================================
# SECTION 9: MAIN FUNCTION
# =============================================================================

# Main app functionality
if __name__ == "__main__":
    # The main app code is already defined above with the Streamlit UI
    pass