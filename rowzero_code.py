Y' and merge with production data
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