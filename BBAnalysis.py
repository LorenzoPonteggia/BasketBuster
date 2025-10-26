import pandas as pd
import numpy as np
import openpyxl
import os

# We can re-use the currency conversion logic from our existing graph module
from BBGraph import convert_prices_to_usd


def analyze_hedge_performance(basket_prices_usd, target_stock_prices_usd):
    """
    Assesses the performance of a basket at hedging a target stock.

    This function calculates key metrics to determine how effectively a portfolio (basket)
    tracks the price movements of a target asset.

    Args:
        basket_prices_usd (pd.Series): Time series of the basket's value in USD.
        target_stock_prices_usd (pd.Series): Time series of the target stock's price in USD.

    Returns:
        dict: A dictionary containing key hedging performance metrics.
    """
    # 1. Calculate daily returns from the price/value series
    basket_returns = basket_prices_usd.pct_change().dropna()
    target_returns = target_stock_prices_usd.pct_change().dropna()

    # 2. Align the two return series by date, dropping any non-matching dates
    df = pd.DataFrame({'basket': basket_returns, 'target': target_returns}).dropna()

    if len(df) < 2:
        return {
            'error': 'Not enough overlapping data to calculate performance.'
        }

    # 3. Calculate key performance and risk metrics
    correlation = df['basket'].corr(df['target'])
    r_squared = correlation ** 2

    tracking_error_daily = (df['basket'] - df['target']).std()
    tracking_error_annualized = tracking_error_daily * np.sqrt(252)

    # Calculate the beta of the basket relative to the target stock
    covariance = df.cov()
    beta = covariance.loc['basket', 'target'] / covariance.loc['target', 'target']

    # Calculate hedge effectiveness by measuring the reduction in variance
    variance_target = df['target'].var()
    variance_hedged_portfolio = (df['target'] - df['basket']).var()
    variance_reduction = (variance_target - variance_hedged_portfolio) / variance_target if variance_target > 0 else 0

    metrics = {
        'correlation': correlation,
        'r_squared': r_squared,
        'beta': beta,
        'tracking_error_daily': tracking_error_daily,
        'tracking_error_annualized': tracking_error_annualized,
        'variance_reduction_pct': variance_reduction * 100
    }

    return metrics


def calculate_performance_attribution(basket_weights, prices_local, fx_rates, currency_map, basket_name=None, log_callback=None):
    """
    Performs PnL attribution for a basket, breaking down cumulative returns by constituent
    and by local price changes vs. FX changes.

    This uses a standard additive attribution model for multi-currency portfolios,
    ensuring that the sum of individual stock daily PnL contributions (as returns)
    equals the total basket daily return.

    Args:
        basket_weights (pd.DataFrame): DataFrame with 'Stock' as index and 'Weight' column.
        prices_local (pd.DataFrame): DataFrame of daily prices in local currency.
        fx_rates (pd.DataFrame): DataFrame of daily FX rates for converting to USD.
        currency_map (dict): Dictionary mapping stock tickers to their local currency.
        basket_name (str, optional): The name of the basket for more descriptive warnings. Defaults to None.

    Returns:
        tuple: A tuple containing two DataFrames:
               - daily_attribution_df: A detailed breakdown of daily PnL (as % return contribution) per stock.
               - total_attribution_df: A summary of total PnL (as % return contribution) from each component per stock.
    """
    daily_attribution_components = []
    
    # Calculate full USD prices for all relevant stocks
    full_prices_usd = convert_prices_to_usd(prices_local, currency_map, fx_rates)
    
    # Ensure basket weights align with available stocks and sum to 1 (or close to 1)
    valid_stocks_in_basket = [stock for stock in basket_weights.index if stock in full_prices_usd.columns]
    if not valid_stocks_in_basket:
        if log_callback:
            log_callback("Warning: No valid stocks found in basket for attribution. Returning empty DataFrames.", "WARNING")
        else:
            print("Warning: No valid stocks found in basket for attribution. Returning empty DataFrames.")
        return pd.DataFrame(), pd.DataFrame()

    basket_weights_aligned = basket_weights.loc[valid_stocks_in_basket, 'Weight']
    # Normalize weights if they don't sum to 1, to ensure attribution sums correctly
    if not np.isclose(basket_weights_aligned.sum(), 1.0):
        # FIX: Use the basket_name parameter for a more informative warning message.
        warning_msg = f"Warning: Basket weights for '{basket_name if basket_name else 'current basket'}' do not sum to 1. Normalizing weights."
        if log_callback:
            log_callback(warning_msg, "WARNING")
        else:
            print(warning_msg)
        basket_weights_aligned = basket_weights_aligned / basket_weights_aligned.sum()

    # Get the daily returns for each stock in USD
    daily_returns_usd_all = full_prices_usd[valid_stocks_in_basket].pct_change().fillna(0)

    
    # Calculate the daily return of the entire basket (for reconciliation)
    basket_total_daily_return = (daily_returns_usd_all * basket_weights_aligned).sum(axis=1)

    # Now, calculate attribution for each stock
    for stock in valid_stocks_in_basket:
        weight = basket_weights_aligned.loc[stock]
        currency = currency_map.get(stock, 'USD') # Default to USD

        price_series_local = prices_local[stock] # Local currency prices
        
        # Get aligned FX series
        fx_series_aligned = pd.Series(1.0, index=price_series_local.index) # Default to 1.0 for USD
        if currency != 'USD':
            if currency in fx_rates.columns:
                fx_series_aligned = fx_rates[currency].reindex(price_series_local.index).ffill().bfill()
            else:
                print(f"Warning: Missing FX data for {currency} (stock: {stock}). Treating as USD for attribution.")
        
        if fx_series_aligned.isnull().any():
            print(f"Warning: Could not find valid FX rates for {currency} (stock: {stock}) in required date range. Treating as USD for attribution.")
            fx_series_aligned = pd.Series(1.0, index=price_series_local.index)


        # Create a DataFrame for this stock's data
        stock_df = pd.DataFrame({
            'price_local': price_series_local,
            'fx': fx_series_aligned
        }).dropna()

        # Calculate daily returns for local price, FX, and USD price
        stock_df['local_return'] = stock_df['price_local'].pct_change().fillna(0)
        stock_df['fx_return'] = stock_df['fx'].pct_change().fillna(0)
        
        # The total USD return for this stock for the day
        stock_df['total_usd_return'] = daily_returns_usd_all[stock]

        # --- Performance Attribution for the stock's USD return ---
        # The goal is to break down stock_df['total_usd_return'] into local and FX components,
        # and then multiply by the basket weight.

        # 1. Local Price Contribution (adjusted for previous day's FX)
        # This is the return from the local price change, if FX rate had been constant.
        # stock_usd_t_minus_1 = stock_df['price_local'].shift(1) * stock_df['fx'].shift(1)
        # pnl_local = (stock_df['price_local'] - stock_df['price_local'].shift(1)) * stock_df['fx'].shift(1)
        # contrib_local = pnl_local / stock_usd_t_minus_1_for_calc if stock_usd_t_minus_1_for_calc is not 0

        # Simplified additive attribution:
        # PnL_USD_Total = W * (P_t * F_t - P_{t-1} * F_{t-1})
        # PnL_USD_Total = W * [ (P_t - P_{t-1}) * F_{t-1}   (Local Price Change, fixed FX)
        #                   +  P_{t-1} * (F_t - F_{t-1})   (FX Change, fixed Local Price)
        #                   +  (P_t - P_{t-1}) * (F_t - F_{t-1}) ] (Interaction Term)
        
        # Expressing as contributions to the *return* of the basket
        # Contribution (Local) = Weight * Local_Return
        stock_df['pnl_local_return_contrib'] = weight * stock_df['local_return']
        
        # Contribution (FX) = Weight * (1 + Local_Return) * FX_Return
        # This ensures additivity: Total Return = W * Local_Return + W * (1+Local_Return) * FX_Return
        # (1+total_return) = (1+local_return)(1+fx_return)
        # total_return = local_return + fx_return + local_return*fx_return
        # So, for weighted contributions:
        # W * total_return = W * local_return + W * fx_return + W * local_return * fx_return
        # We can then attribute:
        # Local component: W * local_return
        # FX component: W * fx_return + W * local_return * fx_return  => W * fx_return * (1 + local_return)
        
        stock_df['pnl_fx_return_contrib'] = weight * stock_df['fx_return'] * (1 + stock_df['local_return'])
        
        # Total contribution for this stock to the basket's return for the day
        stock_df['pnl_total_return_contrib'] = stock_df['pnl_local_return_contrib'] + stock_df['pnl_fx_return_contrib']
        
        # Store these daily percentage contributions
        stock_attribution_df = stock_df[['pnl_local_return_contrib', 'pnl_fx_return_contrib', 'pnl_total_return_contrib']].dropna()
        stock_attribution_df['stock'] = stock
        daily_attribution_components.append(stock_attribution_df)

    if not daily_attribution_components:
        return pd.DataFrame(), pd.DataFrame()

    # Concatenate all daily attribution components
    combined_daily_attribution_df = pd.concat(daily_attribution_components).sort_index()

    # --- Reconciliation Check ---
    # Sum of individual stock daily PnL contributions should now perfectly equal total basket daily return
    daily_reconciliation_check = combined_daily_attribution_df.groupby(combined_daily_attribution_df.index)['pnl_total_return_contrib'].sum()
    
    # Optional: reconciliation check
    if log_callback:
        log_callback("Daily Basket Return vs. Sum of Attributed Returns (first 5 days):")
        reco_df = pd.DataFrame({'Basket_Return': basket_total_daily_return, 'Attributed_Sum': daily_reconciliation_check})
        log_callback(f"Reconciliation check - Max difference: {(reco_df['Basket_Return'] - reco_df['Attributed_Sum']).abs().max():.10f}")
    else:
        print("\nDaily Basket Return vs. Sum of Attributed Returns (first 5 days):")
        reco_df = pd.DataFrame({'Basket_Return': basket_total_daily_return, 'Attributed_Sum': daily_reconciliation_check})
        print(reco_df.head())
        print(f"Max difference: {(reco_df['Basket_Return'] - reco_df['Attributed_Sum']).abs().max():.10f}")

    # Total attribution summary: sum of daily contributions for the period
    total_attribution_df = combined_daily_attribution_df.groupby('stock')[
        ['pnl_local_return_contrib', 'pnl_fx_return_contrib', 'pnl_total_return_contrib']
    ].sum().sort_values('pnl_total_return_contrib', ascending=False)
    
    # Rename columns for clarity for the output Excel file
    total_attribution_df.columns = ['Total Local PnL (%)', 'Total FX PnL (%)', 'Total Stock PnL (%)']
    
    # The daily attribution df contains all details
    daily_attribution_output_df = combined_daily_attribution_df.copy()
    daily_attribution_output_df.columns = ['Local PnL (%)', 'FX PnL (%)', 'Total PnL (%)', 'stock']

    return daily_attribution_output_df, total_attribution_df


def run_backtest_analysis(prices_file, currency_file, fx_file, basket_folder, basket_files, 
                         start_date, end_date, target_stock=None, output_file=None, log_callback=None):
    """
    Main function to run backtest analysis for multiple baskets and save results to Excel.
    
    Parameters:
    - prices_file: Path to stock prices CSV file
    - currency_file: Path to asset info/currency CSV file  
    - fx_file: Path to FX rates CSV file
    - basket_folder: Path to folder containing basket CSV files
    - basket_files: List of basket CSV filenames to analyze
    - start_date: Start date for analysis (string)
    - end_date: End date for analysis (string)
    - target_stock: Optional target stock ticker for hedge analysis (if None, only attribution analysis is performed)
    - output_file: Path for output Excel file
    - log_callback: Optional callback function for logging
    
    Returns:
    - output_file: Path to the created Excel file
    """
    
    def log(message, level="INFO"):
        if log_callback:
            log_callback(message, level)
        else:
            print(f"[{level}] {message}")
    
    try:
        log("Starting backtest analysis...")
        
        # Load all common data files once
        log("Loading and preparing common data files...")
        prices_local = pd.read_csv(prices_file, parse_dates=['Date'], index_col='Date')
        prices_local = prices_local.loc[start_date:end_date]
        
        asset_info = pd.read_csv(currency_file)
        # Use 'Name' column as key to match tickers used in price data and baskets
        asset_info['Name'] = asset_info['Name'].str.strip()
        currency_map = asset_info.set_index('Name')['CCY'].to_dict()

        fx_rates = pd.read_csv(fx_file, parse_dates=['Date'], index_col='Date')
        fx_rates = fx_rates.loc[start_date:end_date]
        
        log(f"Loaded data for {len(prices_local.columns)} stocks from {start_date} to {end_date}")
        
        # Log analysis mode
        if target_stock:
            log(f"Analysis mode: Basket performance vs target stock '{target_stock}' + Attribution analysis")
        else:
            log("Analysis mode: Attribution analysis only (no target stock selected)")
        
        # Prepare results containers
        hedge_performance_results = []
        attribution_summary_results = []
        
        # Process each basket
        for basket_filename in basket_files:
            log(f"Processing basket: {basket_filename}")
            
            try:
                # Load basket weights
                basket_path = os.path.join(basket_folder, basket_filename)
                basket_weights = pd.read_csv(basket_path, header=None, names=['Stock', 'Weight'])
                basket_weights = basket_weights.set_index('Stock')
                
                # Filter to stocks available in price data
                available_stocks = [stock for stock in basket_weights.index if stock in prices_local.columns]
                if not available_stocks:
                    log(f"Warning: No stocks from {basket_filename} found in price data", "WARNING")
                    continue
                    
                basket_weights = basket_weights.loc[available_stocks]
                
                # Calculate basket performance in USD
                basket_prices_usd = convert_prices_to_usd(
                    prices_local[available_stocks], 
                    currency_map, 
                    fx_rates
                )
                
                # Calculate basket cumulative returns
                basket_returns = basket_prices_usd.pct_change().fillna(0)
                basket_weights_aligned = basket_weights.reindex(basket_returns.columns).fillna(0)
                basket_daily_returns = basket_returns.dot(basket_weights_aligned['Weight'])
                basket_cumulative_returns = (1 + basket_daily_returns).cumprod() - 1
                
                # Analyze hedge performance if target stock is provided and available
                if target_stock and target_stock in prices_local.columns:
                    target_prices_usd = convert_prices_to_usd(
                        prices_local[[target_stock]], 
                        currency_map, 
                        fx_rates
                    )
                    target_returns = target_prices_usd.pct_change().fillna(0)
                    target_cumulative_returns = (1 + target_returns).cumprod() - 1
                    
                    # Calculate hedge performance metrics
                    hedge_metrics = analyze_hedge_performance(
                        basket_cumulative_returns, 
                        target_cumulative_returns[target_stock]
                    )
                    hedge_metrics['basket'] = basket_filename.replace('.csv', '')
                    hedge_performance_results.append(hedge_metrics)
                    
                    log(f"  Hedge performance - Correlation: {hedge_metrics['correlation']:.4f}, "
                        f"R-squared: {hedge_metrics['r_squared']:.4f}, "
                        f"Tracking Error: {hedge_metrics['tracking_error_annualized']:.4f}")
                elif target_stock and target_stock not in prices_local.columns:
                    log(f"  Warning: Target stock '{target_stock}' not found in price data", "WARNING")
                
                # Calculate performance attribution
                daily_attribution, total_attribution = calculate_performance_attribution(
                    basket_weights, 
                    prices_local[available_stocks], 
                    fx_rates, 
                    currency_map, 
                    basket_filename.replace('.csv', ''),
                    log_callback
                )
                
                # Convert total_attribution DataFrame to dictionary format for summary
                # Get the sum across all stocks for each component
                attribution_summary = {
                    'Total Local PnL (%)': total_attribution['Total Local PnL (%)'].sum(),
                    'Total FX PnL (%)': total_attribution['Total FX PnL (%)'].sum(), 
                    'Total Stock PnL (%)': total_attribution['Total Stock PnL (%)'].sum(),
                    'basket': basket_filename.replace('.csv', '')
                }
                attribution_summary_results.append(attribution_summary)
                
                log(f"  Attribution analysis completed - {len(available_stocks)} stocks analyzed")
                
            except Exception as e:
                log(f"Error processing basket {basket_filename}: {e}", "ERROR")
                continue
        
        # Create Excel file with results
        log("Creating Excel output file...")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Hedge Performance Summary (only if target stock was provided and analysis was performed)
            if hedge_performance_results:
                hedge_df = pd.DataFrame(hedge_performance_results)
                hedge_df.to_excel(writer, sheet_name='Hedge Performance Summary', index=False)
                log(f"  Created 'Hedge Performance Summary' sheet with {len(hedge_df)} baskets")
            elif target_stock:
                log("  No hedge performance results - target stock may not be available in data", "WARNING")
            
            # Attribution Summary
            if attribution_summary_results:
                attribution_df = pd.DataFrame(attribution_summary_results)
                attribution_df.to_excel(writer, sheet_name='Basket Attribution Summary', index=False)
                log(f"  Created 'Basket Attribution Summary' sheet with {len(attribution_df)} baskets")
            
            # Individual basket sheets with daily attribution
            for basket_filename in basket_files:
                try:
                    basket_path = os.path.join(basket_folder, basket_filename)
                    basket_weights = pd.read_csv(basket_path, header=None, names=['Stock', 'Weight'])
                    basket_weights = basket_weights.set_index('Stock')
                    
                    # Filter to available stocks
                    available_stocks = [stock for stock in basket_weights.index if stock in prices_local.columns]
                    if available_stocks:
                        basket_weights = basket_weights.loc[available_stocks]
                        
                        # Calculate attribution for this basket
                        daily_attribution, _ = calculate_performance_attribution(
                            basket_weights, 
                            prices_local[available_stocks], 
                            fx_rates, 
                            currency_map, 
                            basket_filename.replace('.csv', ''),
                            log_callback
                        )
                        
                        # Save to Excel sheet
                        sheet_name = basket_filename.replace('.csv', '')[:31]  # Excel sheet name limit
                        daily_attribution.to_excel(writer, sheet_name=sheet_name, index=True)
                        log(f"  Created '{sheet_name}' sheet with daily attribution data")
                        
                except Exception as e:
                    log(f"Error creating sheet for {basket_filename}: {e}", "WARNING")
                    continue
        
        analysis_type = "hedge performance + attribution" if target_stock else "attribution only"
        log(f"Backtest analysis completed successfully ({analysis_type}). Results saved to: {output_file}")
        return output_file
        
    except Exception as e:
        log(f"Error in backtest analysis: {e}", "ERROR")
        raise


# This block allows the file to be run as a standalone script for testing
if __name__ == '__main__':
    # --- Define Inputs (these would eventually come from the GUI) ---
    PRICES_FILE = 'C:/pythonContainer/BasketBuster/Transfer/BPCData/spy2020.csv'
    CURRENCY_FILE = 'C:/pythonContainer/BasketBuster/Transfer/BPCData/SPYassetInfo.csv'
    FX_FILE = 'C:/pythonContainer/BasketBuster/Transfer/BPCData/FXRatesTRIM.csv'
    
    # NEW: Define a folder and a list of basket files to analyze
    BASKET_FOLDER = 'C:/pythonContainer/BasketBuster/Transfer/BPCBaskets/'
    BASKET_FILES_TO_ANALYZE = ['GS MS BAC.csv', '10names to april25.csv', 'LP CVXPY 2023-2024.csv']
    
    # NEW: Define the output Excel file path
    OUTPUT_EXCEL_FILE = 'C:/pythonContainer/BasketBuster/Transfer/BasketAnalysisOutput.xlsx'

    START_DATE = '2024-01-01'
    END_DATE = '2024-10-10'
    TARGET_STOCK = 'JPM UN'

    # --- Load all common data files once ---
    print("Loading and preparing common data files...")
    prices_local = pd.read_csv(PRICES_FILE, parse_dates=['Date'], index_col='Date')
    prices_local = prices_local.loc[START_DATE:END_DATE]
    
    asset_info = pd.read_csv(CURRENCY_FILE)
    # FIX: The currency map must be built using the 'Name' column as the key,
    # not the 'BBG' column, to match the tickers used in the price data and baskets.
    asset_info['Name'] = asset_info['Name'].str.strip()
    currency_map = asset_info.set_index('Name')['CCY'].to_dict()

    fx_rates = pd.read_csv(FX_FILE, parse_dates=['Date'], index_col='Date')
    fx_rates = fx_rates.loc[START_DATE:END_DATE]
    
    prices_usd = convert_prices_to_usd(prices_local, currency_map, fx_rates)
    
    # --- Initialize containers to store results from all baskets ---
    all_hedge_metrics = []
    # Store total attributions for each basket. Keys are basket filenames, values are DataFrames.
    all_total_attributions = {} 
    # Store daily attributions for each basket. Keys are basket filenames, values are DataFrames.
    all_daily_attributions = {}

    # --- Loop through each basket and perform the analysis ---
    for basket_filename in BASKET_FILES_TO_ANALYZE:
        basket_file_path = os.path.join(BASKET_FOLDER, basket_filename)
        
        if not os.path.exists(basket_file_path):
            print(f"Warning: Basket file not found, skipping: {basket_filename}")
            continue

        print(f"\n--- Analyzing Basket: {basket_filename} ---")
        
        # Load basket weights with whitespace stripping for robustness
        basket_df = pd.read_csv(basket_file_path, header=None, names=['Stock', 'Weight'])
        basket_df['Stock'] = basket_df['Stock'].str.strip()
        basket_weights = basket_df.set_index('Stock')
        print(f"Loaded basket '{basket_filename}' with {len(basket_weights)} stocks.")

        # --- 1. Run Hedge Performance Analysis ---
        basket_constituents = basket_weights.index
        valid_constituents = [s for s in basket_constituents if s in prices_usd.columns]
        
        basket_returns = prices_usd[valid_constituents].pct_change().dot(basket_weights.loc[valid_constituents, 'Weight'])
        basket_value = (1 + basket_returns.fillna(0)).cumprod()
        target_prices = prices_usd[TARGET_STOCK]
        
        hedge_metrics = analyze_hedge_performance(basket_value, target_prices)
        hedge_metrics['basket_name'] = basket_filename
        all_hedge_metrics.append(hedge_metrics)
        
       # --- 2. Run Performance Attribution Analysis ---
        daily_attr, total_attr = calculate_performance_attribution(
            basket_weights, prices_local, fx_rates, currency_map, basket_name=basket_filename
        )
        if not total_attr.empty:
            all_total_attributions[basket_filename] = total_attr
            all_daily_attributions[basket_filename] = daily_attr # Store daily attribution

    # --- Write all collected results to a single Excel file ---
    print(f"\n--- Writing all results to {OUTPUT_EXCEL_FILE} ---")
    try:
        with pd.ExcelWriter(OUTPUT_EXCEL_FILE, engine='openpyxl') as writer:
            # Create and write the hedge performance summary sheet
            hedge_summary_df = pd.DataFrame(all_hedge_metrics).set_index('basket_name')
            hedge_summary_df.to_excel(writer, sheet_name='Hedge Performance Summary')
            print("  ✓ Saved Hedge Performance Summary sheet.")

            # NEW: Create and write a basket-level attribution summary for FX impact
            if all_total_attributions:
                basket_summary_list = []
                for basket_name, attr_df in all_total_attributions.items():
                    total_local = attr_df['Total Local PnL (%)'].sum()
                    total_fx = attr_df['Total FX PnL (%)'].sum()
                    total_pnl = attr_df['Total Stock PnL (%)'].sum()
                    basket_summary_list.append({
                        'Basket': basket_name,
                        'Total Local PnL (%)': total_local,
                        'Total FX PnL (%)': total_fx,
                        'Total Basket PnL (%)': total_pnl
                    })
                basket_summary_df = pd.DataFrame(basket_summary_list).set_index('Basket')
                basket_summary_df.to_excel(writer, sheet_name='Basket Attribution Summary')
                print("  ✓ Saved Basket Attribution Summary sheet.")

            # Create a summary sheet for all total attributions (by stock)
            if all_total_attributions:
                combined_total_attr = pd.concat(all_total_attributions, names=['Basket', 'Stock'])
                combined_total_attr.to_excel(writer, sheet_name='All Baskets Total Attribution')
                print("  ✓ Saved All Baskets Total Attribution sheet.")
            else:
                print("  No total attribution data to save.")

            # Create a separate sheet for each basket's DAILY PnL attribution
            for basket_name, daily_attr_df in all_daily_attributions.items():
                if not daily_attr_df.empty:
                    # Sanitize sheet name (Excel has a 31 char limit)
                    # Use a format that includes stock and date for daily breakdowns
                    sheet_name = os.path.splitext(basket_name)[0][:20] + '_Daily' # Truncate more for safety
                    # FIX: Use the correct column name 'Total PnL (%)' for the pivot table values.
                    daily_attr_df_pivot = daily_attr_df.pivot_table(index=daily_attr_df.index, columns='stock', values='Total PnL (%)')
                    daily_attr_df_pivot.to_excel(writer, sheet_name=sheet_name)
                    print(f"  ✓ Saved daily attribution sheet for '{basket_name}'.")
                else:
                    print(f"  No daily attribution data for '{basket_name}' to save.")
        
        print(f"\nAnalysis complete. Results saved to {OUTPUT_EXCEL_FILE}")
    except Exception as e:
        print(f"\nError: Could not write to Excel file. Please ensure it is not open. Details: {e}")

