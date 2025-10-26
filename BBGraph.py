import pandas as pd
import os
import plotly.graph_objects as go


def load_fx_rates(fx_file, start_date=None, end_date=None):
    """Load FX rates from CSV file with comprehensive error handling"""
    try:
        if not fx_file or not os.path.exists(fx_file):
            raise FileNotFoundError(f"FX rates file not found: {fx_file}")
        
        # Check file size to avoid loading empty files
        if os.path.getsize(fx_file) == 0:
            raise ValueError(f"FX rates file is empty: {fx_file}")
        
        fx_df = pd.read_csv(fx_file, parse_dates=['Date'], index_col='Date')
        
        # Validate that we have a Date column and it's properly parsed
        if fx_df.index.name != 'Date':
            raise ValueError("FX rates file must have a 'Date' column")
        
        if len(fx_df) == 0:
            raise ValueError("FX rates file contains no data")
        
        # Check for missing or invalid dates
        if fx_df.index.isnull().any():
            raise ValueError("FX rates file contains invalid dates")
        
        # Filter by date range if provided
        if start_date:
            start_date = pd.to_datetime(start_date)
            fx_df = fx_df[fx_df.index >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            fx_df = fx_df[fx_df.index <= end_date]
        
        # Validate that we still have data after filtering
        if len(fx_df) == 0:
            raise ValueError(f"No FX data available for the specified date range: {start_date} to {end_date}")
        
        return fx_df
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading FX rates: {e}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"FX rates file is empty or contains no valid data: {fx_file}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing FX rates file {fx_file}: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading FX rates from {fx_file}: {e}")



def convert_prices_to_usd(prices_df, currency_map, fx_df):
    """Convert stock prices to USD using FX rates with comprehensive error handling"""
    try:
        if prices_df is None or prices_df.empty:
            raise ValueError("Prices dataframe is empty or None")
        
        if currency_map is None:
            raise ValueError("Currency mapping is None")
        
        if fx_df is None or fx_df.empty:
            raise ValueError("FX rates dataframe is empty or None")
        
        prices_usd = prices_df.copy()
        conversion_errors = []
        
        for stock in prices_df.columns:
            try:
                currency = currency_map.get(stock, 'USD')
                
                if currency != 'USD':
                    if currency not in fx_df.columns:
                        conversion_errors.append(f"FX rates for currency '{currency}' (stock: {stock}) not found in FX data")
                        continue
                    
                    # Check for missing FX data
                    fx_column = fx_df[currency]
                    if fx_column.isnull().all():
                        conversion_errors.append(f"All FX rates for currency '{currency}' (stock: {stock}) are missing")
                        continue
                    
                    # Align FX rates to price dates and apply
                    rates = fx_column.reindex(prices_df.index).ffill().bfill()
                    
                    # Check if we still have missing rates after forward/backward fill
                    if rates.isnull().any():
                        missing_count = rates.isnull().sum()
                        conversion_errors.append(f"Missing FX rates for {currency} (stock: {stock}): {missing_count} dates")
                        continue
                    
                    # Check for zero or negative FX rates
                    if (rates <= 0).any():
                        invalid_count = (rates <= 0).sum()
                        conversion_errors.append(f"Invalid FX rates for {currency} (stock: {stock}): {invalid_count} zero/negative rates")
                        continue
                    
                    prices_usd[stock] = prices_df[stock] * rates
                    
            except Exception as e:
                conversion_errors.append(f"Error converting {stock} to USD: {e}")
                continue
        
        # Report conversion errors but don't fail completely
        if conversion_errors:
            error_summary = f"Currency conversion issues found:\n" + "\n".join(f"  • {err}" for err in conversion_errors)
            print(f"WARNING: {error_summary}")
        
        return prices_usd
        
    except Exception as e:
        raise RuntimeError(f"Critical error in currency conversion: {e}")



def calculate_cumulative_returns(prices_df, weights_df):
    daily_returns = prices_df.pct_change().fillna(0)
    aligned_weights = weights_df.reindex(daily_returns.columns).fillna(0)
    basket_returns = daily_returns.dot(aligned_weights['Weight'])
    cumulative_returns = (1 + basket_returns).cumprod() - 1
    return cumulative_returns



def plot_cumulative_returns(stocks_cumulative, baskets_cumulative):
    fig = go.Figure()

    # Plot individual stock cumulative returns
    for name, cumulative in stocks_cumulative.items():
        fig.add_trace(go.Scatter(
            x=cumulative.index, y=cumulative.values,
            mode='lines', name=name,
            hovertemplate=
                "%{y:.2%}"
        ))

    # Plot each basket
    for fname, cumulative in baskets_cumulative.items():
        label = os.path.splitext(fname)[0]
        fig.add_trace(go.Scatter(
            x=cumulative.index, y=cumulative.values,
            mode='lines', name=label,
            hovertemplate=
                "%{y:.2%}"
        ))

    fig.update_layout(
        title="Cumulative Returns (USD)",
        yaxis_tickformat=".2%",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        autosize=True,
        height=600,
    )

    #fig.show()
    return fig



def save_cumulative_returns_to_csv(stocks_cumulative, baskets_cumulative, output_file):
    combined = pd.DataFrame(stocks_cumulative)
    for fname, cumulative in baskets_cumulative.items():
        label = os.path.splitext(fname)[0]
        combined[label] = cumulative
    combined.to_csv(output_file)



def prepareData(stock_names, prices_file, weights_folder,
         currency_file, fx_file,
         start_date=None, end_date=None, log_callback=None):
    """Prepare stock price data with comprehensive error handling and validation"""

    def log(message, level="INFO"):
        if log_callback:
            log_callback(message, level)
        else:
            print(f"[{level}] {message}")

    try:
        # Validate input parameters
        if not prices_file or not os.path.exists(prices_file):
            raise FileNotFoundError(f"Stock prices file not found: {prices_file}")
        
        if not currency_file or not os.path.exists(currency_file):
            raise FileNotFoundError(f"Asset info/currency file not found: {currency_file}")
        
        if not fx_file or not os.path.exists(fx_file):
            raise FileNotFoundError(f"FX rates file not found: {fx_file}")

        # Validate and format dates
        try:
            if start_date:
                start_date = str(start_date).replace('-','/')
                start_parsed = pd.to_datetime(start_date)
            if end_date:
                end_date = str(end_date).replace('-', '/')
                end_parsed = pd.to_datetime(end_date)
            
            # Check date logic
            if start_date and end_date and start_parsed >= end_parsed:
                raise ValueError(f"Start date ({start_date}) must be before end date ({end_date})")
                
        except Exception as e:
            raise ValueError(f"Invalid date format: {e}")

        log(f'Date Range: {start_date} to {end_date}')

        # Load price levels with validation
        try:
            if os.path.getsize(prices_file) == 0:
                raise ValueError(f"Prices file is empty: {prices_file}")
            
            prices = pd.read_csv(prices_file, parse_dates=['Date'], index_col='Date')
            
            if len(prices) == 0:
                raise ValueError("Prices file contains no data")
            
            if prices.index.isnull().any():
                raise ValueError("Prices file contains invalid dates")
            
            log(f"Loaded {len(prices)} price observations for {len(prices.columns)} assets")
            
        except pd.errors.EmptyDataError:
            raise ValueError(f"Prices file is empty or contains no valid data: {prices_file}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing prices file {prices_file}: {e}")

        # Filter by date range
        original_length = len(prices)
        if start_date:
            start = pd.to_datetime(start_date)
            prices = prices[prices.index >= start]
        if end_date:
            end = pd.to_datetime(end_date)
            prices = prices[prices.index <= end]
        
        if len(prices) == 0:
            raise ValueError(f"No price data available for date range {start_date} to {end_date}")
        
        if len(prices) < original_length:
            log(f"Filtered to {len(prices)} observations within date range")

        # Load stock currency mapping with validation
        try:
            if os.path.getsize(currency_file) == 0:
                raise ValueError(f"Asset info file is empty: {currency_file}")
            
            asset_info = pd.read_csv(currency_file)
            
            if len(asset_info) == 0:
                raise ValueError("Asset info file contains no data")
            
            required_columns = ['BBG', 'CCY']
            missing_columns = [col for col in required_columns if col not in asset_info.columns]
            if missing_columns:
                raise ValueError(f"Asset info file missing required columns: {missing_columns}")
            
            # Check for missing currency mappings
            asset_info_clean = asset_info.dropna(subset=['BBG', 'CCY'])
            if len(asset_info_clean) < len(asset_info):
                dropped = len(asset_info) - len(asset_info_clean)
                log(f"Warning: Dropped {dropped} asset info rows with missing BBG or CCY data", "WARNING")
            
            currency_map = asset_info_clean.set_index('BBG')['CCY'].to_dict()
            log(f"Loaded currency mapping for {len(currency_map)} assets")
            
        except pd.errors.EmptyDataError:
            raise ValueError(f"Asset info file is empty: {currency_file}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing asset info file {currency_file}: {e}")

        # Load FX rates (this function now has its own comprehensive error handling)
        fx = load_fx_rates(fx_file, start_date, end_date)
        log(f"Loaded FX rates for {len(fx.columns)} currencies")

        # Convert prices to USD (this function now has its own error handling)
        prices_usd = convert_prices_to_usd(prices, currency_map, fx)
        log(f"Currency conversion completed for {len(prices_usd.columns)} assets")

        # Final validation
        if prices_usd.isnull().all().all():
            raise ValueError("All price data is missing after USD conversion")
        
        # Report any assets with significant missing data
        missing_pct = prices_usd.isnull().mean()
        high_missing = missing_pct[missing_pct > 0.1]  # More than 10% missing
        if len(high_missing) > 0:
            log(f"Warning: {len(high_missing)} assets have >10% missing data", "WARNING")
            for asset, pct in high_missing.head(5).items():  # Show top 5
                log(f"  {asset}: {pct:.1%} missing", "WARNING")

        return prices_usd

    except Exception as e:
        log(f"Error in data preparation: {e}", "ERROR")
        raise

def plot(prices_usd, stock_names, weights_folder, selected_baskets=None,
         output_file='cumulative_returns_usd.csv', log_callback=None):

    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)

    # Compute individual stock cumulative returns (only if stocks are selected)
    stocks_cumulative = {}
    if stock_names:  # Only process if we have stocks selected
        for stock in stock_names:
            if stock in prices_usd.columns:  # Check if stock exists in data
                returns = prices_usd[stock].pct_change().fillna(0)
                stocks_cumulative[stock] = (1 + returns).cumprod() - 1
            else:
                log(f"Warning: Stock '{stock}' not found in price data")

    # Compute basket cumulative returns (only for selected baskets)
    baskets_cumulative = {}
    if os.path.isdir(weights_folder) and selected_baskets:
        # Only process if we have selected baskets
        files_to_process = selected_baskets
        
        for fname in files_to_process:
            if fname.endswith('.csv'):
                file_path = os.path.join(weights_folder, fname)
                if not os.path.exists(file_path):
                    log(f"Warning: Basket file '{fname}' not found")
                    continue
                    
                try:
                    # Load and clean the weights data
                    weights_raw = pd.read_csv(file_path, header=None,
                                              names=['Stock', 'Weight'])
                    
                    # Filter out empty rows and invalid data
                    weights_clean = weights_raw.dropna().copy()
                    weights_clean = weights_clean[weights_clean['Stock'].str.strip() != '']
                    weights_clean = weights_clean[weights_clean['Weight'].notna()]
                    
                    # Set index and ensure no duplicates
                    weights = weights_clean.set_index('Stock')
                    
                    # Remove any duplicate stock entries (keep first occurrence)
                    if weights.index.duplicated().any():
                        log(f"Warning: Duplicate stocks found in {fname}, keeping first occurrence")
                        weights = weights[~weights.index.duplicated(keep='first')]
                    
                    baskets_cumulative[fname] = calculate_cumulative_returns(prices_usd, weights)
                    
                except Exception as e:
                    log(f"Error processing basket {fname}: {e}")
    elif selected_baskets is None or len(selected_baskets) == 0:
        log("No baskets selected - skipping basket processing")

    # Check if we have anything to plot
    if not stocks_cumulative and not baskets_cumulative:
        log("Warning: No data to plot (no stocks selected and no valid baskets found)")
        return go.Figure()  # Return empty figure

    # Plot and save
    fig = plot_cumulative_returns(stocks_cumulative, baskets_cumulative)
    save_cumulative_returns_to_csv(stocks_cumulative, baskets_cumulative, output_file)

    return fig