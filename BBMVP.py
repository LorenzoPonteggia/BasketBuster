import numpy as np
import pandas as pd
import cvxpy as cp
import os
import cplex

def optimize_correlation_hedge(
    target_stock="JPM UN",
    input_file="FactorModels/data2023.csv",
    output_file_name="CVXPY CPLEX MIQP v7.csv",
    max_positions=25,
    min_weight=0.01,
    max_weight=0.20,
    shorting_allowed=False,
    apply_outlier_treatment=True,
    outlier_quantiles=(0.01, 0.99),
    start_date="2023/01/01",
    end_date="2024-10-10",
    high_tracking_error_threshold=0.10,
    min_positions_warning=5,
    save_files=True,
    verbose=True,
    hedge_universe=None,
    apply_benchmark_discount=False,
    benchmark_stock=None,
    log_callback=None
):

    def log(message):
        """Log function that respects verbose setting and uses callback if provided"""
        if verbose:
            if log_callback:
                log_callback(message)
            else:
                print(message)
            
    # ---------------------------
    # Enhanced data loading and cleaning
    # ---------------------------
    if verbose:
        log(f"Loading data from {input_file}...")
    
    try:
        # Validate input file exists and is readable
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        if os.path.getsize(input_file) == 0:
            raise ValueError(f"Input file is empty: {input_file}")
        
        # Load data with comprehensive error handling
        try:
            df = pd.read_csv(input_file, parse_dates=["Date"], thousands=",")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Input file contains no data: {input_file}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing input file {input_file}: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Encoding error reading file {input_file}: {e}")
        
        if len(df) == 0:
            raise ValueError(f"Input file contains no rows: {input_file}")
        
        # Validate required Date column
        if "Date" not in df.columns:
            raise ValueError("Input file must contain a 'Date' column")
        
        # Check for duplicate date column names (case insensitive)
        date_columns = [col for col in df.columns if col.lower() == 'date']
        if len(date_columns) > 1:
            log(f"Warning: Multiple date columns found: {date_columns}. Using 'Date'.")
        
        df.set_index("Date", inplace=True)
        
        # Validate date index
        if df.index.isnull().any():
            null_dates = df.index.isnull().sum()
            log(f"Warning: Removing {null_dates} rows with invalid dates")
            df = df[df.index.notnull()]
        
        if len(df) == 0:
            raise ValueError("No valid dates found in input file")
        
        # Convert to numeric with detailed error reporting
        original_columns = len(df.columns)
        numeric_conversion_errors = []
        
        for col in df.columns:
            try:
                # Try to convert column to numeric
                original_values = len(df[col].dropna())
                df[col] = pd.to_numeric(df[col], errors="coerce")
                numeric_values = len(df[col].dropna())
                
                if numeric_values < original_values * 0.5:  # More than 50% data lost
                    numeric_conversion_errors.append(f"{col}: {original_values - numeric_values} non-numeric values")
                    
            except Exception as e:
                numeric_conversion_errors.append(f"{col}: conversion error - {e}")
        
        if numeric_conversion_errors and verbose:
            log("Numeric conversion issues found:")
            for error in numeric_conversion_errors[:5]:  # Show first 5
                log(f"  • {error}")
            if len(numeric_conversion_errors) > 5:
                log(f"  • ... and {len(numeric_conversion_errors) - 5} more")
        
        # Remove columns that are entirely non-numeric
        before_drop = len(df.columns)
        df = df.dropna(axis=1, how='all')
        after_drop = len(df.columns)
        
        if after_drop < before_drop:
            dropped = before_drop - after_drop
            log(f"Removed {dropped} columns with no numeric data")
        
        if len(df.columns) == 0:
            raise ValueError("No numeric columns found in input file")
        
        # Remove rows with all NaN values
        before_drop = len(df)
        df = df.dropna(how='all')
        after_drop = len(df)
        
        if after_drop < before_drop:
            dropped = before_drop - after_drop
            log(f"Removed {dropped} rows with no data")
        
        if len(df) == 0:
            raise ValueError("No valid data rows found after cleaning")
        
        # Validate and filter by date range
        try:
            start_parsed = pd.to_datetime(start_date)
            end_parsed = pd.to_datetime(end_date)
            
            if start_parsed >= end_parsed:
                raise ValueError(f"Start date ({start_date}) must be before end date ({end_date})")
                
        except Exception as e:
            raise ValueError(f"Invalid date format: {e}")
        
        original_length = len(df)
        df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
        
        if len(df) == 0:
            raise ValueError(f"No data found for date range {start_date} to {end_date}. "
                           f"Available data range: {df.index.min()} to {df.index.max()}")
        
        if len(df) < original_length:
            log(f"Filtered to {len(df)} observations from {original_length} total")
            
        # Final data quality checks
        if len(df) < 30:
            log(f"Warning: Very limited data ({len(df)} observations). Results may be unreliable.")
        
        # Check for target stock
        if target_stock not in df.columns:
            available_stocks = sorted(df.columns.tolist())
            raise ValueError(f"Target stock '{target_stock}' not found in data. ")
        
        log(f"Data loaded successfully: {len(df)} observations, {len(df.columns)} stocks")
        
    except Exception as e:
        if "No data found for date range" in str(e) or "not found in data" in str(e):
            raise  # Re-raise specific errors as-is
        else:
            raise RuntimeError(f"Error loading and preparing data: {e}")

    # ---------------------------
    # Filter hedge universe if specified
    # ---------------------------
    if hedge_universe is not None:
        # Ensure target stock is included
        hedge_universe = list(hedge_universe)  # Convert to list if not already
        if target_stock not in hedge_universe:
            hedge_universe.append(target_stock)
            if verbose:
                log(f"✓ Added target stock '{target_stock}' to hedge universe")
        
        # Check which stocks from universe are available in data
        available_stocks = df.columns.tolist()
        valid_universe = [stock for stock in hedge_universe if stock in available_stocks]
        missing_stocks = [stock for stock in hedge_universe if stock not in available_stocks]
        
        if missing_stocks:
            if verbose:
                log(f"⚠️  Warning: {len(missing_stocks)} stocks from hedge universe not found in data:")
                for stock in missing_stocks[:10]:  # Show first 10
                    log(f"   • {stock}")
                if len(missing_stocks) > 10:
                    log(f"   • ... and {len(missing_stocks) - 10} more")
        
        if len(valid_universe) < 2:
            raise ValueError(f"Not enough valid stocks in hedge universe. Need at least 2, got {len(valid_universe)}")
        
        # Filter dataframe to hedge universe
        df = df[valid_universe]
        
        if verbose:
            log(f"✓ Filtered to hedge universe: {len(df.columns)} stocks (from {len(available_stocks)} total)")
    else:
        if verbose:
            log(f"✓ Using full universe: {len(df.columns)} stocks")

    # ---------------------------
    # Enhanced data quality validation
    # ---------------------------
    # Check each stock has sufficient valid price data during the analysis period
    min_observations = max(30, int(len(df) * 0.5))  # At least 30 obs or 50% of period
    stocks_to_remove = []
    
    for stock in df.columns:
        stock_data = df[stock].dropna()
        valid_prices = stock_data[stock_data > 0]  # Remove zero/negative prices
        
        if len(valid_prices) < min_observations:
            stocks_to_remove.append(stock)
            if verbose:
                log(f"⚠️  Removing {stock}: only {len(valid_prices)} valid observations (need ≥{min_observations})")
    
    # Remove stocks with insufficient data
    if stocks_to_remove:
        df = df.drop(columns=stocks_to_remove)
        if verbose:
            log(f"Removed {len(stocks_to_remove)} stocks with insufficient data")
    
    # Check we still have enough stocks after cleaning
    if len(df.columns) < 2:
        raise ValueError(f"Insufficient stocks with valid data. Need at least 2, got {len(df.columns)}. "
                        f"Try expanding your hedge universe or adjusting the date range.")
    
    # Check target stock still exists and has valid data
    if target_stock not in df.columns:
        raise ValueError(f"Target stock '{target_stock}' was removed due to insufficient valid data "
                        f"during period {start_date} to {end_date}")

    # Winsorize extreme returns (cap at specified percentiles) 
    # Use np.log with zero handling
    with np.errstate(divide='ignore', invalid='ignore'):
        log_prices = np.log(df.replace(0, np.nan))  # Replace zeros with NaN before log
    
    returns = log_prices.diff().dropna()

    if len(returns) == 0:
        raise ValueError("No valid return data after cleaning. Check your data quality.")

    if apply_outlier_treatment:
        if verbose:
            log("Applying outlier treatment...")
        for col in returns.columns:
            q_lower, q_upper = returns[col].quantile([outlier_quantiles[0], outlier_quantiles[1]])
            outliers_count = ((returns[col] < q_lower) | (returns[col] > q_upper)).sum()
            if outliers_count > 0:
                returns[col] = returns[col].clip(q_lower, q_upper)
                if verbose:
                    log(f"  Capped {outliers_count} outliers in {col}")
        
        # Reconstruct price data from cleaned returns
        cleaned_log_prices = returns.cumsum() + log_prices.iloc[0]
        price_data = np.exp(cleaned_log_prices)
        if verbose:
            log(f"✓ Outlier treatment applied using {outlier_quantiles[0]:.1%}-{outlier_quantiles[1]:.1%} quantiles")
    else:
        # Use original data without outlier treatment
        price_data = df.copy()
        if verbose:
            log("✓ Outlier treatment skipped")

    if verbose:
        log(f"✓ Data prepared: {len(price_data)} observations, {len(price_data.columns)} stocks")

    # ---------------------------
    # Compute log returns and covariance matrix
    # ---------------------------
    log_prices = np.log(price_data)
    returns = log_prices.diff().dropna()


    # ---------------------------
    # Apply benchmark neutrality if specified
    # ---------------------------
    betas = None
    if apply_benchmark_discount:
        if not benchmark_stock or benchmark_stock not in returns.columns:
            raise ValueError(f"Benchmark stock '{benchmark_stock}' not found for benchmark neutrality.")
        
        log(f"✓ Applying benchmark neutrality constraint using: {benchmark_stock}")
        
        # Calculate betas relative to the benchmark
        market_var = returns[benchmark_stock].var()
        if market_var < 1e-12: # Avoid division by zero
            raise ValueError("Benchmark stock has zero variance, cannot calculate betas.")
            
        cov_with_market = returns.cov()[benchmark_stock]
        betas = cov_with_market / market_var
        
        # Remove benchmark from the dataset for optimization
        returns = returns.drop(columns=[benchmark_stock])
        price_data = price_data.drop(columns=[benchmark_stock])
        
        # Re-check that target stock is not the benchmark
        if target_stock not in price_data.columns:
            raise ValueError(f"Target stock '{target_stock}' cannot be the same as the benchmark stock.")

    R = returns.values
    Sigma = np.cov(R, rowvar=False)

    # ---------------------------
    # Extract target and hedge basket components
    # ---------------------------
    n = Sigma.shape[0]
    target_index = price_data.columns.get_loc(target_stock)
    Sigma_JJ = Sigma[target_index, target_index]
    Sigma_Jx = np.delete(Sigma[target_index, :], target_index)
    Sigma_xx = np.delete(np.delete(Sigma, target_index, axis=0), target_index, axis=1)

    # ---------------------------
    # Define MIQP variables
    # ---------------------------
    w_x = cp.Variable(n - 1)
    z = cp.Variable(n - 1, boolean=True)
    if shorting_allowed:
        z_pos = cp.Variable(n - 1, boolean=True)
        z_neg = cp.Variable(n - 1, boolean=True)

    # ---------------------------
    # Objective function
    # ---------------------------
    # Use psd_wrap only for large universes (>200 stocks) to avoid numerical issues
    if n > 200:
        if verbose:
            print(f"Large universe detected ({n} stocks). Using psd_wrap to handle potential numerical issues.")
        objective = cp.Minimize(cp.quad_form(w_x, cp.psd_wrap(Sigma_xx)) - 2 * Sigma_Jx @ w_x)
    else:
        objective = cp.Minimize(cp.quad_form(w_x, Sigma_xx) - 2 * Sigma_Jx @ w_x)

    # ---------------------------
    # Constraints
    # ---------------------------
    if not shorting_allowed:
        constraints = [
            cp.sum(w_x) == 1,
            w_x >= 0,
            w_x <= max_weight * z,
            w_x >= min_weight * z,
            cp.sum(z) <= max_positions
        ]
    else:
        constraints = [
            cp.sum(w_x) == 1,
            # Link z to z_pos and z_neg: a position is active if it's either long or short
            z_pos + z_neg == z,
            # Enforce weight bounds for long and short positions separately
            w_x >= min_weight * z_pos - max_weight * z_neg,
            w_x <= max_weight * z_pos - min_weight * z_neg,
            # Cardinality constraint
            cp.sum(z) <= max_positions
        ]

    # Add beta neutrality constraint if specified
    if apply_benchmark_discount and betas is not None:
        beta_target = betas[target_stock]
        
        # Get betas for the hedge universe, ensuring correct order
        hedge_stock_names = price_data.columns.drop(target_stock)
        betas_hedge = betas[hedge_stock_names].values

        constraints.append(betas_hedge @ w_x == beta_target)
        log(f"✓ Added beta neutrality constraint (Target Beta: {beta_target:.3f})")



    # ---------------------------
    # Solve the problem with comprehensive error handling
    # ---------------------------
    if verbose:
        log("Solving optimization problem...")
    
    prob = cp.Problem(objective, constraints)
    
    try:
        # Try to solve with CPLEX first
        try:
            prob.solve(
                solver=cp.CPLEX,
                verbose=verbose,
                cplex_params={
                    "timelimit": 300,              # Optional: limit solve time to 5 minutes
                    "mip.tolerances.mipgap": 0.01  # Optional: allow 1% optimality gap
                }
            )
            solver_used = "CPLEX"
            
        except cp.SolverError as e:
            if "CPLEX" in str(e):
                log("CPLEX solver not available or failed. Trying alternative solvers...")
                
                # Try ECOS_BB as backup for MIQP
                try:
                    prob.solve(solver=cp.ECOS_BB, verbose=verbose)
                    solver_used = "ECOS_BB"
                    log("Using ECOS_BB solver as fallback")
                except cp.SolverError:
                    log("ECOS_BB also failed. Trying SCIP...")
                    try:
                        prob.solve(solver=cp.SCIP, verbose=verbose)
                        solver_used = "SCIP"
                        log("Using SCIP solver as fallback")
                    except cp.SolverError:
                        log("All MIQP solvers failed. This problem requires a mixed-integer solver.")
                        raise RuntimeError("No suitable MIQP solver available. Please install CPLEX, SCIP, or ECOS_BB.")
            else:
                raise
                
        # Check solver status and provide detailed feedback
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            status_messages = {
                cp.INFEASIBLE: "Problem is infeasible - constraints cannot be satisfied simultaneously",
                cp.UNBOUNDED: "Problem is unbounded - objective can be improved indefinitely", 
                cp.INFEASIBLE_INACCURATE: "Problem appears infeasible (inaccurate)",
                cp.UNBOUNDED_INACCURATE: "Problem appears unbounded (inaccurate)",
                cp.USER_LIMIT: "Solver stopped due to user limits (time/iterations)",
                None: "Solver failed to return a status"
            }
            
            status_msg = status_messages.get(prob.status, f"Unknown solver status: {prob.status}")
            
            if prob.status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]:
                log("Optimization failed - problem is infeasible. Possible causes:")
                log("  • Constraints are too restrictive (min/max weights, max positions)")
                log("  • Target stock has insufficient correlation with universe")
                log("  • Date range too short or data quality issues")
                raise ValueError(f"Optimization infeasible: {status_msg}")
                
            elif prob.status in [cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE]:
                raise ValueError(f"Optimization unbounded: {status_msg}")
                
            elif prob.status == cp.USER_LIMIT:
                log("Solver hit time/iteration limits but may have found a solution...")
                if w_x.value is None:
                    raise ValueError("Solver stopped due to limits without finding a solution")
                else:
                    log("Using best solution found within limits")
                    
            else:
                raise ValueError(f"Solver failed: {status_msg}")
        
        log(f"Optimization completed with status: {prob.status} (using {solver_used})")
        
    except ImportError as e:
        raise ImportError(f"Required optimization solver not available: {e}. Please install CPLEX or other MIQP solvers.")
    except MemoryError:
        raise MemoryError("Insufficient memory for optimization. Try reducing universe size or max positions.")
    except Exception as e:
        if "solver" in str(e).lower():
            raise RuntimeError(f"Solver error: {e}")
        else:
            raise RuntimeError(f"Unexpected optimization error: {e}")

    # ---------------------------
    # Enhanced performance diagnostics and output
    # ---------------------------
    if w_x.value is None:
        raise ValueError("Solver completed but returned no solution. Check problem formulation and constraints.")

    w_x_opt = w_x.value
    w_full = np.insert(w_x_opt, target_index, -1)

    # Calculate comprehensive performance metrics
    # The returns of the zero-investment portfolio (hedge - target) are the tracking residuals
    residuals = R @ w_full
    target_returns = R[:, target_index]

    # Core metrics
    port_variance = Sigma_JJ - 2 * Sigma_Jx @ w_x_opt + w_x_opt @ Sigma_xx @ w_x_opt
    correlation = np.corrcoef(residuals, target_returns)[0, 1]
    r_squared = correlation ** 2
    tracking_error = np.std(residuals)
    tracking_error_annualized = tracking_error * np.sqrt(252)

    # Risk metrics
    var_95 = np.percentile(residuals, 5)
    cvar_95 = residuals[residuals <= var_95].mean()
    max_residual = np.max(np.abs(residuals))

    # Portfolio composition analysis
    active_positions = np.sum(np.abs(w_x_opt) > 1e-6)
    total_gross_exposure = np.sum(np.abs(w_x_opt))
    largest_position = np.max(np.abs(w_x_opt))

    # ---------------------------
    # Filter out zero weights and prepare output
    # ---------------------------
    stock_names = price_data.columns
    weights_series = pd.Series(w_full, index=stock_names)
    weights_series_excluding_target = weights_series.drop(target_stock)

    # Filter out zero weights
    non_zero_weights = weights_series_excluding_target[weights_series_excluding_target.abs() > 1e-6]
    hedge_weights = non_zero_weights.sort_values(key=abs, ascending=False)
    
     # CSV output (only non-zero weights)
    hedge_weights.to_csv(output_file_name, index=True, header=False)

    # Quality checks
    quality_warnings = []
    if tracking_error_annualized > high_tracking_error_threshold:
        quality_warnings.append(f"High tracking error ({tracking_error_annualized:.3f} > {high_tracking_error_threshold:.2f})")
    if active_positions < min_positions_warning:
        quality_warnings.append(f"Very concentrated hedge ({active_positions} positions)")

    # Prepare metrics dictionary
    metrics = {
        'correlation': correlation,
        'r_squared': r_squared,
        'tracking_error_daily': tracking_error,
        'tracking_error_annualized': tracking_error_annualized,
        'portfolio_variance': port_variance,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'max_residual': max_residual,
        'active_positions': active_positions,
        'total_gross_exposure': total_gross_exposure,
        'largest_position': largest_position
    }

    # ---------------------------
    # Save outputs (if requested)
    # ---------------------------
    if save_files:
        # Create detailed output
        detailed_output = []
        detailed_output.append("CORRELATION HEDGE OPTIMIZATION RESULTS")
        detailed_output.append("=" * 60)
        detailed_output.append("")

        # Optimization metadata
        detailed_output.append("OPTIMIZATION SETUP:")
        detailed_output.append(f"  Target Stock:               {target_stock}")
        if apply_benchmark_discount:
            detailed_output.append(f"  Benchmark Stock:            {benchmark_stock} (discounted)")
        detailed_output.append(f"  Optimization Date:          {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        detailed_output.append(f"  Data Period:                {start_date} to {end_date}")
        detailed_output.append(f"  Data Points:                {len(R)}")
        detailed_output.append(f"  Universe Size:              {len(price_data.columns)} stocks")
        detailed_output.append(f"  Max Positions Allowed:      {max_positions}")
        detailed_output.append(f"  Min Weight:                 {min_weight:.3f}")
        detailed_output.append(f"  Max Weight:                 {max_weight:.3f}")
        detailed_output.append(f"  Shorting Allowed:           {shorting_allowed}")
        detailed_output.append(f"  Solver Status:              {prob.status}")
        detailed_output.append("")

        # Performance metrics
        detailed_output.append("PERFORMANCE METRICS:")
        detailed_output.append(f"  Correlation with target:    {correlation:.4f}")
        detailed_output.append(f"  R-squared:                  {r_squared:.4f}")
        detailed_output.append(f"  Tracking error (daily):     {tracking_error:.6f}")
        detailed_output.append(f"  Tracking error (annualized): {tracking_error_annualized:.4f}")
        detailed_output.append(f"  Portfolio variance:         {port_variance:.6f}")
        detailed_output.append("")

        # Risk metrics
        detailed_output.append("RISK METRICS:")
        detailed_output.append(f"  95% VaR (daily):           {var_95:.6f}")
        detailed_output.append(f"  95% CVaR (daily):          {cvar_95:.6f}")
        detailed_output.append(f"  Maximum residual:          {max_residual:.6f}")
        detailed_output.append("")

        # Portfolio composition
        detailed_output.append("PORTFOLIO COMPOSITION:")
        detailed_output.append(f"  Active positions:          {active_positions}/{len(w_x_opt)}")
        detailed_output.append(f"  Gross exposure:            {total_gross_exposure:.4f}")
        detailed_output.append(f"  Largest position:          {largest_position:.4f}")
        detailed_output.append("")

        # Quality warnings
        if quality_warnings:
            detailed_output.append("QUALITY WARNINGS:")
            for warning in quality_warnings:
                detailed_output.append(f"  • {warning}")
            detailed_output.append("")

        # Hedge weights (only non-zero)
        detailed_output.append(f"OPTIMAL HEDGE WEIGHTS:")
        detailed_output.append(f"Total non-zero positions: {len(hedge_weights)}")
        detailed_output.append("-" * 50)
        detailed_output.append(f"{'Stock':<15} {'Weight':<10} {'% of Hedge':<12}")
        detailed_output.append("-" * 50)

        for stock, weight in hedge_weights.items():
            pct_of_hedge = (abs(weight) / total_gross_exposure) * 100
            detailed_output.append(f"{stock:<15} {weight:>8.4f}   {pct_of_hedge:>8.1f}%")

        detailed_output.append("")
        detailed_output.append(f"Sum of absolute weights: {hedge_weights.abs().sum():.4f}")

        # Save files
        detailed_filename = output_file_name.replace('.csv', '_detailed.txt')
        with open(detailed_filename, 'w') as f:
            f.write('\n'.join(detailed_output))


        if verbose:
            log(f"\n💾 Results saved:")
            log(f"   • Hedge weights (non-zero only): {output_file_name}")
            log(f"   • Detailed analysis: {detailed_filename}")

    # Console output
    if verbose:
        log(f"\n🎯 HEDGE OPTIMIZATION COMPLETE")
        log(f"   Target: {target_stock} | Correlation: {correlation:.4f} | Tracking Error: {tracking_error_annualized:.4f}")
        log(f"   Active positions: {active_positions} | Largest position: {largest_position:.4f}")

        log(f"\n🎯 OPTIMAL HEDGE WEIGHTS (excluding {target_stock} at -1.0000):")
        log("-" * 50)
        for stock, weight in hedge_weights.items():
            log(f"   {stock:<12} {weight:>8.4f}")
        
        log(f"   • {len(hedge_weights)} positions (zeros excluded)")



    # Return results
    return {
        'hedge_weights': hedge_weights,
        'metrics': metrics,
        'solver_status': prob.status,
        'quality_warnings': quality_warnings
    }


# ---------------------------
# Default configuration for direct execution
# ---------------------------
if __name__ == "__main__":
    # Example hedge universe (comment out to use all stocks)
    # hedge_universe = ["JPM UN", "BAC UN", "C UN", "WFC UN", "GS UN", "MS UN"]
    
    # Default parameters
    results = optimize_correlation_hedge(
        target_stock="JPM UN",
        input_file="C:/pythonContainer/BasketBuster/Transfer/BPCData/spy2020.csv",
        output_file_name="test.csv",
        max_positions=25,
        min_weight=0.01,
        max_weight=0.20,
        shorting_allowed=False,
        apply_outlier_treatment=False,
        outlier_quantiles=(0.01, 0.99),
        start_date="2024/01/01",
        end_date="2024-10-10",
        high_tracking_error_threshold=0.10,
        min_positions_warning=5,
        save_files=True,
        verbose=True,
        apply_benchmark_discount=True,
        benchmark_stock="SPY",
        hedge_universe=None,
        log_callback=None   
    )



# # From another file:
# from FactorModels.CorrelationHedgeV1 import optimize_correlation_hedge

# # Run with custom parameters
# results = optimize_correlation_hedge(
#     target_stock="AAPL US",
#     input_file="my_data.csv",
#     max_positions=15,
#     save_files=False,  # Don't save files
#     verbose=False      # Quiet mode
# )

# # Access results
# hedge_weights = results['hedge_weights']
# correlation = results['metrics']['correlation']
# warnings = results['quality_warnings']