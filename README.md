# BasketBuster

BasketBuster is a Windows/PyQt6 desktop toolkit for researching custom equity baskets, running correlation-hedge optimizations, and exporting FX-aware attribution backtests. The screenshots below walk through the main surfaces in the app and explain what each panel represents.

---

## 1. Main Dashboard (BB1–BB4)

Each numbered screenshot corresponds to a specific page within the PyQt6 window.

- **Basket Performance tab (plotting surface).**  
  This is the plotting-focused home screen: select stocks/baskets on the left, choose the analysis window, hit **Plot Graph**, and the embedded Plotly view (right) renders cumulative USD returns. The log window underneath confirms data prep steps.
  <img width="3840" height="2064" alt="BB1" src="https://github.com/user-attachments/assets/85de3bd7-0cd0-4da5-a7b6-2c143a928129" />

- **Baskets / Data tab.**  
  This page is dedicated to data management: browse basket CSVs, inspect constituents in tabular form, and set/write the canonical paths for stock prices, FX rates, asset info, and basket directories.
  <img width="3840" height="2064" alt="BB2" src="https://github.com/user-attachments/assets/82221462-16a8-401e-9935-ab1e95bfd9eb" />

- **MVP Hedger tab.**  
  Configure and launch the correlation-hedge optimizer from here: target stock, quantile winsorization, min/max weights, shorting toggle, hedge universe, benchmark discount, and solver verbosity. The “Run Hedge” button kicks off the threaded MIQP worker and the dedicated log pane captures solver progress.
  <img width="3840" height="2064" alt="BB3" src="https://github.com/user-attachments/assets/9d9f65aa-84d2-4b7c-a448-20d4504a974b" />

- **Backtest tab.**  
  The final tab controls attribution runs: a single-select target stock list, basket checklist, search boxes, start/end dates, and Excel output naming.
  <img width="2632" height="1997" alt="BB4" src="https://github.com/user-attachments/assets/47bfd259-a85f-4afc-b280-b43a88ff832c" />

---

## 2. Hedge Optimization Output (BBhedgeResult)

The MVP tab solves a constrained MIQP hedge (via `cvxpy` + CPLEX/SCIP/ECOS_BB). Once complete, the UI summarizes solver status, optimal weights, correlation, tracking error, and any quality warnings before it writes `*.csv` and `_detailed.txt` to the basket folder.
<img width="450" height="793" alt="BBhedgeResult" src="https://github.com/user-attachments/assets/6df2b40b-926c-4795-8071-4e27b4a28cc9" />

---

## 3. Backtest Workbook Snapshot (BBbacktest)

Running the backtest builds an Excel file per run. The `Bucket Attribution Summary` sheet (pictured) aggregates local-price vs FX contributions so you can see how each basket behaves against the selected start/end dates and target stock.
<img width="899" height="124" alt="BBbacktest" src="https://github.com/user-attachments/assets/4e0d819c-5856-4010-aea5-3a422ccd3775" />

---

## 4. Data Input Examples

To replicate the workflow, supply three core CSVs (examples below):

1. **Asset Info** – ticker-to-currency mapping (used for USD conversion).  
   <img width="1111" height="771" alt="AssetinfoExample" src="https://github.com/user-attachments/assets/97e72277-215b-479d-a6e2-3eaea8c4f20e" />

2. **Stock Prices** – wide panel indexed by `Date` with one column per ticker.  
   <img width="1149" height="412" alt="stkdataExample" src="https://github.com/user-attachments/assets/f5b2bfd7-3b84-4e76-abc6-2fcc8731e68c" />

3. **FX Rates** – daily currency series covering every `CCY` referenced in the asset info file.  
   <img width="1005" height="309" alt="FXdataExample" src="https://github.com/user-attachments/assets/c04b0181-df4a-4e34-9638-7e4c6d674518" />

---

## 5. How cumulative returns are built

1. **Load & align price levels**  
    - `prepareData()` (in `BBGraph.py`) reads the wide prices file (`Date` index) and filters it to the selected start/end dates.  
    - Asset info (`Name`, `CCY`) is mapped into a currency dictionary, while `FXRatesTRIM.csv` is sliced to the same date range.

2. **Convert to USD**  
    - `convert_prices_to_usd()` forward/back fills the FX series per currency and multiplies local prices by their FX rate (after validating non-null/non-zero rates).  
    - Any conversion issues (missing currencies, zero FX values) are logged but don’t crash the run.

3. **Calculate returns**  
    - For each selected stock, `prices_usd[stock].pct_change().fillna(0)` produces a daily return series.  
    - For baskets, the tool reads the CSV weights, cleans duplicates/empty rows, aligns weights to the price columns, and computes a daily weighted return via matrix multiplication.

4. **Accumulate**  
    - Both stock and basket daily returns are compounded: `cumulative = (1 + returns).cumprod() - 1`.  
    - `plot_cumulative_returns()` plots the time series, while `save_cumulative_returns_to_csv()` writes them to `cumulative_returns_usd.csv`.

This flow guarantees that every curve on the Plotly chart represents USD total returns over the exact same date window.

---

## 6. How the MVP Hedger works

1. **Data cleaning & validation** (`BBMVP.py`)  
    - Reads the price panel, enforces presence of a `Date` column, removes duplicate/empty series, and restricts the dataset to the requested date range.  
    - Optionally filters to a user-provided hedge universe and drops stocks with insufficient observations or zero/negative prices.

2. **Return transformation & outlier handling**  
    - Converts prices to log returns; if “Apply Outlier Treatment” is toggled, returns are winsorized between the selected quantiles and re-integrated back into a cleaned price panel.  
    - If benchmark discounting is enabled, the benchmark column is removed and a beta-neutrality constraint is set up using the benchmark’s variance/covariance.

3. **Optimization model**  
    - Defines decision variables `w_x` (weights) and binary selectors `z` controlling which names are active.  
    - Objective: minimize `w_xᵀ Σ_xx w_x - 2 Σ_Jx w_x`, which targets the lowest tracking error between the hedge and the target.  
    - Constraints enforce weight bounds, cardinality, optional shorting logic, and (if enabled) beta neutrality.

4. **Solver execution**  
    - Attempts to solve with CPLEX first; falls back to ECOS_BB or SCIP if unavailable.  
    - Provides detailed status messages (OPTIMAL, INFEASIBLE, USER_LIMIT, etc.) and surfaces warnings when tracking error is high or positions are too concentrated.

5. **Outputs**  
    - Non-zero hedge weights are exported to CSV; a `_detailed.txt` companion file captures metrics (correlation, R², tracking error, VaR/CVaR, position counts).  
    - The GUI log mirrors every step, and the summary panel (BBhedgeResult) displays the final stats so you can quickly evaluate the hedge quality.

---
