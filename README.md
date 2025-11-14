# BasketBuster

BasketBuster is a Windows/PyQt6 desktop toolkit for researching custom equity baskets, running correlation-hedge optimizations, and exporting FX-aware attribution backtests. The screenshots below walk through the main surfaces in the app and explain what each panel represents.

---

## 1. Main Dashboard

These four captures show the primary “Basket Builder” workspace where you load price data, select stocks/baskets, and preview CSV constituents.

- **Configuration + selectors** (left rail) – browse basket files, search tickers, toggle stocks/baskets, and review basket membership tables.
- **Dates & data paths** (top ribbon) – pick analysis ranges and point to the latest `Prices`, `AssetInfo`, `FXRates`, and basket directories.
- **Integrated Plotly chart** (right panel) – after clicking _Plot Graph_, the app renders cumulative USD returns for every selected stock/basket; the embedded log window underneath streams progress messages and validation warnings.

<img width="3840" height="2064" alt="BB1" src="https://github.com/user-attachments/assets/85de3bd7-0cd0-4da5-a7b6-2c143a928129" />
<img width="3840" height="2064" alt="BB2" src="https://github.com/user-attachments/assets/82221462-16a8-401e-9935-ab1e95bfd9eb" />
<img width="3840" height="2064" alt="BB3" src="https://github.com/user-attachments/assets/9d9f65aa-84d2-4b7c-a448-20d4504a974b" />
<img width="2632" height="1997" alt="BB4" src="https://github.com/user-attachments/assets/47bfd259-a85f-4afc-b280-b43a88ff832c" />

---

## 2. Hedge Optimization Output

The MVP tab solves a constrained MIQP hedge (via `cvxpy` + CPLEX/SCIP/ECOS_BB). Once complete, the UI summarizes solver status, optimal weights, correlation, tracking error, and any quality warnings before it writes `*.csv` and `_detailed.txt` artifacts to the basket folder.

<img width="450" height="793" alt="BBhedgeResult" src="https://github.com/user-attachments/assets/6df2b40b-926c-4795-8071-4e27b4a28cc9" />

---

## 3. Backtest Workbook Snapshot

Running the threaded backtest builds an Excel file per run. The `Bucket Attribution Summary` sheet (pictured) aggregates local-price vs FX contributions so you can see how each basket behaves against the selected start/end dates and target stock.

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

These visuals provide the quickest way to understand how BasketBuster stitches price data, FX curves, and basket definitions into a single researcher-friendly UI. For full functionality (hedge optimization, logging, background workers, configuration persistence), run `python BBMain.py` and load your own CSVs following the layouts shown above.

