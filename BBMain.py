import sys
import json
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import cplex
import plotly.io as pio
from BBGraph import *
from BBMVP import *
from BBAnalysis import *

from PyQt6.QtGui import QStandardItemModel, QStandardItem, QMovie
from PyQt6.QtCore import QDate, Qt, QTimer, QObject, QThread, pyqtSignal
from PyQt6.uic import loadUi
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QListWidgetItem, QLabel
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtCore import QDate, Qt
from PyQt6.uic import loadUi

from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings

from datetime import datetime
import traceback


# Worker for MVP Optimization
class MvpWorker(QObject):
    """
    Runs the MVP optimization in a separate thread to prevent GUI freezing.
    """
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    log_message = pyqtSignal(str, str)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.params['log_callback'] = self.log_gui

    def log_gui(self, message, level="INFO"):
        """Emits a signal to log messages on the main GUI thread."""
        self.log_message.emit(message, level)

    def run(self):
        """Executes the optimization and emits signals on completion or error."""
        try:
            # The main work is done here
            result = optimize_correlation_hedge(**self.params)
            self.finished.emit(result)
        except Exception as e:
            # Format a detailed error message with traceback
            error_details = f"Error in optimization thread: {e}\n{traceback.format_exc()}"
            self.error.emit(error_details)


# Worker for Backtest Analysis
class BacktestWorker(QObject):
    """
    Runs the backtest analysis in a separate thread to prevent GUI freezing.
    """
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    log_message = pyqtSignal(str, str)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.params['log_callback'] = self.log_gui

    def log_gui(self, message, level="INFO"):
        """Emits a signal to log messages on the main GUI thread."""
        self.log_message.emit(message, level)

    def run(self):
        """Executes the backtest analysis and emits signals on completion or error."""
        try:
            # The main work is done here
            output_file = run_backtest_analysis(**self.params)
            self.finished.emit(output_file)
        except Exception as e:
            # Format a detailed error message with traceback
            error_details = f"Error in backtest analysis thread: {e}\n{traceback.format_exc()}"
            self.error.emit(error_details)


# Constants
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")
print(cp.installed_solvers())

class BB(QMainWindow):
    prices_usd = None
    available_stocks = []  # Store available stocks
    selected_stocks = []   # Store selected stocks
    available_baskets = [] # Store available baskets
    selected_baskets = []  # Store selected baskets
    
    # Backtest-specific variables
    available_backtest_stocks = []  # Store available stocks for backtest
    selected_backtest_stocks = []   # Store selected stocks for backtest
    available_backtest_baskets = [] # Store available baskets for backtest
    selected_backtest_baskets = []  # Store selected baskets for backtest

    def __init__(self):
        super(BB, self).__init__()
        loadUi("C:/pythonContainer/BasketBuster/Transfer/BB.ui", self)
        self.plotWebView.settings().setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        
        self.showMaximized()
        
        # Set dark background for the plot area
        self.set_initial_plot_background()

        # Load persisted config if exists
        self.load_config()

        # Keep track of temp files to delete later
        self._temp_files = []

        # Set default baskets path
        default_baskets_path = os.path.join(os.path.dirname(__file__), "BPCBaskets")
        if not self.basketsPathLineEdit.text():
            self.basketsPathLineEdit.setText(default_baskets_path)
            
        # Set default backtest output path
        default_backtest_output_path = os.path.join(os.path.dirname(__file__), "BPCBaskets")
        if not self.backtestOutputFolder.text():
            self.backtestOutputFolder.setText(default_backtest_output_path)


        # Connect signals
        self.stkDataPathButton.clicked.connect(self.stkDataPath)
        self.fxDataPathButton.clicked.connect(self.fxDataPath)
        self.assetInfoDataPathButton.clicked.connect(self.assetInfoDataPath)
        self.basketsPathButton.clicked.connect(self.basketsPath)
        self.loadBasketsButton.clicked.connect(self.loadBaskets)
        self.loadBasketsButton2.clicked.connect(self.loadBaskets)        
        self.CSVFiles1.itemClicked.connect(lambda item: self.displayCSV(item, self.cvsTable1))
        self.CSVFiles2.itemClicked.connect(lambda item: self.displayCSV(item, self.cvsTable2))

        self.graphButton.clicked.connect(self.calculate_and_plot)
        self.mvpStkDataPathButton.clicked.connect(self.mvpStkDataPath)
        self.mvpRunHedgeButton.clicked.connect(self.mvpHedge)
        # Connect stock selection widget
        self.StockNamesSelect.itemChanged.connect(self.on_stock_selection_changed)
        
        # Connect basket selection widget (CSVFiles3)
        self.CSVFiles3.itemChanged.connect(self.on_basket_selection_changed)
        
        # Connect unselect all stocks button (only for stocks)
        self.unselectAllStocksButton.clicked.connect(self.unselect_all_stocks)
        
        # Connect stock search functionality
        self.stockSearchLineEdit.textChanged.connect(self.filter_stocks)
        
        # Load stocks when stock data path changes
        self.stkDataPathLineEdit.textChanged.connect(self.load_available_stocks)
        
        # Load stocks on startup if path is already set
        if self.stkDataPathLineEdit.text():
            self.load_available_stocks()
            
        # Connect backtest-specific controls
        self.backtestStockSearchLineEdit.textChanged.connect(self.filter_backtest_stocks)
        self.backtestStockNamesSelect.itemChanged.connect(self.on_backtest_stock_selection_changed)
        self.backtestUnselectAllStocksButton.clicked.connect(self.unselect_all_backtest_stocks)
        self.CSVFiles4.itemChanged.connect(self.on_backtest_basket_selection_changed)
        self.loadBasketsButton3.clicked.connect(self.load_backtest_baskets)
        self.runBacktestButton.clicked.connect(self.run_backtest)
        
        # Load backtest stocks when stock data path changes
        self.stkDataPathLineEdit.textChanged.connect(self.load_available_backtest_stocks)
        
        # Load backtest stocks on startup if path is already set
        if self.stkDataPathLineEdit.text():
            self.load_available_backtest_stocks()
            
        # Initialize log
        self.log("Application started successfully")

        # To hold reference to the worker and thread
        self.mvp_thread = None
        self.mvp_worker = None
        self.backtest_thread = None
        self.backtest_worker = None

    def log(self, message, level="INFO"):
        """Add a message to the log window with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {level}: {message}"
        
        # Add to GUI log
        self.logTextEdit.append(formatted_message)
        self.logTextEditMVP.append(formatted_message)
        self.backtestLogTextEdit.append(formatted_message)
        # print to console for debugging
        print(formatted_message)
        
        # Auto-scroll to bottom
        scrollbar = self.logTextEdit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def handleMvpFinished(self, result):
        """Handles the successful completion of the MVP optimization."""
        self.log("MVP Hedge completed successfully")
        self.mvpRunHedgeButton.setEnabled(True)
        self.mvpRunHedgeButton.setText("Run Hedge")

    def handleMvpError(self, error_message):
        """Handles errors from the MVP optimization thread."""
        self.log("An error occurred during MVP Hedge.", "ERROR")
        self.log(error_message, "ERROR")
        self.mvpRunHedgeButton.setEnabled(True)
        self.mvpRunHedgeButton.setText("Run Hedge")

    def handleBacktestFinished(self, output_file):
        """Handles the successful completion of the backtest analysis."""
        self.log("Backtest analysis completed successfully")
        self.log(f"Results saved to: {output_file}")
        self.runBacktestButton.setEnabled(True)
        self.runBacktestButton.setText("Run Backtest")

    def handleBacktestError(self, error_message):
        """Handles errors from the backtest analysis thread."""
        self.log("An error occurred during backtest analysis.", "ERROR")
        self.log(error_message, "ERROR")
        self.runBacktestButton.setEnabled(True)
        self.runBacktestButton.setText("Run Backtest")

    def load_available_stocks(self):
        """Load available stocks from the selected CSV file and populate the stock list widget"""
        stock_file_path = self.stkDataPathLineEdit.text()
        
        if not stock_file_path or not os.path.exists(stock_file_path):
            self.StockNamesSelect.clear()
            self.available_stocks = []
            self.stockSearchLineEdit.clear()
            return
            
        try:
            # Read the header to get column names
            df = pd.read_csv(stock_file_path, nrows=0)
            # Remove 'Date' column and get stock tickers
            stock_columns = [col for col in df.columns if col.lower() != 'date']
            self.available_stocks = sorted(stock_columns)
            
            # Clear search when loading new stocks
            self.stockSearchLineEdit.clear()
            
            # Populate the list widget with checkable items
            self.StockNamesSelect.clear()
            for stock in self.available_stocks:
                item = QListWidgetItem(stock)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                self.StockNamesSelect.addItem(item)
                
            self.log(f"Loaded {len(self.available_stocks)} stocks from {os.path.basename(stock_file_path)}")
            
        except Exception as e:
            self.log(f"Error loading stocks from {stock_file_path}: {e}", "ERROR")
            self.StockNamesSelect.clear()
            self.available_stocks = []

    def filter_stocks(self):
        search_text = self.stockSearchLineEdit.text().lower().strip()
        
        # If search is empty, show all stocks
        if not search_text:
            for i in range(self.StockNamesSelect.count()):
                item = self.StockNamesSelect.item(i)
                item.setHidden(False)
            return
        
        # Hide/show items based on search
        visible_count = 0
        for i in range(self.StockNamesSelect.count()):
            item = self.StockNamesSelect.item(i)
            stock_name = item.text().lower()
            
            # Check if search text is contained in stock name
            if search_text in stock_name:
                item.setHidden(False)
                visible_count += 1
            else:
                item.setHidden(True)

    def clear_stock_search(self):
        self.stockSearchLineEdit.clear()
        self.filter_stocks()

    def on_stock_selection_changed(self, item):
        self.update_selected_stocks()

    def update_selected_stocks(self):
        """Update the list of selected stocks based on checkbox states"""
        self.selected_stocks = []
        for i in range(self.StockNamesSelect.count()):
            item = self.StockNamesSelect.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                self.selected_stocks.append(item.text())
        
    def unselect_all_stocks(self):
        """Uncheck all stocks in the StockNamesSelect widget"""
        for i in range(self.StockNamesSelect.count()):
            item = self.StockNamesSelect.item(i)
            item.setCheckState(Qt.CheckState.Unchecked)
        
        self.update_selected_stocks()
        self.log("All stocks unselected")

    def on_basket_selection_changed(self, item):
        """Handle when a basket is selected/deselected"""
        self.update_selected_baskets()

    def update_selected_baskets(self):
        """Update the list of selected baskets based on checkbox states"""
        self.selected_baskets = []
        for i in range(self.CSVFiles3.count()):
            item = self.CSVFiles3.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                self.selected_baskets.append(item.text())
        
    def load_config(self):
        """Load configuration with comprehensive error handling and validation"""
        if not os.path.exists(CONFIG_FILE):
            self.log("No configuration file found, using defaults", "INFO")
            return
            
        try:
            # Check file size and readability
            if os.path.getsize(CONFIG_FILE) == 0:
                self.log("Configuration file is empty, using defaults", "WARNING")
                return
                
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if not isinstance(config, dict):
                self.log("Invalid configuration format, using defaults", "ERROR")
                return
            
            # Validate and set file paths with existence checks
            file_path_fields = {
                "stkDataPath": self.stkDataPathLineEdit,
                "fxDataPath": self.fxDataPathLineEdit,
                "basketsPath": self.basketsPathLineEdit,
                "assetInfoDataPath": self.assetInfoDataPathLineEdit,
                "mvpStockDataPath": self.mvpStockDataPath
            }
            
            for config_key, widget in file_path_fields.items():
                path = config.get(config_key, "")
                if path:
                    if os.path.exists(path):
                        widget.setText(path)
                    else:
                        self.log(f"Configuration path not found: {config_key} = {path}", "WARNING")
                        widget.setText("")  # Clear invalid path
                else:
                    widget.setText("")
            
            # Handle special case for MVP output path (uses basketsPath as fallback)
            mvp_output_path = config.get("mvpOutputPath", config.get("basketsPath", ""))
            if mvp_output_path and os.path.exists(mvp_output_path):
                self.mvpOutputPath.setText(mvp_output_path)
            else:
                self.mvpOutputPath.setText("")
            
            # Validate and set dates with error handling
            date_fields = {
                "startDate": self.startDate,
                "endDate": self.endDate,
                "mvpStartDate": self.mvpStartDate,
                "mvpEndDate": self.mvpEndDate,
                "backtestStartDate": self.backtestStartDate,
                "backtestEndDate": self.backtestEndDate
            }
            
            for config_key, widget in date_fields.items():
                date_str = config.get(config_key, "")
                if date_str:
                    try:
                        # Try to parse the date to validate it
                        parsed_date = QDate.fromString(date_str, "yyyy-MM-dd")
                        if parsed_date.isValid():
                            widget.setDate(parsed_date)
                        else:
                            self.log(f"Invalid date format in config: {config_key} = {date_str}", "WARNING")
                    except Exception as e:
                        self.log(f"Error parsing date {config_key}: {e}", "WARNING")
            
            # Set text fields with validation
            text_fields = {
                "mvpTargetStock": self.mvpTargetStock,
                "backtestOutputFolder": self.backtestOutputFolder,
            }
            
            for config_key, widget in text_fields.items():
                value = config.get(config_key, "")
                if isinstance(value, str):
                    widget.setText(value)
                else:
                    widget.setText("")
                    if value:  # Only warn if there was a non-empty value
                        self.log(f"Invalid text value in config: {config_key} = {value}", "WARNING")
                
            self.log("Configuration loaded successfully")
                    
        except json.JSONDecodeError as e:
            self.log(f"Invalid JSON in configuration file: {e}", "ERROR")
        except FileNotFoundError:
            self.log("Configuration file not found", "WARNING")
        except PermissionError:
            self.log("Permission denied reading configuration file", "ERROR")
        except UnicodeDecodeError as e:
            self.log(f"Encoding error reading configuration file: {e}", "ERROR")
        except Exception as e:
            self.log(f"Unexpected error loading configuration: {e}", "ERROR")

    def closeEvent(self, event):
        """Save configuration with comprehensive error handling"""
        try:
            config = {
                "stkDataPath": self.stkDataPathLineEdit.text(),
                "fxDataPath": self.fxDataPathLineEdit.text(),
                "basketsPath": self.basketsPathLineEdit.text(),
                "assetInfoDataPath": self.assetInfoDataPathLineEdit.text(),
                "startDate": str(self.startDate.date().toPyDate()),
                "endDate": str(self.endDate.date().toPyDate()),
                "mvpTargetStock": self.mvpTargetStock.text(),
                "mvpStartDate": str(self.mvpStartDate.date().toPyDate()),
                "mvpEndDate": str(self.mvpEndDate.date().toPyDate()),
                "mvpStockDataPath": self.mvpStockDataPath.text(),
                "mvpOutputPath": self.mvpOutputPath.text(),
                "backtestStartDate": str(self.backtestStartDate.date().toPyDate()),
                "backtestEndDate": str(self.backtestEndDate.date().toPyDate()),
                "backtestOutputFolder": self.backtestOutputFolder.text(),
            }
            
            # Validate configuration before saving
            config_errors = []
            
            # Check that dates are valid
            for date_key in ["startDate", "endDate", "mvpStartDate", "mvpEndDate", "backtestStartDate", "backtestEndDate"]:
                try:
                    pd.to_datetime(config[date_key])
                except Exception:
                    config_errors.append(f"Invalid date: {date_key}")
            
            # Check file paths exist (only warn, don't prevent saving)
            file_paths = ["stkDataPath", "fxDataPath", "assetInfoDataPath", "mvpStockDataPath"]
            for path_key in file_paths:
                path = config.get(path_key, "")
                if path and not os.path.exists(path):
                    self.log(f"Warning: Path in config doesn't exist: {path_key} = {path}", "WARNING")
            
            # Check directory paths
            dir_paths = ["basketsPath", "mvpOutputPath", "backtestOutputFolder"]
            for path_key in dir_paths:
                path = config.get(path_key, "")
                if path and not os.path.isdir(path):
                    self.log(f"Warning: Directory in config doesn't exist: {path_key} = {path}", "WARNING")
            
            if config_errors:
                self.log(f"Configuration validation errors: {', '.join(config_errors)}", "ERROR")
                # Still try to save, but warn user
            
            # Create backup of existing config if it exists
            if os.path.exists(CONFIG_FILE):
                backup_file = CONFIG_FILE + '.backup'
                try:
                    import shutil
                    shutil.copy2(CONFIG_FILE, backup_file)
                except Exception as e:
                    self.log(f"Warning: Could not create config backup: {e}", "WARNING")
            
            # Save configuration
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            
            self.log("Configuration saved successfully")
            
        except PermissionError:
            self.log("Permission denied saving configuration file", "ERROR")
        except OSError as e:
            self.log(f"System error saving configuration: {e}", "ERROR")
        except json.JSONEncodeError as e:
            self.log(f"Error encoding configuration to JSON: {e}", "ERROR")
        except Exception as e:
            self.log(f"Unexpected error saving configuration: {e}", "ERROR")
        
        event.accept()

    def stkDataPath(self):
        StkDataPath = QFileDialog.getOpenFileName(self, 'Open File', './BPCData', filter='CSV Files (*.csv)')
        if StkDataPath[0]:  # Only update if a file was selected
            self.stkDataPathLineEdit.setText(StkDataPath[0])
            self.log(f"Stock data path selected: {os.path.basename(StkDataPath[0])}")

    def mvpStkDataPath(self):
        mvpStkDataPath = QFileDialog.getOpenFileName(self, 'Open File', './BPCData', filter='CSV Files (*.csv)')
        if mvpStkDataPath[0]:  # Only update if a file was selected
            self.mvpStkDataPath.setText(mvpStkDataPath[0])
            self.log(f"Stock data path selected: {os.path.basename(mvpStkDataPath[0])}")

    def fxDataPath(self):
        fxDataPath = QFileDialog.getOpenFileName(self, 'Open File', './BPCData', filter='CSV Files (*.csv)')
        if fxDataPath[0]:
            self.fxDataPathLineEdit.setText(fxDataPath[0])
            self.log(f"FX data path selected: {os.path.basename(fxDataPath[0])}")

    def assetInfoDataPath(self):
        assetInfoDataPath = QFileDialog.getOpenFileName(self, 'Open File', './BPCData', filter='CSV Files (*.csv)')
        if assetInfoDataPath[0]:
            self.assetInfoDataPathLineEdit.setText(assetInfoDataPath[0])
            self.log(f"Asset info path selected: {os.path.basename(assetInfoDataPath[0])}")

    def basketsPath(self):
        baskets_path = QFileDialog.getExistingDirectory(self, 'Select Folder', './BPCBaskets')
        if baskets_path:
            self.basketsPathLineEdit.setText(baskets_path)
            self.log(f"Baskets folder selected: {os.path.basename(baskets_path)}")

    def loadBaskets(self):
        baskets_path = self.basketsPathLineEdit.text()
        if not os.path.isdir(baskets_path):
            self.log(f"Invalid folder path: {baskets_path}", "ERROR")
            return

        csv_files = sorted(f for f in os.listdir(baskets_path) if f.lower().endswith('.csv'))
        self.available_baskets = csv_files
        
        # Load CSVFiles1 and CSVFiles2 as before (for display only)
        self.CSVFiles1.clear()
        self.CSVFiles1.addItems(csv_files)
        self.CSVFiles2.clear()
        self.CSVFiles2.addItems(csv_files)
        
        # Load CSVFiles3 with checkable items for selection
        self.CSVFiles3.clear()
        for basket_file in csv_files:
            item = QListWidgetItem(basket_file)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.CSVFiles3.addItem(item)
        
        self.log(f"Loaded {len(csv_files)} baskets from {os.path.basename(baskets_path)}")

    def load_available_backtest_stocks(self):
        """Load available stocks from the selected CSV file and populate the backtest stock list widget"""
        stock_file_path = self.stkDataPathLineEdit.text()
        
        if not stock_file_path or not os.path.exists(stock_file_path):
            self.backtestStockNamesSelect.clear()
            self.available_backtest_stocks = []
            self.backtestStockSearchLineEdit.clear()
            return
            
        try:
            # Read the header to get column names
            df = pd.read_csv(stock_file_path, nrows=0)
            # Remove 'Date' column and get stock tickers
            stock_columns = [col for col in df.columns if col.lower() != 'date']
            self.available_backtest_stocks = sorted(stock_columns)
            
            # Clear search when loading new stocks
            self.backtestStockSearchLineEdit.clear()
            
            # Populate the list widget with checkable items
            self.backtestStockNamesSelect.clear()
            for stock in self.available_backtest_stocks:
                item = QListWidgetItem(stock)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                self.backtestStockNamesSelect.addItem(item)
                
            self.log(f"Loaded {len(self.available_backtest_stocks)} stocks for backtest analysis")
            
        except Exception as e:
            self.log(f"Error loading backtest stocks from {stock_file_path}: {e}", "ERROR")
            self.backtestStockNamesSelect.clear()
            self.available_backtest_stocks = []

    def filter_backtest_stocks(self):
        search_text = self.backtestStockSearchLineEdit.text().lower().strip()
        
        # If search is empty, show all stocks
        if not search_text:
            for i in range(self.backtestStockNamesSelect.count()):
                item = self.backtestStockNamesSelect.item(i)
                item.setHidden(False)
            return
        
        # Hide/show items based on search
        visible_count = 0
        for i in range(self.backtestStockNamesSelect.count()):
            item = self.backtestStockNamesSelect.item(i)
            stock_name = item.text().lower()
            
            # Check if search text is contained in stock name
            if search_text in stock_name:
                item.setHidden(False)
                visible_count += 1
            else:
                item.setHidden(True)

    def on_backtest_stock_selection_changed(self, item):
        # Only allow one stock selection for backtest
        if item.checkState() == Qt.CheckState.Checked:
            # Uncheck all other items
            for i in range(self.backtestStockNamesSelect.count()):
                other_item = self.backtestStockNamesSelect.item(i)
                if other_item != item and other_item.checkState() == Qt.CheckState.Checked:
                    other_item.setCheckState(Qt.CheckState.Unchecked)
        self.update_selected_backtest_stocks()

    def update_selected_backtest_stocks(self):
        """Update the list of selected backtest stocks based on checkbox states (max 1 stock)"""
        self.selected_backtest_stocks = []
        for i in range(self.backtestStockNamesSelect.count()):
            item = self.backtestStockNamesSelect.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                self.selected_backtest_stocks.append(item.text())
                break  # Only allow one stock
        
    def unselect_all_backtest_stocks(self):
        """Uncheck all stocks in the backtest StockNamesSelect widget"""
        for i in range(self.backtestStockNamesSelect.count()):
            item = self.backtestStockNamesSelect.item(i)
            item.setCheckState(Qt.CheckState.Unchecked)
        
        self.update_selected_backtest_stocks()
        self.log("All backtest stocks unselected")

    def on_backtest_basket_selection_changed(self, item):
        """Handle when a backtest basket is selected/deselected"""
        self.update_selected_backtest_baskets()

    def update_selected_backtest_baskets(self):
        """Update the list of selected backtest baskets based on checkbox states"""
        self.selected_backtest_baskets = []
        for i in range(self.CSVFiles4.count()):
            item = self.CSVFiles4.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                self.selected_backtest_baskets.append(item.text())

    def load_backtest_baskets(self):
        baskets_path = self.basketsPathLineEdit.text()
        if not os.path.isdir(baskets_path):
            self.log(f"Invalid folder path: {baskets_path}", "ERROR")
            return

        csv_files = sorted(f for f in os.listdir(baskets_path) if f.lower().endswith('.csv'))
        self.available_backtest_baskets = csv_files
        
        # Load CSVFiles4 with checkable items for backtest selection
        self.CSVFiles4.clear()
        for basket_file in csv_files:
            item = QListWidgetItem(basket_file)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.CSVFiles4.addItem(item)
        
        self.log(f"Loaded {len(csv_files)} baskets for backtest analysis")

    def run_backtest(self):
        """Run the backtest analysis"""
        self.log("Starting backtest analysis...")
        try:
            # Validate all input fields before processing
            validation_errors = []
            
            # Check if we have baskets selected
            if not self.selected_backtest_baskets:
                validation_errors.append("No baskets selected for backtest analysis")
            
            # Validate file paths exist
            if not self.stkDataPathLineEdit.text().strip() or not os.path.exists(self.stkDataPathLineEdit.text().strip()):
                validation_errors.append("Stock data file not found")
            
            if not self.assetInfoDataPathLineEdit.text().strip() or not os.path.exists(self.assetInfoDataPathLineEdit.text().strip()):
                validation_errors.append("Asset info file not found")
                
            if not self.fxDataPathLineEdit.text().strip() or not os.path.exists(self.fxDataPathLineEdit.text().strip()):
                validation_errors.append("FX data file not found")
            
            # Validate dates
            try:
                start_date = self.backtestStartDate.date().toPyDate()
                end_date = self.backtestEndDate.date().toPyDate()
                
                if start_date >= end_date:
                    validation_errors.append("Start date must be before end date")
            except Exception as e:
                validation_errors.append(f"Invalid date selection: {e}")
            
            # Validate output path
            output_folder = self.backtestOutputFolder.text().strip()
            if not output_folder or not os.path.isdir(output_folder):
                validation_errors.append("Valid output folder is required")
                
            output_name = self.backtestOutputName.text().strip()
            if not output_name:
                validation_errors.append("Output file name is required")
            
            # Report validation errors
            if validation_errors:
                self.log("Validation errors found:", "ERROR")
                for error in validation_errors:
                    self.log(f"  • {error}", "ERROR")
                return
            
            # Disable button and show loading state
            self.runBacktestButton.setEnabled(False)
            self.runBacktestButton.setText("Running...")
            self.log("Starting backtest analysis...")
            
            # Construct output filename
            output_filename = os.path.join(
                output_folder,
                output_name + ".xlsx"
            )
            
            self.log(f"Output file: {output_filename}")
            
            # Collect parameters for the worker
            params = {
                "prices_file": self.stkDataPathLineEdit.text().strip(),
                "currency_file": self.assetInfoDataPathLineEdit.text().strip(),
                "fx_file": self.fxDataPathLineEdit.text().strip(),
                "basket_folder": self.basketsPathLineEdit.text().strip(),
                "basket_files": self.selected_backtest_baskets,
                "start_date": str(start_date),
                "end_date": str(end_date),
                "target_stock": self.selected_backtest_stocks[0] if self.selected_backtest_stocks else None,  # Use selected stock or None
                "output_file": output_filename
            }

            # Setup and start the thread
            self.backtest_thread = QThread()
            self.backtest_worker = BacktestWorker(params)
            self.backtest_worker.moveToThread(self.backtest_thread)

            # Connect signals and slots
            self.backtest_thread.started.connect(self.backtest_worker.run)
            self.backtest_worker.finished.connect(self.handleBacktestFinished)
            self.backtest_worker.error.connect(self.handleBacktestError)
            self.backtest_worker.log_message.connect(self.log)

            # Clean up after the thread is done
            self.backtest_worker.finished.connect(self.backtest_thread.quit)
            self.backtest_worker.finished.connect(self.backtest_worker.deleteLater)
            self.backtest_thread.finished.connect(self.backtest_thread.deleteLater)

            self.backtest_thread.start()

        except Exception as e:
            self.log(f"Unexpected error during backtest analysis: {e}", "ERROR")
            self.log(f"Technical details: {traceback.format_exc()}", "ERROR")
            self.runBacktestButton.setEnabled(True)
            self.runBacktestButton.setText("Run Backtest")

    def displayCSV(self, item, table):
        filename = item.text()
        folder_path = self.basketsPathLineEdit.text()
        file_path = os.path.join(folder_path, filename)

        try:
            df = pd.read_csv(file_path, header=None)
            df.columns = ["Name", "Weight"]
            model = QStandardItemModel()
            model.setColumnCount(len(df.columns))
            model.setRowCount(len(df.index))
            model.setHorizontalHeaderLabels(df.columns)

            for row in range(len(df.index)):
                for col in range(len(df.columns)):
                    value = str(df.iat[row, col])
                    model.setItem(row, col, QStandardItem(value))

            table.setModel(model)
            table.resizeColumnsToContents()
        except Exception as e:
            self.log(f"Failed to load CSV {filename}: {e}", "ERROR")

    def calculate_and_plot(self):
        """Combined function that calculates data and generates plot in one step"""
        self.log("Starting data calculation and plotting...")
        
        # First, run data calculation
        data_calc_success = self._calculate_data_internal()
        if not data_calc_success:
            return  # Error messages already logged in _calculate_data_internal
        
        # If data calculation succeeded, proceed with plotting
        self._show_plot_internal()

    def _calculate_data_internal(self):
        """Internal data calculation function that returns success/failure status"""
        try:
            # Validate date inputs
            start = self.startDate.date().toPyDate()
            end = self.endDate.date().toPyDate()
            
            # Check date logic
            if start >= end:
                self.log("Start date must be before end date", "ERROR")
                return False
            
            # Check if dates are too far in the future or past
            from datetime import date
            today = date.today()
            if start > today:
                self.log("Start date cannot be in the future", "WARNING")
            if end > today:
                self.log("End date is in the future - data may not be available", "WARNING")
                
        except Exception as e:
            self.log(f"Invalid date selection: {e}", "ERROR")
            return False

        # Check if we have either stocks or baskets to analyze
        if not self.selected_stocks and not self.selected_baskets:
            self.log("No stocks or baskets selected! Please select at least one stock or basket.", "WARNING")
            return False

        # Validate file paths
        file_paths = {
            "Stock Data": self.stkDataPathLineEdit.text(),
            "Asset Info": self.assetInfoDataPathLineEdit.text(),
            "FX Data": self.fxDataPathLineEdit.text()
        }
        
        missing_paths = []
        for name, path in file_paths.items():
            if not path or not os.path.exists(path):
                missing_paths.append(f"{name}: {path}")
        
        if missing_paths:
            self.log("Missing or invalid file paths:", "ERROR")
            for path_info in missing_paths:
                self.log(f"  • {path_info}", "ERROR")
            return False

        # Use selected stocks (empty list if none selected)
        stocks_to_process = self.selected_stocks if self.selected_stocks else []

        try:
            self.prices_usd = prepareData(
                stocks_to_process, 
                str(self.stkDataPathLineEdit.text()), 
                str(self.basketsPathLineEdit.text()),
                str(self.assetInfoDataPathLineEdit.text()), 
                str(self.fxDataPathLineEdit.text()),
                start, end, self.log
            )

            self.log("Data calculation completed successfully")
            return True
            
        except FileNotFoundError as e:
            self.log(f"File not found: {e}", "ERROR")
            return False
        except ValueError as e:
            self.log(f"Data validation error: {e}", "ERROR")
            return False
        except pd.errors.EmptyDataError as e:
            self.log(f"Empty data file: {e}", "ERROR")
            return False
        except pd.errors.ParserError as e:
            self.log(f"File parsing error: {e}", "ERROR")
            return False
        except MemoryError:
            self.log("Insufficient memory to load data. Try reducing the date range or number of assets.", "ERROR")
            return False
        except Exception as e:
            self.log(f"Unexpected error during data calculation: {e}", "ERROR")
            import traceback
            self.log(f"Technical details: {traceback.format_exc()}", "ERROR")
            return False

    def _show_plot_internal(self):
        """Internal plotting function"""
        try:
            # Data should already be available from calculation step
            if self.prices_usd is None:
                self.log("Data calculation failed - cannot generate plot", "ERROR")
                return

            # Validate baskets path if baskets are selected
            baskets_path = str(self.basketsPathLineEdit.text()).strip()
            if self.selected_baskets and (not baskets_path or not os.path.isdir(baskets_path)):
                self.log("Invalid baskets folder path. Please select a valid baskets folder.", "ERROR")
                return

            # Use selected stocks and baskets
            stocks_to_plot = self.selected_stocks if self.selected_stocks else []
            baskets_to_plot = self.selected_baskets if self.selected_baskets else []
            
            # Validate that selected stocks exist in data
            if stocks_to_plot:
                missing_stocks = [stock for stock in stocks_to_plot if stock not in self.prices_usd.columns]
                if missing_stocks:
                    self.log(f"Warning: Selected stocks not found in data: {', '.join(missing_stocks)}", "WARNING")
                    stocks_to_plot = [stock for stock in stocks_to_plot if stock in self.prices_usd.columns]
                    
                if not stocks_to_plot and not baskets_to_plot:
                    self.log("No valid stocks or baskets available for plotting.", "ERROR")
                    return

            # Call plot function with comprehensive error handling
            try:
                fig = plot(
                    self.prices_usd,
                    stocks_to_plot,
                    baskets_path,
                    baskets_to_plot,
                    output_file='cumulative_returns_usd.csv',
                    log_callback=self.log
                )
                
                # Validate that we got a valid figure
                if fig is None:
                    raise ValueError("Plot function returned None")

                if not hasattr(fig, 'data') or len(fig.data) == 0:
                    raise ValueError("Plot contains no data traces")

            except FileNotFoundError as e:
                self.log(f"File not found during plotting: {e}", "ERROR")
                return
            except ValueError as e:
                self.log(f"Data validation error during plotting: {e}", "ERROR")
                return
            except Exception as e:
                self.log(f"Error in plot generation: {e}", "ERROR")
                return

            # Generate HTML with error handling
            try:
                html = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')
                
                if not html or len(html.strip()) == 0:
                    raise ValueError("Generated HTML is empty")
                
            except Exception as e:
                self.log(f"Error converting plot to HTML: {e}", "ERROR")
                # Try fallback with inline plotly
                try:
                    html = pio.to_html(fig, full_html=True, include_plotlyjs=True)
                    self.log("Using inline Plotly.js as fallback", "WARNING")
                except Exception as e2:
                    self.log(f"Fallback HTML generation also failed: {e2}", "ERROR")
                    return

            # Inject comprehensive dark styling
            dark_css = """
            <style>
                body { 
                    background-color: #1e1e1e !important; 
                    margin: 0; 
                    padding: 0; 
                    font-family: Arial, sans-serif;
                }
                .plotly-graph-div {
                    background-color: transparent !important;
                }
            </style>
            """

            try:
                html = html.replace('<head>', f'<head>{dark_css}')
            except Exception as e:
                self.log(f"Warning: Could not inject dark theme CSS: {e}", "WARNING")

            # Display in WebView with error handling
            try:
                self.plotWebView.setHtml(html)
                
            except Exception as e:
                self.log(f"Error displaying plot in web view: {e}", "ERROR")
                # Try to show a simple error message in the web view
                error_html = f"""
                <html>
                <body style="background-color: #1e1e1e; color: #ffffff; font-family: Arial;">
                    <h3>Plot Display Error</h3>
                    <p>Could not display the generated plot: {str(e)}</p>
                    <p>Please check the log for more details.</p>
                </body>
                </html>
                """
                try:
                    self.plotWebView.setHtml(error_html)
                except Exception:
                    pass  # If even error display fails, give up silently
            
        except MemoryError:
            self.log("Insufficient memory to generate plot. Try reducing the number of assets or date range.", "ERROR")
        except Exception as e:
            self.log(f"Unexpected error during plot generation: {e}", "ERROR")
            import traceback
            self.log(f"Technical details: {traceback.format_exc()}", "ERROR")

    def set_initial_plot_background(self):
        """Set a dark background for the plot area on startup"""
        initial_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { 
                    background-color: #1e1e1e; 
                    margin: 0; 
                    padding: 0; 
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    font-family: Arial, sans-serif;
                    color: #ffffff;
                }
                .placeholder {
                    text-align: center;
                    opacity: 0.6;
                }
            </style>
        </head>
        <body>
            <div class="placeholder">
                <h3>Plot Area</h3>
                <p>Select stocks or baskets and click "Plot Graph" to display charts</p>
            </div>
        </body>
        </html>
        """
        self.plotWebView.setHtml(initial_html)

    def mvpHedge(self):
        self.log("Starting MVP Hedge...")
        try:
            # Validate all input fields before processing
            validation_errors = []
            
            # Check required text fields
            required_fields = {
                "Target Stock": self.mvpTargetStock.text().strip(),
                "Output Name": self.mvpOutputName.text().strip(),
                "Stock Data Path": self.mvpStockDataPath.text().strip(),
                "Output Path": self.mvpOutputPath.text().strip()
            }
            
            for field_name, value in required_fields.items():
                if not value:
                    validation_errors.append(f"{field_name} is required")
            
            # Validate file paths exist
            if self.mvpStockDataPath.text().strip() and not os.path.exists(self.mvpStockDataPath.text().strip()):
                validation_errors.append(f"Stock data file not found: {self.mvpStockDataPath.text()}")
            
            if self.mvpOutputPath.text().strip() and not os.path.isdir(self.mvpOutputPath.text().strip()):
                validation_errors.append(f"Output directory not found: {self.mvpOutputPath.text()}")
            
            # Validate numeric inputs
            try:
                max_positions = int(self.mvpMaxPositions.text())
                if max_positions <= 0:
                    validation_errors.append("Max positions must be a positive integer")
            except ValueError:
                validation_errors.append("Max positions must be a valid integer")
            
            try:
                min_weight = float(self.mvpMinWeight.text())
                if min_weight < 0 or min_weight > 1:
                    validation_errors.append("Min weight must be between 0 and 1")
            except ValueError:
                validation_errors.append("Min weight must be a valid number")
            
            try:
                max_weight = float(self.mvpMaxWeight.text())
                if max_weight < 0 or max_weight > 1:
                    validation_errors.append("Max weight must be between 0 and 1")
            except ValueError:
                validation_errors.append("Max weight must be a valid number")
            
            # Check weight logic
            try:
                if float(self.mvpMinWeight.text()) >= float(self.mvpMaxWeight.text()):
                    validation_errors.append("Min weight must be less than max weight")
            except ValueError:
                pass  # Already caught above
            
            # Validate quantiles
            try:
                max_quant = float(self.mvpMaxQuant.text())
                min_quant = float(self.mvpMinQuant.text())
                
                if not (0 < min_quant < max_quant < 1):
                    validation_errors.append("Quantiles must satisfy: 0 < min_quantile < max_quantile < 1")
                    
                mvpQuantiles = (min_quant, max_quant)
            except ValueError:
                validation_errors.append("Quantiles must be valid numbers")
                mvpQuantiles = (0.01, 0.99)  # fallback
            
            # Validate dates
            try:
                start_date = self.mvpStartDate.date().toPyDate()
                end_date = self.mvpEndDate.date().toPyDate()
                
                if start_date >= end_date:
                    validation_errors.append("Start date must be before end date")
            except Exception as e:
                validation_errors.append(f"Invalid date selection: {e}")
            
            # Parse hedge universe
            hedge_universe_text = str(self.mvpHedgeUniverse.text()).strip()
            if hedge_universe_text:
                try:
                    hedgeUniverse = [stock.strip() for stock in hedge_universe_text.split(',') if stock.strip()]
                    if len(hedgeUniverse) == 0:
                        hedgeUniverse = None
                    else:
                        self.log(f"Using hedge universe with {len(hedgeUniverse)} stocks")
                except Exception:
                    validation_errors.append("Invalid hedge universe format (should be comma-separated)")
                    hedgeUniverse = None
            else:
                hedgeUniverse = None
            
# Parse benchmark discount options
            apply_benchmark_discount = self.mvpBenchmarkDiscount.isChecked()
            benchmark_stock = str(self.mvpBenchmark.text()).strip()

            if apply_benchmark_discount and not benchmark_stock:
                validation_errors.append("Benchmark stock is required when discount option is enabled")

            # Report validation errors
            if validation_errors:
                self.log("Validation errors found:", "ERROR")
                for error in validation_errors:
                    self.log(f"  • {error}", "ERROR")
                return
            
            # Disable button and show loading state
            self.mvpRunHedgeButton.setEnabled(False)
            self.mvpRunHedgeButton.setText("Running...")
            self.log("Starting MVP Hedge...")
            
            # Construct output filename
            mvpOutputFileName = os.path.join(
                str(self.mvpOutputPath.text()).strip(),
                str(self.mvpOutputName.text()).strip() + ".csv"
            )
            
            self.log(f"Output file: {mvpOutputFileName}")
            
            # Collect parameters for the worker
            params = {
                "target_stock": str(self.mvpTargetStock.text()).strip(),
                "input_file": str(self.mvpStockDataPath.text()).strip(),
                "output_file_name": mvpOutputFileName,
                "max_positions": int(self.mvpMaxPositions.text()),
                "min_weight": float(self.mvpMinWeight.text()),
                "max_weight": float(self.mvpMaxWeight.text()),
                "shorting_allowed": bool(self.mvpShortingAllowed.isChecked()),
                "apply_outlier_treatment": bool(self.mvpOutlierTreatment.isChecked()),
                "outlier_quantiles": mvpQuantiles,
                "start_date": str(self.mvpStartDate.date().toPyDate()),
                "end_date": str(self.mvpEndDate.date().toPyDate()),
                "high_tracking_error_threshold": 0.50,  # hardcoded in original
                "min_positions_warning": 5,     # hardcoded in original
                "save_files": self.mvpSaveDetails.isChecked(),
                "verbose": self.mvpVerbose.isChecked(),
                "hedge_universe": hedgeUniverse,
                "apply_benchmark_discount": apply_benchmark_discount,
                "benchmark_stock": benchmark_stock,
            }

            # Setup and start the thread
            self.mvp_thread = QThread()
            self.mvp_worker = MvpWorker(params)
            self.mvp_worker.moveToThread(self.mvp_thread)

            # Connect signals and slots
            self.mvp_thread.started.connect(self.mvp_worker.run)
            self.mvp_worker.finished.connect(self.handleMvpFinished)
            self.mvp_worker.error.connect(self.handleMvpError)
            self.mvp_worker.log_message.connect(self.log)

            # Clean up after the thread is done
            self.mvp_worker.finished.connect(self.mvp_thread.quit)
            self.mvp_worker.finished.connect(self.mvp_worker.deleteLater)
            self.mvp_thread.finished.connect(self.mvp_thread.deleteLater)

            self.mvp_thread.start()

        except FileNotFoundError as e:
            self.log(f"File not found: {e}", "ERROR")
            self.mvpRunHedgeButton.setEnabled(True)
            self.mvpRunHedgeButton.setText("Run Hedge")

        except ValueError as e:
            self.log(f"Input validation error: {e}", "ERROR")
            self.mvpRunHedgeButton.setEnabled(True)
            self.mvpRunHedgeButton.setText("Run Hedge")

        except MemoryError:
            self.log("Insufficient memory for optimization. Try reducing the universe size or date range.", "ERROR")
            self.mvpRunHedgeButton.setEnabled(True)
            self.mvpRunHedgeButton.setText("Run Hedge")

        except ImportError as e:
            self.log(f"Missing required solver (CPLEX): {e}", "ERROR")
            self.mvpRunHedgeButton.setEnabled(True)
            self.mvpRunHedgeButton.setText("Run Hedge")

        except Exception as e:
            self.log(f"Unexpected error during MVP Hedge: {e}", "ERROR")
            self.log(f"Technical details: {traceback.format_exc()}", "ERROR")
            self.mvpRunHedgeButton.setEnabled(True)
            self.mvpRunHedgeButton.setText("Run Hedge")


# def optimize_correlation_hedge(
#     target_stock="JPM UN",
#     input_file="FactorModels/data2023.csv",
#     output_file_name="CVXPY CPLEX MIQP v7.csv",
#     max_positions=25,
#     min_weight=0.01,
#     max_weight=0.20,
#     shorting_allowed=False,
#     apply_outlier_treatment=True,
#     outlier_quantiles=(0.01, 0.99),
#     start_date="2023/01/01",
#     end_date="2024-10-10",
#     high_tracking_error_threshold=0.10,
#     min_positions_warning=5,
#     save_files=True,
#     verbose=True,
#     hedge_universe=None  # NEW PARAMETER
# ):

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BB()
    window.setWindowTitle("BasketBuster")
    window.show()
    sys.exit(app.exec())
