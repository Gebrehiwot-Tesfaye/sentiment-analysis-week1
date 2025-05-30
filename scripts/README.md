# 📂 Scripts Documentation

This directory contains utility scripts for data loading and financial analysis.

## 📚 Files Overview

### 1. read.py

A utility module for reading CSV data files.

#### Key Function

```python
read(file_path: str) -> pd.DataFrame
```

- **Purpose**: Loads CSV files into pandas DataFrames
- **Parameters**: `file_path` - Path to the CSV file
- **Returns**: Pandas DataFrame with loaded data
- **Error Handling**: Manages file not found and format errors

### 2. financial_analyzer.py

A comprehensive financial analysis toolkit with various market analysis functions.

#### Key Functions

##### Data Loading

```python
load_historical_data(ticker: str) -> pd.DataFrame
```

- Loads historical market data for specified ticker symbols

##### Sentiment Analysis

```python
get_sentiment(text: str) -> float
numberOfArticlesWithSentimentAnalysis(news_data: pd.DataFrame)
getSentimentAnalysisOfPublisher(news_data: pd.DataFrame, target_publisher: str)
```

- Text sentiment scoring
- Sentiment distribution visualization
- Publisher-specific sentiment analysis

##### Technical Analysis

```python
calculateTechnicalIndicator(stock_data: pd.DataFrame)
```

- Calculates key technical indicators:
  - Simple Moving Average (SMA)
  - Relative Strength Index (RSI)
  - Exponential Moving Average (EMA)
  - MACD (Moving Average Convergence Divergence)

##### Visualization Functions

```python
analysisClosingPriceWithDate()
technicalIndicatorsVsClosingPrice()
closingPriceRelativeStrengthIndex()
closingPriceMovingAverageConvergenceDivergence()
```

- Multiple stock comparison plots
- Technical indicator visualizations
- RSI and MACD analysis charts

##### Portfolio Analysis

```python
calculatePortfolioWeightAndPerformance()
```

- Optimal portfolio weight calculation
- Performance metrics computation
- Sharpe ratio optimization

## 🔧 Dependencies

- pandas
- numpy
- matplotlib
- textblob
- yfinance
- pypfopt (for portfolio optimization)

## 📈 Usage Example

```python
from read import read
from financial_analyzer import load_historical_data, calculateTechnicalIndicator

# Load data
news_data = read('../data/raw/news_data.csv')
stock_data = load_historical_data('AAPL')

# Calculate technical indicators
calculateTechnicalIndicator(stock_data)
```

## 📝 Notes

- Ensure all required data files are in the correct paths
- Technical indicators require valid OHLCV data
- Visualization functions support multiple stock comparisons
