# 📓 Jupyter Notebooks Documentation

This directory contains Jupyter notebooks for data analysis and visualization.

## 📊 Notebooks Overview

### 1. 01_data_exploration.ipynb

Initial data exploration and analysis notebook.

#### Key Components

##### Data Loading

```python
import os
import sys
import numpy as np
import pandas as pd
from scripts.read import read
```

- Sets up Python path for custom modules
- Imports required libraries
- Uses custom read utility from scripts

##### News Data Analysis

- Loads news headlines dataset
- Calculates headline statistics:
  - Mean length: 73.12 characters
  - Standard deviation: 40.73
  - Min length: 3 characters
  - Max length: 512 characters
  - Median length: 64 characters

##### Technical Analysis

- Uses FinancialAnalyzer class for:
  - Moving Average calculations
  - RSI (Relative Strength Index) analysis
  - Technical indicators visualization

#### Visualizations

- Moving Average plots
- RSI trend analysis
- Price movement charts

## 🔧 Dependencies

- numpy
- pandas
- matplotlib
- custom scripts:
  - read.py
  - financial_analyzer.py

## 📈 Usage Example

```python
# Load data
file_path = "../data/raw_analyst_ratings.csv"
news_data = read(file_path)

# Analyze headlines
headline_lengths = news_data['headline'].apply(len)
np.round(headline_lengths.describe(), 3)

# Technical Analysis
analyzer = FinancialAnalyzer(stock_data)
analyzer.plot_moving_average()
analyzer.plot_rsi()
```

## 📝 Notes

- Ensure correct file paths for data loading
- Required custom modules must be in Python path
- Some visualizations require specific data formats
