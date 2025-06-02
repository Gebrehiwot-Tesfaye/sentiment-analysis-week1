import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from textblob import TextBlob
# import talib as yf
import yfinance as yf

import pandas_ta as ta


def load_historical_data(ticker):
    stock_data=pd.read_csv(f'../data/yfinance_data/{ticker}_historical_data.csv')
    return stock_data



def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def numberOfArticlesWithSentimentAnalysis(news_data):
    sentiment_counts = news_data['sentiment_score_word'].value_counts().sort_index()

    # Define colors for each category
    colors = {'positive': 'green', 'negative': 'red', 'neutral': 'yellow'}

    # Create the bar plot with specified colors
    sentiment_counts.plot(kind="bar", figsize=(10, 4), title='Sentiment Analysis',
                        xlabel='Sentiment categories', ylabel='Number of Published Articles',
                        color=[colors[category] for category in sentiment_counts.index])

    plt.show()



def getSentimentAnalysisOfPublisher(news_data, target_publisher):
    colors = {'positive': 'green', 'negative': 'red', 'neutral': 'blue'}
    # Filter data for the target publisher
    publisher_data = news_data[news_data['publisher'] == target_publisher]
    sentiment_counts = publisher_data['sentiment_score_word'].value_counts().sort_index()

    sentiment_counts.plot(kind="bar", figsize=(10, 4), title=f'Sentiment Analysis of {target_publisher}',
                      xlabel='Sentiment categories', ylabel='Number of Published Articles',
                      color=[colors[category] for category in sentiment_counts.index])



def checkMissingValueOfHistoricalDataset(stock_data_aapl,stock_data_amzn,stock_data_goog,stock_data_meta,stock_data_msft,stock_data_nvda,stock_data_tsla):
    combined_df = pd.concat([stock_data_aapl.isnull().sum(),
                            stock_data_goog.isnull().sum(),
                            stock_data_amzn.isnull().sum(),
                            stock_data_msft.isnull().sum(),
                            stock_data_meta.isnull().sum(),
                            stock_data_nvda.isnull().sum(),
                            stock_data_tsla.isnull().sum()],
                            axis=0)
    combined_df.head()
    

def calculateDescriptiveStatisticsOfHistoricalData(stock_data_aapl,stock_data_amzn,stock_data_goog,stock_data_meta,stock_data_msft,stock_data_nvda,stock_data_tsla):
    aapl_stats = stock_data_aapl['Close'].describe().to_frame('AAPL')
    goog_stats = stock_data_goog['Close'].describe().to_frame('GOOG')
    amzn_stats = stock_data_amzn['Close'].describe().to_frame('AMZN')
    msft_stats = stock_data_msft['Close'].describe().to_frame('MSFT')
    meta_stats = stock_data_meta['Close'].describe().to_frame('META')
    nvda_stats = stock_data_nvda['Close'].describe().to_frame('NVDA')
    tsla_stats = stock_data_tsla['Close'].describe().to_frame('TSLA')
    combined_stats = pd.concat([aapl_stats, goog_stats,amzn_stats,msft_stats,meta_stats,nvda_stats,tsla_stats], axis=1)
    return combined_stats



def analysisClosingPriceWithDate(stock_data_aapl,stock_data_amzn,stock_data_goog,stock_data_meta,stock_data_msft,stock_data_nvda):
    # Create subplots for side-by-side display
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))  # Adjust figsize as needed

    axs[0,0].plot(stock_data_aapl['Date'], stock_data_aapl['Close'], label='Close',color='green')
    axs[0,0].set_title('AAPL')
    axs[0,0].legend()

    axs[0,1].plot(stock_data_amzn['Date'], stock_data_amzn['Close'], label='AMZN')
    axs[0,1].set_title('AMZN')
    axs[0,1].legend()


    axs[0,2].plot(stock_data_goog['Date'], stock_data_goog['Close'], label='Close',color='yellow')
    axs[0,2].set_title('GOOG')
    axs[0,2].legend()


    axs[1,0].plot(stock_data_nvda['Date'], stock_data_nvda['Close'], label='Close',color='brown')
    axs[1,0].set_title('NVDA')
    axs[1,0].legend()
    axs[1,0].set_xlabel('Date')


    axs[1,1].plot(stock_data_msft['Date'], stock_data_msft['Close'], label='Close',color='purple')
    axs[1,1].set_title('MSFT')
    axs[1,1].legend()
    axs[1,1].set_xlabel('Date')

    axs[1,2].plot(stock_data_meta['Date'], stock_data_meta['Close'], label='Close',color='orange')
    axs[1,2].set_title('META')
    axs[1,2].legend()
    axs[1,2].set_xlabel('Date')

    plt.show()




def calculateTechnicalIndicator(stock_data):
    """Calculate technical indicators for a given stock dataset"""
    # Create a copy of the dataframe
    df = stock_data.copy()
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate Simple Moving Average (SMA)
    df['SMA'] = df['Close'].rolling(window=20).mean()
    
    # Calculate other indicators
    df['RSI'] = df.ta.rsi(close='Close', length=14)
    df['EMA'] = df.ta.ema(close='Close', length=20)
    
    # Calculate MACD
    macd = df.ta.macd(close='Close')
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    
    # Forward fill any NaN values
    df = df.fillna(method='ffill')
    
    return df

def technicalIndicatorsVsClosingPrice(stock_data_aapl, stock_data_amzn, stock_data_goog, 
                                    stock_data_meta, stock_data_msft, stock_data_nvda, indicator):
    """Plot technical indicators against closing prices"""
    # Process all stock data first
    processed_data = {
        'AAPL': calculateTechnicalIndicator(stock_data_aapl),
        'AMZN': calculateTechnicalIndicator(stock_data_amzn),
        'GOOG': calculateTechnicalIndicator(stock_data_goog),
        'META': calculateTechnicalIndicator(stock_data_meta),
        'MSFT': calculateTechnicalIndicator(stock_data_msft),
        'NVDA': calculateTechnicalIndicator(stock_data_nvda)
    }
    
    # Verify indicator exists in processed data
    for ticker, data in processed_data.items():
        if indicator not in data.columns:
            raise KeyError(f"Indicator '{indicator}' not found in {ticker} data. Available columns: {data.columns.tolist()}")
    
    # Create plot
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    plot_config = [
        ('AAPL', (0,0), 'green'), ('AMZN', (0,1), 'blue'), ('GOOG', (0,2), 'yellow'),
        ('NVDA', (1,0), 'brown'), ('MSFT', (1,1), 'purple'), ('META', (1,2), 'orange')
    ]
    
    # Plot each stock
    for ticker, (row, col), color in plot_config:
        data = processed_data[ticker]
        ax = axs[row, col]
        
        # Plot closing price and indicator
        ax.plot(data['Date'], data['Close'], label='Closing price', color=color)
        ax.plot(data['Date'], data[indicator], label=indicator, color='red')
        
        # Configure plot
        ax.set_title(ticker)
        ax.legend()
        if row == 1:  # Only set xlabel for bottom row
            ax.set_xlabel('Date')
            
    plt.tight_layout()
    plt.show()

def closingPriceRelativeStrengthIndex(stock_data_aapl, stock_data_amzn, stock_data_goog, stock_data_meta, stock_data_msft, stock_data_nvda):
    # Process all stock data first
    processed_data = {
        'AAPL': calculateTechnicalIndicator(stock_data_aapl),
        'AMZN': calculateTechnicalIndicator(stock_data_amzn),
        'GOOG': calculateTechnicalIndicator(stock_data_goog),
        'META': calculateTechnicalIndicator(stock_data_meta),
        'MSFT': calculateTechnicalIndicator(stock_data_msft),
        'NVDA': calculateTechnicalIndicator(stock_data_nvda)
    }
    
    fig, axs = plt.subplots(6, 2, gridspec_kw={"height_ratios": [1, 1, 1, 1, 1, 1]}, figsize=(16, 22))

    # For AAPL
    axs[0][0].plot(processed_data['AAPL']['Date'], processed_data['AAPL']['Close'], label="Close")
    axs[0][0].set_title("AAPL Stock Price")
    axs[0][0].legend()
    axs[1][0].axhline(y=70, color='r', linestyle="--")
    axs[1][0].axhline(y=30, color='g', linestyle="--")
    axs[1][0].plot(processed_data['AAPL']['Date'], processed_data['AAPL']['RSI'], color='orange', label="RSI")
    axs[1][0].legend()

    # For GOOG
    axs[0][1].plot(processed_data['GOOG']['Date'], processed_data['GOOG']['Close'], label="Close")
    axs[0][1].set_title("GOOG Stock Price")
    axs[0][1].legend()
    axs[1][1].axhline(y=70, color='r', linestyle="--")
    axs[1][1].axhline(y=30, color='g', linestyle="--")
    axs[1][1].plot(processed_data['GOOG']['Date'], processed_data['GOOG']['RSI'], color='orange', label="RSI")
    axs[1][1].legend()

    # For AMZN
    axs[2][0].plot(processed_data['AMZN']['Date'], processed_data['AMZN']['Close'], label="Close")
    axs[2][0].set_title("AMZN Stock Price")
    axs[2][0].legend()
    axs[3][0].axhline(y=70, color='r', linestyle="--")
    axs[3][0].axhline(y=30, color='g', linestyle="--")
    axs[3][0].plot(processed_data['AMZN']['Date'], processed_data['AMZN']['RSI'], color='orange', label="RSI")
    axs[3][0].legend()

    # For NVDA
    axs[2][1].plot(processed_data['NVDA']['Date'], processed_data['NVDA']['Close'], label="Close")
    axs[2][1].set_title("NVDA Stock Price")
    axs[2][1].legend()
    axs[3][1].axhline(y=70, color='r', linestyle="--")
    axs[3][1].axhline(y=30, color='g', linestyle="--")
    axs[3][1].plot(processed_data['NVDA']['Date'], processed_data['NVDA']['RSI'], color='orange', label="RSI")
    axs[3][1].legend()

    # For MSFT
    axs[4][0].plot(processed_data['MSFT']['Date'], processed_data['MSFT']['Close'], label="Close")
    axs[4][0].set_title("MSFT Stock Price")
    axs[4][0].legend()
    axs[5][0].axhline(y=70, color='r', linestyle="--")
    axs[5][0].axhline(y=30, color='g', linestyle="--")
    axs[5][0].plot(processed_data['MSFT']['Date'], processed_data['MSFT']['RSI'], color='orange', label="RSI")
    axs[5][0].legend()

    # For META
    axs[4][1].plot(processed_data['META']['Date'], processed_data['META']['Close'], label="Close")
    axs[4][1].set_title("META Stock Price")
    axs[4][1].legend()
    axs[5][1].axhline(y=70, color='r', linestyle="--")
    axs[5][1].axhline(y=30, color='g', linestyle="--")
    axs[5][1].plot(processed_data['META']['Date'], processed_data['META']['RSI'], color='orange', label="RSI")
    axs[5][1].legend()
    
    plt.tight_layout()
    plt.show()

def closingPriceMovingAverageConvergenceDivergence(stock_data_aapl, stock_data_amzn, stock_data_goog, stock_data_meta, stock_data_msft, stock_data_nvda, stock_data_tsla):
    # Process all stock data first
    processed_data = {
        'AAPL': calculateTechnicalIndicator(stock_data_aapl),
        'AMZN': calculateTechnicalIndicator(stock_data_amzn),
        'GOOG': calculateTechnicalIndicator(stock_data_goog),
        'META': calculateTechnicalIndicator(stock_data_meta),
        'MSFT': calculateTechnicalIndicator(stock_data_msft),
        'NVDA': calculateTechnicalIndicator(stock_data_nvda),
        'TSLA': calculateTechnicalIndicator(stock_data_tsla)
    }
    
    fig, axs = plt.subplots(6, 2, gridspec_kw={"height_ratios": [1, 1, 1, 1, 1, 1]}, figsize=(16, 22))

    # For AAPL
    axs[0][0].plot(processed_data['AAPL']['Date'], processed_data['AAPL']['Close'], label="Close")
    axs[0][0].set_title("AAPL Stock Price")
    axs[0][0].legend()
    axs[1][0].axhline(y=5, color='r', linestyle="--")
    axs[1][0].axhline(y=-5, color='g', linestyle="--")
    axs[1][0].plot(processed_data['AAPL']['Date'], processed_data['AAPL']['MACD'], color='orange', label="MACD")
    axs[1][0].plot(processed_data['AAPL']['Date'], processed_data['AAPL']['MACD_Signal'], color='r', label="MACD_Signal")
    axs[1][0].legend()

    # For GOOG
    axs[0][1].plot(processed_data['GOOG']['Date'], processed_data['GOOG']['Close'], label="Close")
    axs[0][1].set_title("GOOG Stock Price")
    axs[0][1].legend()
    axs[1][1].axhline(y=5, color='r', linestyle="--")
    axs[1][1].axhline(y=-5, color='g', linestyle="--")
    axs[1][1].plot(processed_data['GOOG']['Date'], processed_data['GOOG']['MACD'], color='orange', label="MACD")
    axs[1][1].plot(processed_data['GOOG']['Date'], processed_data['GOOG']['MACD_Signal'], color='r', label="MACD_Signal")
    axs[1][1].legend()
    
    # For AMZN
    axs[2][0].plot(processed_data['AMZN']['Date'], processed_data['AMZN']['Close'], label="Close")
    axs[2][0].set_title("AMZN Stock Price")
    axs[2][0].legend()
    axs[3][0].axhline(y=5, color='r', linestyle="--")
    axs[3][0].axhline(y=-5, color='g', linestyle="--")
    axs[3][0].plot(processed_data['AMZN']['Date'], processed_data['AMZN']['MACD'], color='orange', label="MACD")
    axs[3][0].plot(processed_data['AMZN']['Date'], processed_data['AMZN']['MACD_Signal'], color='r', label="MACD_Signal")
    axs[3][0].legend()
    
    # For NVDA
    axs[2][1].plot(processed_data['NVDA']['Date'], processed_data['NVDA']['Close'], label="Close")
    axs[2][1].set_title("NVDA Stock Price")
    axs[2][1].legend()
    axs[3][1].axhline(y=5, color='r', linestyle="--")
    axs[3][1].axhline(y=-5, color='g', linestyle="--")
    axs[3][1].plot(processed_data['NVDA']['Date'], processed_data['NVDA']['MACD'], color='orange', label="MACD")
    axs[3][1].plot(processed_data['NVDA']['Date'], processed_data['NVDA']['MACD_Signal'], color='r', label="MACD_Signal")
    axs[3][1].legend()
    
    # For MSFT
    axs[4][0].plot(processed_data['MSFT']['Date'], processed_data['MSFT']['Close'], label="Close")
    axs[4][0].set_title("MSFT Stock Price")
    axs[4][0].legend()
    axs[5][0].axhline(y=5, color='r', linestyle="--")
    axs[5][0].axhline(y=-5, color='g', linestyle="--")
    axs[5][0].plot(processed_data['MSFT']['Date'], processed_data['MSFT']['MACD'], color='orange', label="MACD")
    axs[5][0].plot(processed_data['MSFT']['Date'], processed_data['MSFT']['MACD_Signal'], color='r', label="MACD_Signal")
    axs[5][0].legend()
    
    # For META
    axs[4][1].plot(processed_data['META']['Date'], processed_data['META']['Close'], label="Close")  
    axs[4][1].set_title("META Stock Price")
    axs[4][1].legend()
    axs[5][1].axhline(y=5, color='r', linestyle="--")
    axs[5][1].axhline(y=-5, color='g', linestyle="--")
    axs[5][1].plot(processed_data['META']['Date'], processed_data['META']['MACD'], color='orange', label="MACD")
    axs[5][1].plot(processed_data['META']['Date'], processed_data['META']['MACD_Signal'], color='r', label="MACD_Signal")
    axs[5][1].legend()
    
    # For TSLA
    axs[6][0].plot(processed_data['TSLA']['Date'], processed_data['TSLA']['Close'], label="Close")
    axs[6][0].set_title("TSLA Stock Price")
    axs[6][0].legend()
    axs[7][0].axhline(y=5, color='r', linestyle="--")
    axs[7][0].axhline(y=-5, color='g', linestyle="--")  
    axs[7][0].plot(processed_data['TSLA']['Date'], processed_data['TSLA']['MACD'], color='orange', label="MACD")
    axs[7][0].plot(processed_data['TSLA']['Date'], processed_data['TSLA']['MACD_Signal'], color='r', label="MACD_Signal")
    axs[7][0].legend()
    
    # For TSLA
    
    
    

    # Similar pattern for other stocks...
    # (AMZN, NVDA, MSFT, META plotting code follows the same pattern)
    
    plt.tight_layout()
    plt.show()

def calculatePortfolioWeightAndPerformance():
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt import risk_models
    from pypfopt import expected_returns

    tickers =['AAPL','AMZN','GOOG','MSFT','NVDA','META','TSLA']
    # Load data from each ticker file
    dataframes = [load_historical_data(ticker) for ticker in tickers]

    # Combine dataframes into a single DataFrame
    combined_data = pd.concat(dataframes, axis=1)['Close']

    new_column_names = ['AAPL', 'AMZN','GOOG', 'META','MSFT','NVDA','TSLA'] 
    combined_data.columns = new_column_names

    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(combined_data)
    S = risk_models.sample_cov(combined_data)

    # Optimize for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    weights = dict(zip(['AAPL', 'AMZN','GOOG', 'META','MSFT','NVDA','TSLA'],[round(value, 2) for value in weights.values()]))

    # Print Portfolio weights
    print("Portfolio Weights:")
    print(weights)


    # Calculate and print portfolio performance
    print("\nPortfolio Performance:")
    ef.portfolio_performance(verbose=True)
    
    
    