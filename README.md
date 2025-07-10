# Market Trend Analysis Agent

## Overview

The Market Trend Analysis Agent is a machine learning-based agent designed to predict stock price movements using historical market data and technical indicators. The agent uses a combination of Random Forest, XGBoost, and other machine learning models to forecast the next day's stock price movement (up or down). Additionally, various technical indicators like SMA, RSI, MACD, and Volatility Features are used to generate actionable insights.

# Features

- Data Collection: Real-time stock price data and historical price data fetched from Yahoo Finance using the `yfinance` library.

- Feature Engineering: Common technical indicators like Moving Averages (SMA), RSI, MACD, and ATR to capture market trends.

- Machine Learning Models: Uses Random Forest, XGBoost, and Logistic Regression to classify the next day's price movement (up or down).

- Model Evaluation: Evaluates the model using accuracy, F1-score, and ROC-AUC score.

- Prediction: Provides a prediction for the next day (Buy/Sell recommendation).

  # Requirements

  To run the project, you need to install the following Python libraries:
  ~~~bash
  pip install yfinance pandas numpy scikit-learn xgboost ta-lib imbalanced-learn matplotlib
  ~~~

  # Project Structure
~~~bash
market-trend-analysis/
├── data/
│   ├── AAPL.csv               # Example of historical data
├── notebooks/
│   ├── analysis.ipynb         # Jupyter notebook for data exploration and analysis
├── src/
│   ├── main.py                # Main script to train and evaluate models
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation (this file)
~~~

# Usage

1.Data Collection: The script fetches stock data for a specific symbol from Yahoo Finance.

2.Feature Engineering: Various technical indicators are calculated and used as features.

3.Model Training: Random Forest, XGBoost, and other classifiers are trained on the dataset.

4.Prediction: The trained model predicts whether the stock price will go up or down the next day.

# Steps to Run

1.Clone the repository or download the files.

2.Run the main script `(main.py)` to fetch data, train models, and get predictions.

3.Evaluate the model's performance using various metrics like accuracy, F1-score, and ROC-AUC.

# Explanation

## 1. Data Download:

The data for the stock symbol is fetched from Yahoo Finance using the `yfinance` library. You can modify the `ticker` and `dates` in the run_pipeline() function.

## 2. Feature Engineering:

- We calculate several technical indicators such as:

- SMA (Simple Moving Average)

- RSI (Relative Strength Index)

- MACD (Moving Average Convergence Divergence)

- ATR (Average True Range)

- Lagged Returns: To capture recent market momentum

  ## 3. Model Training:

  The model used is a Random Forest classifier. We train it on the features derived from the stock price and technical indicators.

  ## 4. Evaluation:

  We evaluate the model using:

- Accuracy

- Classification Report (Precision, Recall, F1-score)

- ROC-AUC score

  ## 5. SMOTE for Class Imbalance:

  Since stock price movements can often be imbalanced (e.g., more "up" days than "down"), we use SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes.

  # How to Run the Script

1.Clone the repository or download the `src/main.py` file.

2.Install required dependencies `(pip install -r requirements.txt)`.

3.Run the script by executing:
~~~bash
python src/main.py
~~~

# Results

- The script will output the accuracy of the model, along with the classification report and ROC-AUC score.

- It will print a prediction for whether the stock will go up or down the next day based on the trained model.

# Enhancements

- Model Improvement: You can experiment with other models like XGBoost or LSTM for time-series predictions.

- Sentiment Analysis Integration: Integrate sentiment scores from news or social media to refine predictions.

- Real-time Prediction: Modify the script to fetch real-time stock data and predict price movements in real-time.
