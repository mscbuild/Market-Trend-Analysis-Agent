import yfinance as yf
import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# Step 1: Download stock data (AAPL example)
def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Step 2: Feature Engineering (Adding Technical Indicators)
def add_technical_indicators(data):
    # Moving Averages
    data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
    data['SMA_200'] = talib.SMA(data['Close'], timeperiod=200)
    
    # RSI (Relative Strength Index)
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    
    # MACD (Moving Average Convergence Divergence)
    data['MACD'], _, _ = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    
    # Average True Range (ATR)
    data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
    
    # Lagged returns
    data['return'] = data['Close'].pct_change()
    data['return_lag1'] = data['return'].shift(1)
    data['return_lag2'] = data['return'].shift(2)
    
    # Drop NaN values
    data.dropna(inplace=True)
    return data

# Step 3: Model Training and Prediction
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 4: Prediction
def make_prediction(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

# Step 5: Evaluate Model
def evaluate_model(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# Step 6: Full Pipeline
def run_pipeline(ticker='AAPL', start_date='2010-01-01', end_date='2023-01-01'):
    # Download data
    data = download_data(ticker, start_date, end_date)
    
    # Add technical indicators
    data = add_technical_indicators(data)
    
    # Feature selection (Use indicators as features)
    features = ['SMA_50', 'SMA_200', 'RSI', 'MACD', 'ATR', 'return_lag1', 'return_lag2']
    X = data[features]
    
    # Target: Predict next day's price direction (up=1, down=0)
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    y = data['Target']
    
    # Scaling features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Handle class imbalance (using SMOTE)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Train model
    model = train_model(X_resampled, y_resampled)
    
    # Make predictions
    y_pred = make_prediction(model, X_test)
    
    # Evaluate the model
    evaluate_model(y_test, y_pred)

if __name__ == "__main__":
    run_pipeline()
