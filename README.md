# Stock Price Prediction using LSTM

This project predicts stock prices using a Long Short-Term Memory (LSTM) neural network. 
The model is trained on historical stock data obtained from Yahoo Finance and learns patterns 
in time-series data to forecast future prices.

## Features
- Data collection using `yfinance`
- Data preprocessing using scaling and sliding window technique
- LSTM neural network for time-series prediction
- Model evaluation using RMSE, MAE, MAPE, and Directional Accuracy
- Visualization of actual vs predicted stock prices

## Project Structure

LSTM Time Series Analysis
│
├── data_collection.py
├── preprocessing.py
├── model.py
├── train.py
├── prediction.py
├── visualize.py
├── stock_data.csv
├── requirements.txt
└── README.md

## Installation

Install required libraries:

pip install -r requirements.txt

## Run the Project

Run the prediction script:

python prediction.py

This will load the trained LSTM model and generate a graph comparing actual vs predicted stock prices.

## Output

The project generates a visualization showing the comparison between actual and predicted stock prices.

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- yfinance
