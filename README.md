# Stock Price Prediction Using Machine Learning

**Aditya Singh Yadav**, **Garvit Kochar**, **Himeksh Malhotra**, **Lakshay Kumar**  
2022039, 2022185, 2022218, 2022266  
Indraprastha Institute of Information Technology Delhi  

## Motivation

Stock market prediction has always intrigued financial analysts and enthusiasts. With the volatility of stock prices, accurate predictions can be valuable in making informed investment decisions. The goal of this project is to predict stock price movements for companies like **GameStop (GME)**, **Google (GOOGL)**, and **Nvidia (NVDA)**, using machine learning techniques. By analyzing historical stock data, the project aims to develop a predictive model that can assist in understanding future stock price behavior, ultimately helping to make better investment decisions.

## Introduction

Predicting stock prices is a highly challenging task due to the fluctuating nature of the market. This project focuses on using machine learning models to predict stock price movements based on historical data. We aim to utilize various technical indicators such as **Simple Moving Average (SMA)**, **Exponential Moving Average (EMA)**, **Relative Strength Index (RSI)**, **Bollinger Bands (BB)**, and **Moving Average Convergence Divergence (MACD)** to build predictive models for **GameStop**, **Google**, and **Nvidia**. We will also explore the relationships between historical prices and indicators, visualizing these patterns using UMAP and violin plots.

## Literature Survey

Various studies have explored different machine learning algorithms for stock price prediction. For example, **D. Bhuriya (2017)** used linear regression to predict stock prices, while **J. Hota et al. (2020)** employed models like **Random Forest**, **Decision Tree**, **SVR**, and **ANN**. The study concluded that **Random Forest** outperforms other models, achieving an impressive **MAPE of 0.36**. Another study emphasized the advantages of **Long Short-Term Memory (LSTM)** networks, which showed better performance than traditional algorithms for time-series data.

## Dataset

The dataset used in this project includes stock data from **Google**, **Nvidia**, and **GameStop** covering a period from **2004 to 2024**. This data is comprehensive enough to allow the training of robust machine learning models. 

### Attributes
- Date
- Open
- High
- Low
- Close
- Adjusted Close
- Volume

### Data Preprocessing
1. **Data Normalization**: We applied **Min-Max Scaling** to normalize the features to a range of 0 to 1.
2. **Outlier Detection**: Extreme values were identified and handled using **IQR Analysis**.
3. **Data Splitting**: The dataset was split into an **80% training** set and **20% testing** set.

### Data Visualization
To gain better insights from the data, we performed various visualizations:
- **UMAP**: Visualized relationships and clusters over time.
- **Violin Plots**: Showed the distribution of stock prices over time.
- **Pair Plots**: Analyzed relationships between multiple variables for each stock.

### Technical Indicators
We computed several technical indicators to enhance stock price predictions:
- **Simple Moving Average (SMA)**: 20-day SMA to observe long-term trends.
- **Exponential Moving Average (EMA)**: 20-day EMA for capturing short-term trends.
- **Relative Strength Index (RSI)**: Measures whether the stock is overbought or oversold.
- **Bollinger Bands (BB)**: Helps identify volatility and potential price reversals.
- **MACD**: An indicator to analyze the momentum and direction of the stock price.

## Methodology

The methodology of the project involves the following steps:
1. **Data Collection**: Gather raw stock data from public sources.
2. **Preprocessing**: Clean and standardize the data.
3. **Data Splitting**: Split the data into training and testing sets while keeping the time-dependent nature intact.
4. **Model Training and Testing**: We train four models: **SVR**, **Random Forest**, **ARIMA**, and **SARIMA**, and test their performance.
5. **Performance Evaluation**: We evaluate the models based on metrics like **R-squared** and **Root Mean Square Error (RMSE)**.

### Models Used:
- **Support Vector Regression (SVR)**: Captures complex patterns using kernel-based methods.
- **Random Forest Regression**: An ensemble learning technique using multiple decision trees.
- **ARIMA**: A classical time-series forecasting model.
- **SARIMA**: An extension of ARIMA, accounting for seasonal components in data.

## Analysis

### Performance Evaluation
The performance of each model was evaluated based on **Mean Squared Error (MSE)**, **Mean Absolute Error (MAE)**, and **R-squared** scores:

#### 1. **RandomForest Regressor**:
- **Google**: MSE: 1.9709, MAE: 1.0835, R²: 0.9924
- **Nvidia**: MSE: 622.0219, MAE: 12.9418, R²: 0.1079
- **GameStop**: MSE: 3.9865, MAE: 1.5034, R²: 0.9309

#### 2. **ARIMA**:
- **Google**: MSE: 6.4636, MAE: 1.9009
- **Nvidia**: MSE: 2.0248, MAE: 0.9325
- **GameStop**: MSE: 2.4292, MAE: 1.0036

#### 3. **SARIMA**:
The performance metrics for **SARIMA** were identical to **ARIMA** as it is an extension of ARIMA for seasonal data.

## Conclusion

The project successfully demonstrated the use of machine learning techniques for predicting stock prices. The **Random Forest** model performed exceptionally well on the Google dataset, while the **ARIMA** and **SARIMA** models showed promising results on the Nvidia and GameStop datasets. Overall, this project provides a strong foundation for using machine learning to analyze and predict stock price movements based on historical data and technical indicators.

## Setup and Installation

To run the project, you will need the following dependencies:
- Python 3.x
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **Statsmodels**

You can install the required libraries using the following command:
```bash
pip install -r requirements.txt
```

### How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock-price-prediction.git
   ```
2. Navigate to the project directory and run the **main.py** file:
   ```bash
   python main.py
   ```

## Contact
For further inquiries, feel free to reach out to the team members:
- **Aditya Singh Yadav**: aditya22039@iiitd.ac.in
- **Garvit Kochar**: garvit22185@iiitd.ac.in
- **Himeksh Malhotra**: himeksh22218@iiitd.ac.in
- **Lakshay Kumar**: lakshay22266@iiitd.ac.in

