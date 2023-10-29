#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Fetching the data from yahoo finance
from pandas_datareader.data import DataReader  # To read stock data
import yfinance as yf  # Yahoo Finance API
from pandas_datareader import data as pdr

yf.pdr_override()
from datetime import datetime


# In[2]:


#List of top Indian stocks for analysis
indian_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS']

#Defining the time period for data retrieval
end_date = datetime.now()
start_date = datetime(end_date.year - 1, end_date.month, end_date.day)

#Creating a dictionary to store stock data
stock_data_dict = {}

#Retrieving stock data for the specified Indian stocks
for stock_symbol in indian_stocks:
    stock_data_dict[stock_symbol] = yf.download(stock_symbol, start_date, end_date)

#Defining company names for the stocks
company_names = ["Reliance Industries", "Tata Consultancy Services", "HDFC Bank", "Infosys"]

#Assigning company names to the stock data
for stock_symbol, company_name in zip(stock_data_dict.keys(), company_names):
    stock_data_dict[stock_symbol]["Company Name"] = company_name

indian_stock_df = pd.concat(stock_data_dict.values(), axis=0)
indian_stock_df.tail(10)


# In[3]:


indian_stock_df.info()


# In[4]:


indian_stock_df.describe()


# In[5]:


indian_stock_df[indian_stock_df["Company Name"] == "Tata Consultancy Services"].describe()


# In[6]:


#Let's see a historical view of the closing price
plt.figure(figsize=(15, 10))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, stock_symbol in enumerate(indian_stocks, 1):
    plt.subplot(2, 2, i)
    stock_data_dict[stock_symbol]['Adj Close'].plot()
    plt.ylabel('Adj Close')
    plt.xlabel(None)
    plt.title(f"Closing Price of {company_names[i - 1]}")
    
plt.tight_layout()


# In[7]:


#subplots for trading volume
plt.figure(figsize=(15, 10))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, stock_symbol in enumerate(indian_stocks, 1):
    plt.subplot(2, 2, i)
    stock_data_dict[stock_symbol]['Volume'].plot()
    plt.ylabel('Volume')
    plt.xlabel(None)
    plt.title(f"Trading Volume of {company_names[i - 1]}")
    
plt.tight_layout()


# In[8]:


#subplots for moving averages (10 days, 50 days, and 100 days)
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(10)
fig.set_figwidth(15)

ma_days = [10, 50, 100]

for i, stock_symbol in enumerate(indian_stocks, 1):
    for ma_day in ma_days:
        column_name = f"MA for {ma_day} days"
        stock_data_dict[stock_symbol][column_name] = stock_data_dict[stock_symbol]['Close'].rolling(ma_day).mean()
    
    stock_data_dict[stock_symbol][[f"MA for {ma_day} days" for ma_day in ma_days]].plot(ax=axes[(i - 1) // 2, (i - 1) % 2])
    axes[(i - 1) // 2, (i - 1) % 2].set_title(company_names[i - 1])

fig.tight_layout()
plt.show()


# In[9]:


#daily returns for Indian stocks
for stock_symbol in indian_stocks:
    stock_data_dict[stock_symbol]['Daily Return'] = stock_data_dict[stock_symbol]['Adj Close'].pct_change()

#subplots for daily returns
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(10)
fig.set_figwidth(15)

for i, stock_symbol in enumerate(indian_stocks, 1):
    stock_data_dict[stock_symbol]['Daily Return'].plot(ax=axes[(i - 1) // 2, (i - 1) % 2], legend=True, linestyle='--', marker='o')
    axes[(i - 1) // 2, (i - 1) % 2].set_title(company_names[i - 1])

fig.tight_layout()
plt.show()


# In[10]:


plt.figure(figsize=(12, 9))

for i, stock_symbol in enumerate(indian_stocks, 1):
    plt.subplot(2, 2, i)
    stock_data_dict[stock_symbol]['Daily Return'].hist(bins=50, alpha=0.5, label='Histogram', color='blue')
    stock_data_dict[stock_symbol]['Daily Return'].plot(kind='kde', label='KDE', color='red')
    plt.xlabel('Daily Return')
    plt.ylabel('Counts')
    plt.title(f'{company_names[i - 1]}')
    plt.legend()

plt.tight_layout()
plt.show()


# In[11]:


#DataFrame with the 'Close' columns for each stock and then calculating 
#correlation matrix
closing_prices_df = pd.DataFrame()

for stock_symbol in indian_stocks:
    closing_prices_df[stock_symbol] = stock_data_dict[stock_symbol]['Close']

correlation_matrix = closing_prices_df.corr()
print(correlation_matrix)


# In[12]:


tcs_df = indian_stock_df['Close']  

#scatter plot to compare TCS's daily return to itself
sns.jointplot(x=tcs_df, y=tcs_df, kind='scatter', color='seagreen')


# In[13]:


#daily returns for TCS and Reliance
tcs_df = indian_stock_df[indian_stock_df['Company Name'] == 'Tata Consultancy Services']['Adj Close'].pct_change()
reliance_df = indian_stock_df[indian_stock_df['Company Name'] == 'Reliance Industries']['Adj Close'].pct_change()

#jointplot to compare the daily returns of TCS and Reliance
sns.jointplot(x=tcs_df, y=reliance_df, kind='scatter', color='seagreen')
plt.show()


# In[14]:


print(indian_stock_df.columns)


# In[15]:


#daily returns for Indian stocks
for stock_symbol in indian_stocks:
    stock_data_dict[stock_symbol]['Daily Return'] = stock_data_dict[stock_symbol]['Adj Close'].pct_change()

tech_rets = pd.concat([stock_data_dict[stock_symbol]['Daily Return'] for stock_symbol in indian_stocks], axis=1)
tech_rets.columns = indian_stocks

#pairplot to visualize correlations
sns.pairplot(tech_rets, kind='reg')
plt.show()


# In[16]:


pair_grid = sns.PairGrid(indian_stock_df.dropna())
pair_grid.map_upper(plt.scatter, color='purple')
pair_grid.map_lower(sns.kdeplot, cmap='coolwarm')
pair_grid.map_diag(plt.hist, bins=30)
pair_grid.fig.tight_layout()
plt.show()


# In[17]:


return_fig = sns.PairGrid(indian_stock_df.dropna())
return_fig.map_upper(plt.scatter, color='purple')
return_fig.map_lower(sns.kdeplot, cmap='coolwarm')
return_fig.map_diag(plt.hist, bins=30)
return_fig.fig.tight_layout()
plt.show()


# In[18]:


#daily returns for each stock
for stock_symbol in indian_stocks:
    stock_data_dict[stock_symbol]['Daily Return'] = stock_data_dict[stock_symbol]['Adj Close'].pct_change()

indian_stock_df = pd.concat(stock_data_dict.values(), axis=0)

print(indian_stock_df.head())


# In[19]:


#correlation matrix for daily returns after dropping missing values
correlation_matrix_daily_returns = indian_stock_df.dropna().pivot_table(values='Daily Return', index='Date', columns='Company Name', aggfunc='mean').corr()

plt.figure(figsize=(12, 10))

#Correlation of stock return
plt.subplot(2, 2, 1)
sns.heatmap(correlation_matrix_daily_returns, annot=True, cmap='summer')
plt.title('Correlation of Daily Returns')

plt.tight_layout()
plt.show()


# In[20]:


import yfinance as yf
from datetime import datetime

#stock quote for TCS from January 1, 2012, to the current date
df = yf.download('TCS', start='2012-01-01', end=datetime.now())

print(df)


# In[21]:


plt.figure(figsize=(16, 6))
plt.title('Closing Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Closing Price', fontsize=18)
plt.grid(True)
plt.show()


# In[22]:


import numpy as np

#new DataFrame with only the 'Close' column
data = df.filter(['Close'])
dataset = data.values
training_data_len = int(np.ceil(len(dataset) * 0.95))

print("Training data length:", training_data_len)


# In[23]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
scaled_data


# In[24]:


#training data set
train_data = scaled_data[0:int(training_data_len), :]

#number of time steps (sequence length)
time_steps = 60

#lists to store x_train and y_train
x_train = []
y_train = []

#training data sequences
for i in range(time_steps, len(train_data)):
    x_train.append(train_data[i - time_steps:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)


# In[25]:


pip install tensorflow


# In[26]:


from keras.models import Sequential
from keras.layers import Dense, LSTM

#LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size=1, epochs=40)


# In[27]:


test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))



# In[28]:


predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[29]:


rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
rmse


# In[39]:


#new DataFrame for predictions
prediction_df = pd.DataFrame(predictions, columns=["Predictions"])

#new index for the 'valid' DataFrame based on the test data
valid_index = data.index[training_data_len : training_data_len + len(predictions)]
valid = data.iloc[training_data_len : training_data_len + len(predictions)].copy()
valid.index = valid_index


if len(predictions) > len(valid):
    predictions = predictions[:len(valid)]
else:
    valid = valid.iloc[:len(predictions)]

valid['Predictions'] = predictions

#Plotting the data
plt.figure(figsize=(16, 6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(data['Close'][:training_data_len], label='Train')
plt.plot(valid.index, valid[['Close', 'Predictions']])
plt.legend(loc='lower right')
plt.show()


# In[40]:


valid


# In[ ]:




