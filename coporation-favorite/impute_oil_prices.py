import pandas as pd
import lightgbm as lgb
import numpy as np
from datetime import date, timedelta
import matplotlib.pyplot as plt

#impute missing values in oil dataset
#for a missing date d_0
#its imputed price is the evarage of prices in d_(-2), d_(-1), d_(1), and d_(2)

def get_nearby_prices(data, d, timespan = 2, prev = True):
    if prev:
        operator = -1
    else:
        operator = 1
    oil_prices = []
    previous_date = d
    while len(oil_prices) < timespan:
        if previous_date in data.columns:
            oil_prices.append(float(data[previous_date]))
        previous_date = previous_date + (operator) * timedelta(days=1)
        
    return oil_prices


oils = pd.read_csv('oil.csv', parse_dates = ['date'])
oils = oils[oils['date']>= pd.datetime(2016,8,1)]
oils.columns = ['date', 'price']
data = pd.pivot_table(oils, values = 'price', columns = ['date'])
data.columns = data.columns.get_level_values(0)
#I need to impute missing values in oil price data
oil_prices = []
date_range = pd.date_range(date(2016,12,31), end = date(2017, 8, 31), frequency = 'D')
timespan = 2       
for d in date_range:
    if d in data.columns:
        oil_prices.append(float(data[d]))
    else:
        if len(oil_prices) < 2:
            previous_prices = get_nearby_prices(data, d, timespan = timespan, prev = True)
        else:
            previous_prices = oil_prices[-timespan:]
        later_prices = get_nearby_prices(data, d, timespan = timespan, prev = False)
        average = (sum(previous_prices + later_prices)) / (2 * timespan)
        oil_prices.append(average)
new_data = pd.DataFrame.from_dict({'date': pd.date_range(date(2016,12,31), end = date(2017, 8, 31), frequency = 'D'),
                              'oil_price': oil_prices})
new_data.to_csv('imputed_oils.csv')
