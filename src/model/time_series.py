from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

months = {
    'January': '01',
    'Jan': '01',
    'February': '02',
    'Feb': '02',
    'March': '03',
    'Mar': '03',
    'April': '04',
    'Apr': '04',
    'May': '05',
    'June': '06',
    'Jun': '06',
    'July': '07', 
    'Jul': '07',
    'August': '08',
    'Aug': '08',
    'September': '09',
    'Sept': '09',
    'Sep': '09',
    'October': '10',
    'Oct': '10',
    'November': '11',
    'Nov': '11',
    'December': '12',
    'Dec': '12'  
}

if __name__ == '__main__':
    # Data loading
    stock_path = 'data\covid\stock.csv'
    df = pd.read_csv(stock_path)
    data = df.to_dict('records')
    
    price = 'Close*'
    time_series = []
    
    for i_s, sample in enumerate(data):
        date = sample['Date'].split('-')
        formatted_date = ['20' + date[2], 
                          months[date[1]], 
                          date[0] if len(date[0]) == 2 else '0' + date[0]]
        formatted_date = int(''.join(formatted_date))
        time_series.append({'formatted_time': formatted_date, 
                            'time': sample['Date'], 
                            'price': float(sample[price].replace(',', ''))})
    
    df_ts = pd.DataFrame(time_series)
    df_ts = df_ts[df_ts['formatted_time'] > 20200301]
    df_ts.sort_values(by=['formatted_time'], inplace=True)
    df_ts.to_csv('test.csv', index=False)
    
    # ARIMA Modeling
    order = (2, 1, 2)
    train_data = df_ts[['time', 'price']]
    train_data.set_index('time', inplace=True)
    arima_kousei = ARIMA(train_data['price'].values, order=order)
    model = arima_kousei.fit()
    print(model.summary())
    print(pd.DataFrame(model.resid).describe())
    
    