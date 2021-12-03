from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

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

def load_df_ts(path, price_type):
    stock_path = path
    df = pd.read_csv(stock_path, thousands=',')
    data = df.to_dict('records')
    
    time_series = []
    
    for i_s, sample in enumerate(data):
        date = sample['Date'].split('-')
        formatted_date = ['20' + date[2], 
                          months[date[1]], 
                          date[0] if len(date[0]) == 2 else '0' + date[0]]
        formatted_date = int(''.join(formatted_date))
        time_series.append({'formatted_time': formatted_date, 
                            'time': sample['Date'], 
                            'price': float(sample[price_type])})
    
    df_ts = pd.DataFrame(time_series)
    df_ts = df_ts[df_ts['formatted_time'] > 20200301]
    df_ts.sort_values(by=['formatted_time'], inplace=True)
    # df_ts.to_csv('test.csv', index=False)
    
    return df_ts
    


def arima_fit(p, d, q, df_ts):
    order = (p, d, q)
    train_data = df_ts[['time', 'price']]
    train_data.set_index('time', inplace=True)
    arima_kousei = ARIMA(train_data['price'].values, order=order)
    model = arima_kousei.fit()
    # print(model.summary())
    resid = [abs(round(err, 3)) for err in model.resid]
    ape = [abs(err) / abs(gold) for gold, err in zip(train_data['price'].values, resid)]
    mape = sum(ape) / len(ape)
    # print(resid)
    # mae = sum(resid[1:]) / len(resid[1:])
    # print(mae)
    
    return mape

def arima_grid_search(df_ts):
    best_loss = 100000
    # best_param = (0, 0, 0)
    ps = range(7, 14, 2)
    ds = [1]
    qs = range(7, 14, 2)
    for p in ps:
        for d in ds:
            for q in qs:
                try:
                    mae = arima_fit(p, d, q, df_ts)
                    if mae < best_loss:
                        best_loss = mae
                        best_param = (p, d, q)
                        # print(best_loss)
                        # print(best_param)
                except:
                    pass
                
    print(best_param, best_loss)
    return (best_param, best_loss)

if __name__ == '__main__':
    print('hello world')
    
    
    # results = []
    # for s_type in tqdm(['Open', 'High', 'Low', 'Adj Close**', 'Close*']):
    #     # Data loading
    #     df_ts = load_df_ts('data\\financris\stock_2008.csv', s_type)
        
    #     # ARIMA Modeling
    #     out = arima_grid_search(df_ts)
    #     results.append([s_type, out])
    
    # print(results)
    
    # df_ts = load_df_ts('data\\financris\stock_2008.csv', 'Open')
    # mape = arima_fit(11, 1, 7, df_ts)
    # print(mape)
    