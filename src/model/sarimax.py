import statsmodels.api as sm
import numpy as np
import pandas as pd
import json
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm
from operator import itemgetter
import warnings
warnings.filterwarnings("ignore")


def json_to_dataframe(path, price_type):
    with open(path, 'r') as f:
        data = json.load(f)
    
    data = sorted(data, key=itemgetter('formatted_time')) 
    # data = [sample for sample in data if int(sample['formatted_time']) > 20200301]
    
    price_series = [float(sample[price_type].replace(',', '')) for sample in data]
    time_series = [sample['formatted_time'] for sample in data]
    reddit_pol_series, reddit_sub_series = [], []
    headline_pol_series, headline_sub_series = [], []
    for i in range(0, len(data)):
        reddit_pol_series.append(data[i].get('reddit_polarity', 0))
        reddit_sub_series.append(data[i].get('reddit_subjectivity', 0))
        headline_pol_series.append(data[i].get('headline_polarity', 0))
        headline_sub_series.append(data[i].get('headline_subjectivity', 0))
    
    # print(len(price_series))
    # print(len(reddit_pol_series))
    assert len(price_series) == len(reddit_pol_series)
    assert len(price_series) == len(reddit_sub_series)
    assert len(price_series) == len(headline_pol_series)
    assert len(price_series) == len(headline_sub_series)
    
    to_df = list(zip(time_series, price_series, reddit_pol_series, reddit_sub_series, headline_pol_series, headline_sub_series))
    df = pd.DataFrame(to_df, columns = ['time', 'price', 'rpol', 'rsub', 'hpol', 'hsub'])
    df.set_index('time', inplace=True)
    
    # df_ts = pd.DataFrame(time_series)
    # # df_ts = df_ts[df_ts['formatted_time'] > 20200301]
    # df_ts.sort_values(by=['formatted_time'], inplace=True)
    return df



if __name__ == '__main__':
    print('hello world')
    # df = json_to_dataframe('data\\financris\\full_2008.json', 'Close*')
    df = json_to_dataframe('data\covid\\full_2020.json', 'Close*')
    split_index = int(0.7*len(df))
    df_train = df.iloc[:split_index, :]
    df_test = df.iloc[split_index:, :]
    sarimax = SARIMAX(endog = df_train['price'], 
                      exog = df_train[['rpol', 'rsub', 'hpol', 'hsub']],
                      order = (2, 1, 2))
    model = sarimax.fit()
    print(model.summary())
    pred = model.get_prediction(start=0, end=len(df_test),
                                exog = df_test[['rpol', 'rsub', 'hpol', 'hsub']]
                                )
    forecast = list(pred.predicted_mean)[1:]
    golds = df['price'].tolist()[1:]
    ae = 0
    for i, j in zip(forecast, golds):
        ae += abs(i-j)
    mae = ae / len(golds)
    print(mae)
    # resid = [abs(round(err, 4)) for err in model.resid]
    # mae = sum(resid[1:]) / len(resid[1:])
    # print(mae)