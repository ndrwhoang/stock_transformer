import torch
torch.manual_seed(0)
import torchtext
from torchtext.data import get_tokenizer
import random
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from operator import itemgetter
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from src.model.lstm import BaseLSTM, OFLinear
from src.dataset.preprocessing import *


def load_data(window, future, price_type):
    print('Start loading data')
    with open('data\\financris\\full_2008.json', 'r') as f:
        data = json.load(f)
        
    data = sorted(data, key=itemgetter('formatted_time')) 
    # data = [sample for sample in data if int(sample['formatted_time']) > 20200301]
    
    # closing_series = [{sample['formatted_time'] : sample['Close*']} for sample in data]
    # pprint(closing_series)
    closing_series = [float(sample[price_type].replace(',', '')) for sample in data]
    # mean = sum(closing_series)/len(closing_series)
    # sd = np.std(closing_series)
    # closing_series = [(value - mean)/sd for value in closing_series]
    
    training_data = []
    # window = 2
    for i in range(len(closing_series)-window-future):
        input_seq = closing_series[i: i+window]
        # target = closing_series[i+window+future:i+window+future+1]
        target = closing_series[i+1:i+window+1]
        training_data.append((torch.tensor(input_seq, dtype=torch.float), torch.tensor(target, dtype=torch.float)))
    
    print(f'Finished loading data, n_sample : {len(training_data)}')
    # for sample in training_data[:10]:
    #     print(sample[0])
    #     print(sample[1])
    
    return {'training_data': training_data}

def load_data_with_sentiment(window, future, price_type):
    print('Start loading data')
    with open('data\covid\\full.json', 'r') as f:
        data = json.load(f)
    
        
    data = sorted(data, key=itemgetter('formatted_time')) 
    # data = [sample for sample in data if int(sample['formatted_time']) > 20080915]
    
    raw_time_series = [float(sample[price_type].replace(',', '')) for sample in data]   
    split_index = int(0.7*len(raw_time_series))
    mean_price = sum(raw_time_series[:split_index]) / len(raw_time_series[:split_index])
    price_series = [i - mean_price for i in raw_time_series]
    # detrend_price_series = []
    # for i in range(1, len(raw_time_series)):
    #     detrend_price_series.append(raw_time_series[i] - raw_time_series[i-1])
    # price_series = detrend_price_series
    # print(price_series[:split_index])
    reddit_pol_series, reddit_sub_series = [], []
    headline_pol_series, headline_sub_series = [], []
    # for i in range(0, len(data)):
    #     reddit_pol_series.append(data[i].get('reddit_polarity', 0))
    #     reddit_sub_series.append(data[i].get('reddit_subjectivity', 0))
    #     headline_pol_series.append(data[i].get('headline_polarity', 0))
    #     headline_sub_series.append(data[i].get('headline_subjectivity', 0))
        
    for i in range(1, len(data)):
        reddit_pol_series.append(0)
        reddit_sub_series.append(0)
        headline_pol_series.append(0)
        headline_sub_series.append(0)
    
    # for price, rpol, rsub, hpol, hsub in zip(price_series, reddit_pol_series, reddit_sub_series, headline_pol_series, headline_sub_series):
    #     print(price, rpol, rsub, hpol, hsub)        
    
    training_data = []
    for i in range(len(price_series)-window-future):
        price_seq = torch.tensor(price_series[i: i+window], dtype=torch.float)
        rpol_seq = torch.tensor(reddit_pol_series[i: i+window], dtype=torch.float)
        rsub_seq = torch.tensor(reddit_sub_series[i: i+window], dtype=torch.float)
        hpol_seq = torch.tensor(headline_pol_series[i: i+window], dtype=torch.float)
        hsub_seq = torch.tensor(headline_sub_series[i: i+window], dtype=torch.float)
        input_seq = torch.stack([price_seq, rpol_seq, rsub_seq, hpol_seq, hsub_seq], dim=1)
        target_seq = torch.tensor(price_series[i+window:i+window+1], dtype=torch.float)
        training_data.append((input_seq, target_seq))
    
    split_index = int(0.7*len(training_data))
    train = training_data[:split_index]
    test = training_data[split_index:]
    raw_test = raw_time_series[split_index+1:]
    
    print(f'Finished loading data, n_sample : {len(training_data)}')
    return {'training_data': train, 'testing_data': test, 'raw_testing_data': raw_test}

def do_train(model, training_data, n_epoch, lr, device):
    # training_data = data['training_data']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()
    n_step = len(training_data)
    maes = []
    
    for i in range(n_epoch):
        mae = 0
        random.shuffle(training_data)
        pbar = tqdm(enumerate(training_data))
        for i_s, sample in pbar:
            sample = tuple(item.to(device) for item in sample)  
            (input_seq, target) = sample         
            out = model(sample)
            
            # print('out', out.size())
            # print('target', target.size())
            
            loss = loss_fn(out, target)
            with torch.no_grad():
                mae += torch.abs(out - target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_description(f'Epoch {i} - {i_s} / {n_step} - loss: {loss}')
        
        print(f'================ Epoch {i} MAE: {mae.item()/n_step}')
        
    return model

def do_valid(model, validation_data, device):
    mae = 0
    loss_fn = nn.L1Loss()
    all_preds = []
    all_true = []
    random.shuffle(validation_data)
    pbar = tqdm(enumerate(validation_data))
    n_step = len(validation_data)
    for i_s, sample in pbar:
        sample = tuple(item.to(device) for item in sample)  
        (input_seq, target) = sample        
        out = model(sample)
        all_preds.append(out.item())
        all_true.append(target.item())
        
        with torch.no_grad():
            loss = loss_fn(out, target)
            mae += torch.abs(out - target)
    
    print(f'================ Validation MAE: {mae.item()/n_step}')

def base_lstm_train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BaseLSTM(1, 800)
    data = load_data(20, 0, 'Close*')
    # for sample in data['training_data']:
    #     print(sample)
    model.to(device)
    do_train(model, data['training_data'], 50, 0.01, device)
    
    return model

def sentiment_lstm_train():
    device = torch.device("cuda:0")
    
    model_input = 5
    model_hidden = 20
    window = 25
    skip_forecast = 0
    n_epoch = 1
    lr = 0.01
    
    model = BaseLSTM(model_input, model_hidden)
    data = load_data_with_sentiment(window, skip_forecast, 'Close*')
    for i_s, sample in enumerate(data['training_data']):
        if i_s == 3: break
        print(sample[0].size())
        print(sample[0][:, 0], sample[1])
        print(sample)
    model.to(device)
    model = do_train(model, data['training_data'], n_epoch, lr, device)
    # do_valid(model, data['testing_data'], device)

def linear_train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = OFLinear(5, 200)
    data = load_training_data_with_sentiment(5, 0, 'Close*')
    model.to(device)
    do_train(model, data['training_data'], 50, 0.01, device)
 
def _merge_training_data_with_sentiment():
    print('Start loading data')
    with open('data\covid\stock_with_covid_sentiment.json', 'r') as f:
        data = json.load(f)
    data = sorted(data, key=itemgetter('formatted_time')) 
    data = [sample for sample in data if int(sample['formatted_time']) > 20200301]
    
    headlines = pd.read_csv('data\covid\headline_sentiment.csv', thousands=',')
    headlines = headlines.to_dict('records')
    
    data_out = {}
    for i_c, comment in enumerate(headlines):
        polarity = comment['polarity']
        subjectivity = comment['subjectivity']
        if polarity * subjectivity != 0:
            if comment['time'] not in data_out:
                data_out[int(comment['time'])] = {
                    'polarity': [polarity],
                    'subjectivity': [subjectivity]
                }
            else:
                data_out[int(comment['time'])]['polarity'].append(polarity)
                data_out[int(comment['time'])]['subjectivity'].append(subjectivity)
    
    for i_s, sample in enumerate(data):
        polarity = data_out[int(sample['formatted_time'])]['polarity']
        subjectivity = data_out[int(sample['formatted_time'])]['subjectivity']
        dm_polarity = sum(polarity) / len(polarity)
        dm_subjectivity = sum(subjectivity) / len(subjectivity)
        sample['headline_polarity'] = dm_polarity
        sample['headline_subjectivity'] = dm_subjectivity

    
    with open('data\covid\\full.json', 'w') as f:
        json.dump(data, f)



if __name__ == '__main__':
    print('hello world')
    sentiment_lstm_train()
    