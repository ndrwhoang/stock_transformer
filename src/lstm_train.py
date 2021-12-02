import torch
torch.manual_seed(0)
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from operator import itemgetter
from pprint import pprint
from src.model.lstm import BaseLSTM


def load_training_data(window, future, price_type):
    print('Start loading data')
    with open('data\covid\stockprice_per_date.json', 'r') as f:
        data = json.load(f)
        
    data = sorted(data, key=itemgetter('formatted_time')) 
    data = [sample for sample in data if int(sample['formatted_time']) > 20200301]
    
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

def load_training_data_with_sentiment(window, future, price_type):
    print('Start loading data')
    with open('data\covid\\full.json', 'r') as f:
        data = json.load(f)
        
    data = sorted(data, key=itemgetter('formatted_time')) 
    data = [sample for sample in data if int(sample['formatted_time']) > 20200301]
    
    price_series = [float(sample[price_type].replace(',', '')) for sample in data]   
    reddit_pol_series, reddit_sub_series = [0], [0]
    headline_pol_series, headline_sub_series = [0], [0]
    for i in range(1, len(data)):
        reddit_pol_series.append(data[i-1].get('reddit_polarity', 0))
        reddit_sub_series.append(data[i-1].get('reddit_subjectivity', 0))
        headline_pol_series.append(data[i-1].get('headline_polarity', 0))
        headline_sub_series.append(data[i-1].get('headline_subjectivity', 0))
    
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
        # target = closing_series[i+window+future:i+window+future+1]
        target_seq = torch.tensor(price_series[i+1:i+window+1], dtype=torch.float)
        training_data.append((input_seq, target_seq))
    
    print(f'Finished loading data, n_sample : {len(training_data)}')
    return {'training_data': training_data}

def do_train(model, training_data, n_epoch, lr, device):
    # training_data = data['training_data']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()
    
    for i in range(n_epoch):
        mae = 0
        training_data = training_data.shuffle()
        for i_s, sample in enumerate(training_data):
            sample = tuple(item.to(device) for item in sample)  
            (input_seq, target) = sample        
            model.hidden_cell = (torch.zeros(model.n_layer, 1, model.hidden_dim, device='cuda:0'), 
                                 torch.zeros(model.n_layer, 1, model.hidden_dim, device='cuda:0'))  
            out = model(sample)
            
            # if i_s > 50 and i_s < 55:
            #     print(out[0].item(), target[0].item())
            
            if i == n_epoch-1:
                print(input_seq[:,0].tolist(), out.tolist(), target.tolist())
            
            loss = loss_fn(out, target.unsqueeze(1))
            with torch.no_grad():
                mae += torch.mean(torch.abs(out - target))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'================ Epoch {i} MAE: {loss}')

def base_lstm_train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BaseLSTM(1, 800)
    data = load_training_data(3, 0, 'Close*')
    # for sample in data['training_data']:
    #     print(sample)
    model.to(device)
    do_train(model, data['training_data'], 50, 0.01, device)

def sentiment_lstm_train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BaseLSTM(5, 800)
    data = load_training_data_with_sentiment(2, 0, 'Close*')
    # for sample in data['training_data']:
    #     print(sample[0].size())
    #     print(sample[0][:, 0], sample[1])
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
    base_lstm_train()
    