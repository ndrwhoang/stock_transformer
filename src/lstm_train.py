import torch
import torch.nn as nn
import numpy as np
import json
from operator import itemgetter
from pprint import pprint
from src.model.lstm import BaseLSTM
from torch.optim import Adam

def load_training_data(window, future, price_type):
    print('Start loading data')
    with open('data\covid\stockprice_per_date.json', 'r') as f:
        data = json.load(f)
        
    data = sorted(data, key=itemgetter('formatted_time')) 
    data = [sample for sample in data if int(sample['formatted_time']) > 20200301]
    
    # closing_series = [{sample['formatted_time'] : sample['Close*']} for sample in data]
    # pprint(closing_series)
    closing_series = [float(sample[price_type].replace(',', '')) for sample in data]
    mean = sum(closing_series)/len(closing_series)
    sd = np.std(closing_series)
    # closing_series = [(value - mean)/sd for value in closing_series]
    
    training_data = []
    # window = 2
    for i in range(len(closing_series)-window-future):
        input_seq = closing_series[i: i+window]
        target = closing_series[i+window+future:i+window+future+1]
        training_data.append((torch.tensor(input_seq, dtype=torch.float), torch.tensor(target, dtype=torch.float)))
    
    print(f'Finished loading data, n_sample : {len(training_data)}')
    # for sample in training_data[:10]:
    #     print(sample[0])
    #     print(sample[1])
    
    return {'training_data': training_data, 'mean': mean, 'sd': sd}
    
def do_train(model, training_data, n_epoch, lr):
    # training_data = data['training_data']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()
    
    for i in range(n_epoch):
        total_loss = 0
        for i_s, sample in enumerate(training_data):
            (input_seq, target) = sample            
            out = model(sample)
            
            # if i_s > 50 and i_s < 55:
            #     print(out[0].item(), target[0].item())
            
            if i == 49:
                print(out[0].item(), target[0].item())
            
            loss = torch.sqrt(loss_fn(out, target))
            with torch.no_grad():
                total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'================ Total epoch {i} loss is: {total_loss/len(training_data)}')
        
def base_lstm_train():
    model = BaseLSTM(1, 1000)
    data = load_training_data(2, 0, 'Close*')
    do_train(model, data['training_data'], 15, 0.01)

if __name__ == '__main__':
    print('hello world')
    