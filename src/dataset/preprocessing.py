import json
import time
import datetime
from datetime import timedelta
import os
import pandas as pd
from pprint import pprint
from operator import itemgetter

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

def process_headlines():
    df = pd.read_csv(os.path.join(*'data\mergeddata.csv'.split('\\')),
                     header=0,
                    #  nrows=100
                     )
    
    data = df.to_dict('records')
    for sample in data:
        formatted_time = [0]*3
        time = sample['time']
        
        if ',' in time:
            time = time.split(',')[-1].split(' ')
            time = [i for i in time if i != '']
            assert len(time) == 3
            formatted_time[0] = time[2]
            formatted_time[1] = months[time[1]]
            formatted_time[2] = time[0] if len(time[0]) == 2 else '0' + time[0]
        elif '-' in time:
            time = time.split('-')
            if len(time) != 3:
                continue
            formatted_time[0] = '20' + time[2]
            formatted_time[1] = months[time[1]]
            formatted_time[2] = time[0] if len(time[0]) == 2 else '0' + time[0]
        else:
            time = time.split(' ')
            assert len(time) == 3
            formatted_time[0] = time[2]
            formatted_time[1] = months[time[0]]
            formatted_time[2] = time[1] if len(time[1]) == 2 else '0' + time[1]
        
        # date_time = datetime.strptime(' '.join(formatted_time), '%m %d %Y')
        # sample['formatted_time'] = date_time
        sample['formatted_time'] = ''.join(formatted_time)
        
    with open('data\headline_per_article.json', 'w') as f:
        json.dump(data, f)
    
def merge_headlines_by_date():
    with open('data\headline_per_article.json', 'r') as f:
        data = json.load(f)
    f.close()
    headline_per_day = {}
    for i_s, sample in enumerate(data):
        # if i_s == 5000: break
        headline = sample['title']
        try:
            time = sample['formatted_time']
        except KeyError:
            continue
        
        if time not in headline_per_day:
            headline_per_day[time] = [headline]
        elif time in headline_per_day:
            headline_per_day[time].append(headline)
    
    with open('data/headline_per_date.json', 'w') as f:
        json.dump(headline_per_day, f)

def process_stock_prices():
    df = pd.read_csv(os.path.join(*'data\stock.csv'.split('\\')),
                     header = 0,
                    #  nrows=100
                     )
    
    data = df.to_dict('records')
    for i, sample in enumerate(data):
        # if i == 15: break
        formatted_time = [0]*3
        time = sample['Date']
        time = time.split('-')
        if len(time[0]) == 2:
            formatted_time[2] = time[0]
        else:
            formatted_time[2] = '0' + time[0]
        formatted_time[1] = months[time[1]]
        formatted_time[0] = '20' + time[2]
        
        sample['formatted_time'] = ''.join(formatted_time)
        
    with open('data\stockprice_per_date.json', 'w') as f:
        json.dump(data, f)

def make_samples():
    headlines = 'data\headline_per_date.json'
    stockprices = 'data\stockprice_per_date.json'
    comments = ''
    
    with open(headlines, 'r') as f:
        headline_data = json.load(f)
    f.close()
    with open(stockprices, 'r') as f:
        stockprice_data = json.load(f)
    f.close()
    
    samples = []
    stockprice_data = sorted(stockprice_data, key=itemgetter('formatted_time'))
    for i_sample, sample in enumerate(stockprice_data):
        if i_sample == 10: break
        
        merged_sample = {}
        date = sample['formatted_time']
        headline_sample = headline_data.get(str(date), [])
        merged_sample['input'] = headline_sample
        try:
            prev_input = {}
            prev_step = stockprice_data[i_sample-1]
            prev_input['prev_open'] = prev_step['Open']
            prev_input['prev_high'] = prev_step['High']
            prev_input['prev_low'] = prev_step['Low']
            prev_input['prev_close'] = prev_step['Close*']
            prev_input['prev_adj_close'] = prev_step['Adj Close**']
        except:
            merged_sample['time_input'] = {}

if __name__ == '__main__':
    print('hello world')
    
    # process_headlines()
    # merge_headlines_by_date()
    # process_stock_prices()
    # make_samples()
    
    # with open('data\stockprice_per_date.json', 'r') as f:
    #     data = json.load(f)
    # f.close()
    # for sample in data:
    #     if 20200301 < int(sample['formatted_time']):
    #         time_lagged = datetime.datetime.strptime(sample['formatted_time'], "%Y%m%d") - timedelta(days=1)
    #         epoch_time = time.mktime(time_lagged.timetuple())
    #         print(sample['formatted_time'], epoch_time)
    
    # with open('data\headline_per_article.json', 'r') as f:
    #     data = json.load(f)
    # df = pd.DataFrame(data)
    # df.to_csv('data\headline_per_article.csv', index=False)
    
    
    