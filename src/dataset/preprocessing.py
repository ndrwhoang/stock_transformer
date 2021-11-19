import json
import os
import pandas as pd
from pprint import pprint
from datetime import datetime

months = {
    'January': '1',
    'Jan': '1',
    'February': '2',
    'Feb': '2',
    'March': '3',
    'Mar': '3',
    'April': '4',
    'Apr': '4',
    'May': '5',
    'June': '6',
    'Jun': '6',
    'July': '7', 
    'Jul': '7',
    'August': '8',
    'Aug': '8',
    'September': '9',
    'Sept': '9',
    'Sep': '9',
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
            formatted_time[0] = months[time[1]]
            formatted_time[1] = time[0]
            formatted_time[2] = time[2]
        elif '-' in time:
            time = time.split('-')
            if len(time) != 3:
                continue
            formatted_time[0] = months[time[1]]
            formatted_time[1] = time[0]
            formatted_time[2] = '20' + time[2]
        else:
            time = time.split(' ')
            assert len(time) == 3
            formatted_time[0] = months[time[0]]
            formatted_time[1] = time[1]
            formatted_time[2] = time[2]
        
        # date_time = datetime.strptime(' '.join(formatted_time), '%m %d %Y')
        # sample['formatted_time'] = date_time
        sample['formatted_time'] = ' '.join(formatted_time)
        
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
                     nrows=100
                     )
    
    data = df.to_dict('records')
    for sample in data:
        formatted_time = [0]*3
        time = sample['Date']
        time = time.split('-')
        formatted_time[0] = months[time[1]]
        formatted_time[1] = time[0]
        formatted_time[2] = '20' + time[2]
        
        sample['formatted_time'] = ' '.join(formatted_time)
        
    with open('data\stockprice_per_date.json', 'w') as f:
        json.dump(data, f)
        

if __name__ == '__main__':
    # process_headlines()
    # merge_headlines_by_date()
    # process_stock_prices()
    print('hello world')
    
    