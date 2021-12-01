import json
import time
import datetime
from datetime import timedelta
import os
import csv
import pandas as pd
from pprint import pprint
from operator import index, itemgetter

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
    df = pd.read_csv(os.path.join(*'data/covid/mergeddata.csv'.split('\\')),
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
        
    with open('data/covid/headline_per_article.json', 'w') as f:
        json.dump(data, f)
    
def merge_headlines_by_date():
    with open('data/covid/headline_per_article.json', 'r') as f:
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
    
    with open('data/covid/headline_per_date.json', 'w') as f:
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

def process_reddit_comments():
    # For covid
    df = pd.read_csv(os.path.join(*'data\comment_by_date_feb\combined_comments.csv'.split('\\')),
                     header=0,
                    #  nrows=100
                     )
    
    data = df.to_dict('records')
    print(len(data))
    data_out = []
    for i, sample in enumerate(data):
        # if i == 5: break
        sample_out = {}
        date = sample['converted_time'].split(' ')[0].split('-')
        # normalized_date = [0]*3
        # normalized_date[0] = date[2]
        # normalized_date[1] = months[date[0]]
        # normalized_date[2] = date[1] if len(date[1]) == 2 else '0' + date[1]
        # normalized_date = [date[2], date[0] if len(date[0]) == 2 else '0' + date[0], date[1] if len(date[1]) == 2 else '0' + date[1]]
        normalized_date = [date[0], date[1] if len(date[1]) == 2 else '0' + date[1], date[2] if len(date[2]) == 2 else '0' + date[2]]

        
        sample_out['comment'] = sample['comment']
        sample_out['formatted_time'] = ''.join(normalized_date)
        try:
            sample_out['polarity'] = sample['polarity']
            sample_out['subjectivity'] = sample['subjectivity']
        except KeyError:
            pass
    
        if len(sample_out['comment'].split(' ')) > 3 and len(sample_out['comment'].split(' ')) < 200 and 'https://' not in sample_out['comment'] and 'http://' not in sample_out['comment']:
            data_out.append(sample_out)
    
    print(len(data_out))
    print(data_out)
    
    with open('data\comment_by_date_feb\combined_comments.json', 'w') as f:
        json.dump(data_out, f)
 
def merge_comments_by_date():
    with open('data\\financris\\reddit_financris_merged.json', 'r') as f:
        data = json.load(f)
    comment_per_day = {}
    try:
        test = data[0]['polarity']
        for i_s, sample in enumerate(data):
            # if i_s == 15: break
            # comment = sample['comment']
            # polarity = sample['polarity']
            # subjectivity = sample['subjectivity']
            if sample['formatted_time'] not in comment_per_day:
                comment_per_day[sample['formatted_time']] = [{'comment': sample['comment'].replace('\\n', '').replace('\\r', ''),
                                                            'polarity': sample['polarity'],
                                                            'subjectivity': sample['subjectivity']}]
            else:
                comment_per_day[sample['formatted_time']].append({'comment': sample['comment'].replace('\\n', '').replace('\\r', ''),
                                                                'polarity': sample['polarity'],
                                                                'subjectivity': sample['subjectivity']})
    except KeyError:
        for i_s, sample in enumerate(data):
            if sample['formatted_time'] not in comment_per_day:
                comment_per_day[sample['formatted_time']] = [{'comment': sample['comment'].replace('\\n', '').replace('\\r', '')}]
            else:
                comment_per_day[sample['formatted_time']].append({'comment': sample['comment'].replace('\\n', '').replace('\\r', '')})
    
    with open('data\\financris\comment_per_date.json', 'w') as f:
        json.dump(comment_per_day, f)
 
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

def convert_epoch_time():
    df = pd.read_csv('data\covid\stock.csv',
                    header=0,
                #  nrows=100
                    )
    data = df.to_dict('records')
    
    for sample in data:
        a = datetime.datetime.strptime(sample['Date'], '%d-%b-%y')
        a = a - timedelta(days=1)
        a = a.timestamp()
        print(sample['Date'], a)
    
def process_reddit_comments_fincris():
    data_paths = []
    for file in os.listdir('data\\financris'):
        if file.endswith('.json'):
            data_paths.append(os.path.join('data', 'financris', file).replace("\\","/"))
    print(data_paths)
    
    data_all = []
    for data_path in data_paths:
        with open(data_path, 'r') as f:
            data = json.load(f)
        # print(len(data))
        data_all.extend(data)
    
    for i_s, sample in enumerate(data_all):
        epoch_time = sample['time']
        formatted_time = time.strftime('%Y%m%d', time.localtime(int(epoch_time)))
        sample['formatted_time'] = formatted_time    
    
    data_out = []
    for i_s, sample in enumerate(data_all):
        if len(sample['comment'].split(' ')) > 3 and len(sample['comment'].split(' ')) < 200 and 'https://' not in sample['comment'] and 'http://' not in sample['comment']:
            data_out.append(sample)

    
    with open('data\\financris\\reddit_financris_merged.json', 'w') as f:
        json.dump(data_out, f)


def merge_reddit_comments_fincris():
    data_paths = []
    for file in os.listdir('data\comment_by_date_feb'):
        if file.endswith('.csv'):
            data_paths.append(os.path.join('data', 'comment_by_date_feb', file).replace("\\","/"))
    print(data_paths)
    combined_csv = pd.concat([pd.read_csv(f) for f in data_paths])
    df_out = combined_csv[['subreddit', 'body', 'created_utc']]
    df_out = df_out.rename(columns={'body': 'comment'})
    df_out['converted_time'] = pd.to_datetime(df_out['created_utc'], unit='s')
    df_out.to_csv('data\comment_by_date_feb\combined_comments.csv', index=False)
        

if __name__ == '__main__':
    print('hello world')
    
    # process_headlines()
    # merge_headlines_by_date()
    # process_stock_prices()
    # make_samples()
    # merge_comments_by_date()
    # merge_reddit_comments_fincris()
    
    # process_reddit_comments_fincris()
    # convert_epoch_time()
    # process_reddit_comments()
    with open('data\\financris\\reddit_financris_merged.json', 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(df.head())
    # df.to_excel('data\\financris\\reddit_financris_merged.xlsx', index=False)