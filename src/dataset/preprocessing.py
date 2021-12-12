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

def process_stock_prices(path_in):
    df = pd.read_csv(os.path.join(*path_in.split('\\')),
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
        
    with open(path_in.replace('.csv', '.json'), 'w') as f:
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

def _merge_training_data_with_sentiment():
    print('Start loading data')
    with open('data\\normal\stocknormaltime.json', 'r') as f:
        data = json.load(f)
    data = sorted(data, key=itemgetter('formatted_time')) 
    data = [sample for sample in data if int(sample['formatted_time']) > 20200201]
        
    sentiment_data = pd.read_csv('data\\normal\\normaldata.csv')
    sentiment_data = sentiment_data.to_dict('records')
    
    headline_out = {}
    comment_out = {}
    for i_c, sample in enumerate(sentiment_data):
        polarity = sample['polarity']
        subjectivity = sample['subjectivity']
        # if len(sample['time'].split(' ')[0].split('/')) != 3:
        #     continue
        try:
            time = int(sample['time'])
        except:
            ditmemergecaifilenguvaicalon = sample['time'].split(' ')[0].split('/')
            date = ditmemergecaifilenguvaicalon
            try:
                time = [date[2], date[0] if len(date[0]) == 2 else '0' + date[0], date[1] if len(date[1]) == 2 else '0' + date[1]]                
                time = int(''.join(time))
            except:
                continue
            
        # if polarity * subjectivity != 0:
        #     if sample['type'] == 'comment':
        #         # dit con me thang dau buoi re rach cho code roi ma van deo biet doi ngay thang
        #         if time not in headline_out:
        #             headline_out[time] = {
        #                 'polarity': [polarity],
        #                 'subjectivity': [subjectivity]
        #             }
        #         else:
        #             headline_out[time]['polarity'].append(polarity)
        #             headline_out[time]['subjectivity'].append(subjectivity)
        #     else:
        #         if time not in comment_out:
        #             comment_out[time] = {
        #                 'polarity': [polarity],
        #                 'subjectivity': [subjectivity]
        #             }
        #         else:
        #             comment_out[time]['polarity'].append(polarity)
        #             comment_out[time]['subjectivity'].append(subjectivity)
                    
        if polarity * subjectivity != 0:
            if time not in comment_out:
                comment_out[time] = {
                    'polarity': [polarity],
                    'subjectivity': [subjectivity]
                }
            else:
                comment_out[time]['polarity'].append(polarity)
                comment_out[time]['subjectivity'].append(subjectivity)
        
    
    for i_s, sample in enumerate(data):
        polarity = headline_out.get(int(sample['formatted_time']), {'polarity': [0]})['polarity']
        subjectivity = headline_out.get(int(sample['formatted_time']), {'subjectivity': [0]})['subjectivity']
        avg_hpol = sum(polarity) / len(polarity)
        avg_hsub = sum(subjectivity) / len(subjectivity)
        
        polarity = comment_out.get(int(sample['formatted_time']), {'polarity': [0]})['polarity']
        subjectivity = comment_out.get(int(sample['formatted_time']), {'subjectivity': [0]})['subjectivity']
        avg_rpol = sum(polarity) / len(polarity)
        avg_rsub = sum(subjectivity) / len(subjectivity)
        
        
        sample['reddit_polarity'] = avg_rpol
        sample['reddit_subjectivity'] = avg_rsub 
        sample['headline_polarity'] = avg_hpol
        sample['headline_subjectivity'] = avg_hsub

    
    with open('data\\normal\\normal_full.json', 'w') as of:
        json.dump(data, of)


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
        
def stock_with_sentiment_per_date():
    with open('data\covid\stockprice_per_date.json', 'r') as f:
        data = json.load(f)
    print(len(data))
    
    data = sorted(data, key=itemgetter('formatted_time')) 
    data = [sample for sample in data if int(sample['formatted_time']) > 20200301]
    
    with open('data\covid\\reddit_covid_sentiment.json', 'r') as f:
        sentiment = json.load(f)
    
    date_comment = {}
    for i_c, comment in enumerate(sentiment):
        polarity = comment['polarity']
        subjectivity = comment['subjectivity']
        if polarity * subjectivity != 0:
            if comment['formatted_time'] not in date_comment:
                date_comment[int(comment['formatted_time'])] = {
                    'polarity': [polarity],
                    'subjectivity': [subjectivity]
                }
            else:
                date_comment[int(comment['formatted_time'])]['polarity'].append(polarity)
                date_comment[int(comment['formatted_time'])]['subjectivity'].append(subjectivity)
    
    for i_s, sample in enumerate(data):
        try:
            polarity = date_comment[int(sample['formatted_time'])]['polarity']
            subjectivity = date_comment[int(sample['formatted_time'])]['subjectivity']
            dm_polarity = sum(polarity) / len(polarity)
            dm_subjectivity = sum(subjectivity) / len(subjectivity)
            sample['reddit_polarity'] = dm_polarity
            sample['reddit_subjectivity'] = dm_subjectivity
        except KeyError:
            pass
        
    with open('data\covid\stock_with_covid_sentiment.json', 'w') as f:
        json.dump(data, f)
        
def process_normal_time_json():
    with open('data\\normal\\normal_ful_wrong_time.json', 'r') as f:
        data = json.load(f)
        
    for i_s, sample in enumerate(data):
        date = sample['Date'].split('-')
        formatted_time = [date[2] if len(date[2]) == 4 else '20' + date[2],
                          months[date[1]] if len(months[date[1]]) == 2 else '0' + date[1], 
                          date[0] if len(date[0]) == 2 else '0' + date[0]]
        sample['formatted_time'] = ''.join(formatted_time)
    
    data = sorted(data, key=itemgetter('formatted_time')) 
    with open('data\\normal\\normal_ful.json', 'w') as f:
        json.dump(data, f)

        
        
    
if __name__ == '__main__':
    print('hello world')
    process_normal_time_json()
    # process_headlines()
    # merge_headlines_by_date()
    # process_stock_prices('data\\normal\stocknormaltime.csv')
    # make_samples()
    # merge_comments_by_date()
    # merge_reddit_comments_fincris()
    
    # process_reddit_comments_fincris()
    # convert_epoch_time()
    # process_reddit_comments()
    
    # sentiment_per_date()
    # _merge_training_data_with_sentiment()