import json
import pandas as pd
import os
import time

if __name__ == '__main__':
    df = pd.read_csv('data\comment_by_date\\reddit_covid.csv',
                     header=0,
                    #  nrows=100,
                     encoding='latin1'
                     )
    
    df['converted_time'] = pd.to_datetime(df['time'], unit='s')
    df.to_csv('data\comment_by_date\\reddit_covid_formatted_time.csv', index=False)