from model import features as f
import pymongo
import time
from datetime import datetime

client = pymongo.MongoClient()
db = client['cryptobot']
books_db = db['btcc_btccny_books']

def append_df_to_csv(df, csvFilePath, sep=","):
    import os
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=True, sep=sep)
    else:
        df.to_csv(csvFilePath, mode='a', index=True, sep=sep, header=False)

def get_current_time_seconds_utc():
    return (datetime.utcnow()-datetime(1970,1,1)).total_seconds()

def get_latest_book_timestamp():
    book = books_db.find({},{'_id': 1}).sort('_id', -1).limit(1).next()
    return book['_id']

last_data_timestamp = 0

while True:
    start = get_current_time_seconds_utc()

    this_data_timestamp = get_latest_book_timestamp()
    if this_data_timestamp < (start-3):
        # print "Data hasn't been updated in less than 3 seconds, skipping...",this_data_timestamp,start-3
        None
    elif last_data_timestamp != 0 and last_data_timestamp == this_data_timestamp:
        # print "Last data timestamp is equal to this data timestamp, skipping..."
        None
    else:
        last_data_timestamp = this_data_timestamp

        data = f.make_features(1,
                               [],
                               [10, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180],
                               [2, 4, 8],
                               True)

        append_df_to_csv(data, 'data/data_live.tsv', '\t')

    time_delta = get_current_time_seconds_utc()-start
    if time_delta < 1.0:
        time.sleep(1-time_delta)