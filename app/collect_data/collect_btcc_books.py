#python collect_btcc_books.py
from urllib2 import urlopen
import time
import json
from pymongo import MongoClient
import sys
import datetime

api = 'http://data.btcchina.com'
book_url = '{0}/data/orderbook?market=btccny&limit=10'.format(api)
client = MongoClient()
db = client['cryptobot']
books_collection = db['btcc_btccny_books']
sleep_between_requests_secs = 1.0
timestamp_format = "%Y-%m-%d %H:%M:%S.%f"

def get_formatted_time_string(this_time):
    return datetime.datetime.utcfromtimestamp(this_time).strftime(timestamp_format)

def format_book_entry(entry):
    '''
    Converts book data to float
    '''
    bids = entry['bids']
    new_bids = []
    for row in bids:
        new_row = {}
        new_row['price'] = float(row[0])
        new_row['amount'] = float(row[1])
        new_row['timestamp'] = float(entry['date'])
        new_bids.append(new_row)
    entry['bids'] = new_bids

    asks = entry['asks']
    new_asks = []
    for row in asks:
        new_row = {}
        new_row['price'] = float(row[0])
        new_row['amount'] = float(row[1])
        new_row['timestamp'] = float(entry['date'])
        new_asks.append(new_row)
    entry['asks'] = new_asks

    return entry


def get_json(url):
    '''
    Gets json from the API
    '''
    resp = urlopen(url,timeout=5)
    return json.load(resp, object_hook=format_book_entry), resp.getcode()


print 'Running...'
while True:
    start = time.time()
    print '*** Getting books at',get_formatted_time_string(start),start
    try:
        book, code = get_json(book_url)
    except Exception as e:
        print e
        sys.exc_clear()
    else:
        if code != 200:
            print code
        else:
            book.pop('date')
            book['_id'] = time.time()
            books_collection.insert_one(book)
            time_delta = time.time()-start
            if time_delta < sleep_between_requests_secs:
                time.sleep(sleep_between_requests_secs-time_delta)
