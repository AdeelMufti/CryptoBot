#python collect_btcc_ticks.py
from urllib2 import urlopen
import time
import json
from pymongo import MongoClient
import sys
import datetime

api = 'http://data.btcchina.com'
tick_url = '{0}/data/ticker?market=btccny'.format(api)
client = MongoClient()
db = client['cryptobot']
ticks_collection = db['btcc_btccny_ticks']
sleep_between_requests_secs = 1.0
timestamp_format = "%Y-%m-%d %H:%M:%S.%f"

def get_formatted_time_string(this_time):
    return datetime.datetime.utcfromtimestamp(this_time).strftime(timestamp_format)

def get_json(url):
    '''
    Gets json from the API
    '''
    resp = urlopen(url,timeout=5)
    entry = json.load(resp)['ticker']
    entry['_id'] = entry.pop('date')
    for key in entry:
        entry[key] = float(entry[key])
    # tick = {}
    # tick['_id'] = float(entry['date'])
    # tick['high'] = float(entry['high'])
    # tick['low'] = float(entry['low'])
    # tick['buy'] = float(entry['buy'])
    # tick['sell'] = float(entry['sell'])
    return entry, resp.getcode()

print 'Running...'
while True:
    start = time.time()
    print '*** Getting tick at',get_formatted_time_string(start),start,'.',
    try:
        tick, code = get_json(tick_url)
    except Exception as e:
        print e
        sys.exc_clear()
    else:
        if code != 200:
            print code
        else:
            print 'Gotten it for',get_formatted_time_string(tick['_id']),tick['_id']
            ticks_collection.update_one({'_id': tick['_id']},
                                         {'$setOnInsert': tick}, upsert=True)
            time_delta = time.time()-start
            if time_delta < sleep_between_requests_secs:
                time.sleep(sleep_between_requests_secs-time_delta)
