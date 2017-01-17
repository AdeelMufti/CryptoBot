#python crawl_btcc_trades_history.py <continuous true/false>
from urllib2 import urlopen
import sys
import json
import time
from pymongo import MongoClient
import datetime

api = 'http://data.btcchina.com'
trades_history_url = '{0}/data/historydata?since={1}&limit=5000&sincetype=time';
client = MongoClient()
db = client['cryptobot']
trades_collection = db['btcc_btccny_trades']
sleep_between_requests_secs = 1.0
timestamp_format = "%Y-%m-%d %H:%M:%S.%f"

def get_formatted_time_string(this_time):
    return datetime.datetime.utcfromtimestamp(this_time).strftime(timestamp_format)

def format_trade(trade):
    '''
    Formats trade data
    '''
    if all(key in trade for key in ('tid', 'amount', 'price', 'date')):
        trade['_id'] = int(trade.pop('tid'))
        trade['amount'] = float(trade['amount'])
        trade['price'] = float(trade['price'])
        trade['timestamp'] = float(trade.pop('date'))
    return trade


def get_json(url):
    '''
    Gets json from the API
    '''
    resp = urlopen(url,timeout=5)
    return json.load(resp, object_hook=format_trade), resp.getcode()

def get_latest_time(time_to_fetch):
    cursor = trades_collection.find().sort("$natural", -1).limit(1)
    for document in cursor:
        return document['timestamp']
    return time_to_fetch

continuous = False
if len(sys.argv) == 2:
    if sys.argv[1] == 'true':
        continuous = True
start_time = 1476093600
time_to_fetch = get_latest_time(start_time)
trades_count = 1
while continuous or trades_count > 0:
    start = time.time()
    url = trades_history_url.format(api, int(time_to_fetch))
    print '*** Getting trades at',get_formatted_time_string(time_to_fetch),time_to_fetch,'.',
    try:
        trades, code = get_json(url)
    except Exception as e:
        print e
        sys.exc_clear()
    else:
        if code != 200:
            print code
        else:
            for trade in trades:
                trades_collection.update_one({'_id': trade['_id']},
                                      {'$setOnInsert': trade}, upsert=True)
            time_to_fetch = get_latest_time(time_to_fetch)
            #time_to_fetch = trades[len(trades)-1]['timestamp']
            trades_count = len(trades)
            print 'Got',trades_count,'trades.'
        time_delta = time.time() - start
        if time_delta < sleep_between_requests_secs:
            time.sleep(sleep_between_requests_secs - time_delta)