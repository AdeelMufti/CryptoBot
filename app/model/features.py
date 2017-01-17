#python features.py <limit> <dump file for features>.pkl
#python -W ignore -u features.py 0 ../data/data.pkl
#python -W ignore -u features.py 1035650 ../data/data.pkl
import datetime
import pymongo
import pandas as pd
from math import log
from time import time
import sys
from scipy.stats import linregress
import pickle
import numpy as np

client = pymongo.MongoClient()
db = client['cryptobot']
books_db = db['btcc_btccny_books']
trades_db = db['btcc_btccny_trades']
# ticks_db = db['btcc_btccny_ticks']
timestamp_format = "%Y-%m-%d %H:%M:%S.%f"


def get_formatted_time_string(this_time):
    return datetime.datetime.utcfromtimestamp(this_time).strftime(timestamp_format)

def get_book_df(limit, convert_timestamps=False, skip=0, live=False, theoretical_order=None):
    '''
    Returns a DataFrame of book data
    '''
    if live:
        cursor = books_db.find().sort('_id', -1).limit(limit)
    else:
        cursor = books_db.find().sort('_id', 1).skip(skip).limit(limit)
    books = pd.DataFrame(list(cursor))

    books = books.set_index('_id')
    if convert_timestamps:
        books.index = pd.to_datetime(books.index, unit='s')

    if live and theoretical_order:
        order_type = theoretical_order[0]
        price = theoretical_order[1]
        amount_order_in_btc = theoretical_order[2]
        new_order = {}
        new_order['timestamp'] = books.iloc[0].asks[0]['timestamp']
        new_order['price'] = price
        new_order['amount'] = amount_order_in_btc
        if order_type == 'bid':
            books.iloc[0].bids = books.iloc[0].bids[:9]
            bids = [new_order]
            for order in books.iloc[0].bids:
                bids.append(order)
            books.iloc[0].bids = bids
        elif order_type == 'ask':
            books.iloc[0].asks = books.iloc[0].asks[1:]
            asks = []
            for order in books.iloc[0].asks:
                asks.append(order)
            asks.append(new_order)
            books.iloc[0].asks = asks

    def to_df(x):
        return pd.DataFrame(x[:10])

    return books.applymap(to_df).sort_index()


def get_width_and_mid(books):
    '''
    Returns width of best market and midpoint for each data point in DataFrame
    of book data
    '''
    best_bid = books.bids.apply(lambda x: x.price[0])
    best_ask = books.asks.apply(lambda x: x.price[len(x.price)-1])
    return best_ask-best_bid, (best_bid + best_ask)/2

#Since asks/bids seem to be repeating in books for a while, at most (observed so far) every 15 seconds, we want to get the future mid within plus/minus 25 seconds
def get_future_mid(books, offset, sensitivity=1):
    '''
    Returns percent change of future midpoints for each data point in DataFrame
    of book data
    '''

    def future(timestamp):
        i = books.index.get_loc(timestamp+offset, method='nearest')
        if abs(books.index[i] - (timestamp+offset)) < sensitivity:
            return books.mid.iloc[i]
    return (books.index.map(future)/books.mid).apply(log)


def get_power_imbalance(books, n=10, power=2):
    '''
    Returns a measure of the imbalance between bids and offers for each data
    point in DataFrame of book data
    '''

    def calc_imbalance(book):
        def calc(x):
            return 0 if x.price-book.mid==0 else x.amount*(.5*book.width/(x.price-book.mid))**power
        bid_imbalance = book.bids.iloc[:n].apply(calc, axis=1)
        ask_imbalance = book.asks.iloc[:n].apply(calc, axis=1)
        return (bid_imbalance-ask_imbalance).sum()
    imbalance = books.apply(calc_imbalance, axis=1)
    return imbalance


def get_power_adjusted_price(books, n=10, power=2):
    '''
    Returns the percent change of an average of order prices weighted by inverse
    distance-wieghted volume for each data point in DataFrame of book data
    '''

    def calc_adjusted_price(book):
        def calc(x):
            return 0 if x.price-book.mid==0 else x.amount*(.5*book.width/(x.price-book.mid))**power
        bid_inv = 1/book.bids.iloc[:n].apply(calc, axis=1)
        ask_inv = 1/book.asks.iloc[:n].apply(calc, axis=1)
        bid_price = book.bids.price.iloc[:n]
        ask_price = book.asks.price.iloc[:n]
        sum_numerator = (bid_price*bid_inv + ask_price*ask_inv).sum()
        sum_denominator = (bid_inv + ask_inv).sum()
        # if np.isnan(sum_numerator) or np.isinf(sum_numerator) or sum_numerator == 0.0 or np.isnan(sum_denominator) or np.isinf(sum_denominator) or sum_denominator == 0.0:
        #     return 0
        quotient = sum_numerator / sum_denominator
        # if quotient < 0.0:
        #     return 0
        return quotient
    adjusted = books.apply(calc_adjusted_price, axis=1)
    return (adjusted/books.mid).apply(log).fillna(0)


def get_trade_df(books, min_ts, max_ts, live, convert_timestamps=False, theoretical_trade=None):
    '''
    Returns a DataFrame of trades in time range
    '''
    if not live:
        query = {'timestamp': {'$gt': min_ts, '$lt': max_ts}}
        cursor = trades_db.find(query).sort('_id', pymongo.ASCENDING)
    else:
        cursor = trades_db.find({}).sort('$natural', pymongo.DESCENDING).limit(3000)

    trades = pd.DataFrame(list(cursor))

    if live:
        trades = trades[trades.timestamp <= max_ts] #The above is not gte or lte because later on we do a pandas search sorted on a much larger range if not live, and that includes equals
        trades = trades[trades.timestamp >= min_ts]
        trades = trades.sort_values(['timestamp'])

    if live and theoretical_trade:
        trade_type = theoretical_trade[0]
        price = theoretical_trade[1]
        amount_trade_in_btc = theoretical_trade[2]
        trade = {}
        trade['_id']=trades.iloc[-1]['_id']+1
        trade['timestamp']=books.index[0]-8 #So it can be included in the -7.5 offset for trades
        trade['price']=price
        trade['amount']=amount_trade_in_btc
        trade['type']='buy' if trade_type=='bid' else 'sell'
        trades = pd.concat([trades,pd.DataFrame(trade,index=[0])])
        trades = trades.sort_values(['timestamp'])

    #for index, row in trades.iterrows():
    #    print row['timestamp']

    if not trades.empty:
        trades = trades.set_index('_id')
        if convert_timestamps:
            trades.index = pd.to_datetime(trades.index, unit='s')

    # for i in xrange(len(trades)):
    #     print i, trades.index[i],trades.iloc[i]['amount'],trades.iloc[i]['price'],trades.iloc[i]['timestamp'],trades.iloc[i]['type']

    return trades

def get_trades_indexes(books, trades, offset, live=False):
    '''
    Returns indexes of trades in offset range for each data point in DataFrame
    of book data
    '''
    def trades_indexes(ts):
        ts = int(ts)
        i_0 = trades.timestamp.searchsorted([ts-offset], side='left')[0]
        # if live:
        #     i_n = -1
        # else:
        #     i_n = trades.timestamp.searchsorted([ts-1], side='right')[0]
        i_n = trades.timestamp.searchsorted([ts - 7.5], side='right')[0] #because live trades lag behind for about 7-10 seconds
        if i_n == len(trades):
            i_n = i_n-1
        #print offset, ts, len(trades), i_0, i_n, trades.iloc[i_0].timestamp, trades.iloc[i_n].timestamp
        return (i_0, i_n)
    return books.index.map(trades_indexes)

def get_trades_count(books, trades):
    '''
    Returns a count of trades for each data point in DataFrame of book data
    '''
    def count(x):
        return len(trades.iloc[x.trades_indexes[0]:x.trades_indexes[1]])
    return books.apply(count, axis=1)


def get_trades_average(books, trades):
    '''
    Returns the percent change of a volume-weighted average of trades for each
    data point in DataFrame of book data
    '''

    def mean_trades(x):
        trades_n = trades.iloc[x.trades_indexes[0]:x.trades_indexes[1]]
        if not trades_n.empty:
            return (trades_n.price*trades_n.amount).sum()/trades_n.amount.sum()
    return (books.mid/books.apply(mean_trades, axis=1)).apply(log).fillna(0)


def get_aggressor(books, trades):
    '''
    Returns a measure of whether trade aggressors were buyers or sellers for
    each data point in DataFrame of book data
    '''

    def aggressor(x):
        trades_n = trades.iloc[x.trades_indexes[0]:x.trades_indexes[1]]
        if trades_n.empty:
            return 0
        buys = trades_n['type'] == 'buy'
        buy_vol = trades_n[buys].amount.sum()
        sell_vol = trades_n[~buys].amount.sum()
        return buy_vol - sell_vol
    return books.apply(aggressor, axis=1)


def get_trend(books, trades):
    '''
    Returns the linear trend in previous trades for each data point in DataFrame
    of book data
    '''

    def trend(x):
        trades_n = trades.iloc[x.trades_indexes[0]:x.trades_indexes[1]]
        if len(trades_n) < 3:
            return 0
        else:
            return linregress(trades_n.index.values, trades_n.price.values)[0]
    return books.apply(trend, axis=1)


# def get_tick_df(min_ts, max_ts, live, convert_timestamps=False):
#     '''
#     Returns a DataFrame of ticks in time range
#     '''
#     if not live:
#         query = {'_id': {'$gt': min_ts, '$lt': max_ts}}
#         cursor = ticks_db.find(query).sort('_id', pymongo.ASCENDING)
#     else:
#         cursor = ticks_db.find({}).sort('$natural', pymongo.DESCENDING).limit(1)
#
#     ticks = pd.DataFrame(list(cursor))
#
#     if not ticks.empty:
#         ticks = ticks.set_index('_id')
#         if convert_timestamps:
#             ticks.index = pd.to_datetime(ticks.index, unit='s')
#     return ticks
#
# def get_ticks_indexes(books, ticks):
#     '''
#     Returns indexes of ticks closest to each data point in DataFrame
#     of book data
#     '''
#     def ticks_indexes(ts):
#         ts = int(ts)
#         return ticks.index.get_loc(ts, method='nearest')
#     return books.index.map(ticks_indexes)
#
# def get_buys_from_ticks(books, ticks):
#     '''
#     Returns a count of trades for each data point in DataFrame of book data
#     '''
#     def get_buy(x):
#         return ticks.iloc[x.ticks_indexes].buy
#     return books.apply(get_buy, axis=1)
#
# def get_sells_from_ticks(books, ticks):
#     '''
#     Returns a count of trades for each data point in DataFrame of book data
#     '''
#     def get_sell(x):
#         return ticks.iloc[x.ticks_indexes].sell
#     return books.apply(get_sell, axis=1)

def check_times(books):
    '''
    Returns list of differences between collection time and max book timestamps
    for verification purposes
    '''
    time_diff = []
    for i in range(len(books)):
        book = books.iloc[i]
        ask_ts = max(book.asks.timestamp)
        bid_ts = max(book.bids.timestamp)
        ts = max(ask_ts, bid_ts)
        time_diff.append(book.name-ts)
    return time_diff


def make_features(limit, mid_offsets,
                  trades_offsets, powers, live=False, skip=0,
                  theoretical_order=None, theoretical_trade=None):
    '''
    Returns a DataFrame with targets and features
    '''
    start = time()
    stage = time()
    # Book related features:
    books = get_book_df(limit,skip=skip,live=live,theoretical_order=theoretical_order)
    if not live:
        print 'get book data run time:', (time()-stage)/60, 'minutes'
        stage = time()
    books['width'], books['mid'] = get_width_and_mid(books)
    if not live:
        print 'width and mid run time:', (time()-stage)/60, 'minutes'
        stage = time()
    for n in mid_offsets:
        books['mid{}'.format(n)] = get_future_mid(books, n)
    if not live:
        books = books.dropna()
        print 'offset mids run time:', (time()-stage)/60, 'minutes'
        stage = time()
    for p in powers:
        books['imbalance{}'.format(p)] = get_power_imbalance(books, 10, p)
        books['adj_price{}'.format(p)] = get_power_adjusted_price(books, 10, p)
    if not live:
        print 'power calcs run time:', (time()-stage)/60, 'minutes'
        stage = time()
    books = books.drop(['bids', 'asks'], axis=1)

    # Trade related features:
    min_ts = books.index.min() - trades_offsets[-1]
    max_ts = books.index.max()
    if live:
        max_ts += 10
    #print "Getting trades between '",datetime.datetime.utcfromtimestamp(min_ts).strftime(timestamp_format), "' and '", datetime.datetime.utcfromtimestamp(max_ts).strftime(timestamp_format),"'"
    trades = get_trade_df(books, min_ts, max_ts, live, theoretical_trade=theoretical_trade)
    for n in trades_offsets:
        if trades.empty:
            books['trades_indexes'] = 0
            books['t{}_count'.format(n)] = 0
            books['t{}_av'.format(n)] = 0
            books['agg{}'.format(n)] = 0
            books['trend{}'.format(n)] = 0
        else:
            books['trades_indexes'] = get_trades_indexes(books, trades, n, live)
            books['t{}_count'.format(n)] = get_trades_count(books, trades)
            books['t{}_av'.format(n)] = get_trades_average(books, trades)
            books['agg{}'.format(n)] = get_aggressor(books, trades)
            books['trend{}'.format(n)] = get_trend(books, trades)
    if not live:
        print 'trade features run time:', (time()-stage)/60, 'minutes'
        stage = time()
    books = books.drop('trades_indexes', axis=1)

    # # Ticks
    # ticks = get_tick_df(min_ts, max_ts, live)
    # if ticks.empty:
    #     books['ticks_indexes'] = 0
    #     books['tick_buy'] = 0
    #     books['tick_sell'] = 0
    # else:
    #     books['ticks_indexes'] = get_ticks_indexes(books, ticks)
    #     books['tick_buy'] = get_buys_from_ticks(books, ticks)
    #     books['tick_sell'] = get_sells_from_ticks(books, ticks)
    # if not live:
    #     print 'tick features run time:', (time()-stage)/60, 'minutes'
    #     stage = time()
    # books = books.drop('ticks_indexes', axis=1)

    if not live:
        print 'make_features run time:', (time() - start) / 60, 'minutes'

    return books

def make_data(limit, skip=0):
    '''
    Convenience function for calling make_features
    '''
    # data = make_features(limit=limit,
    #                      mid_offsets=[30],
    #                      trades_offsets=[30, 60, 120, 180],
    #                      powers=[2, 4, 8],
    #                      skip=skip)
    data = make_features(limit=limit,
                         mid_offsets=[5, 10, 15, 20, 25, 30, 35, 40, 45],
                         trades_offsets=[10, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180],
                         powers=[2, 4, 8],
                         skip=skip)
    return data

if __name__ == '__main__' and len(sys.argv) == 3:
    print 'Starting at', get_formatted_time_string(time())
    data = make_data(int(sys.argv[1]))
    output_filename = sys.argv[2]
    base_filename = '.'.join(output_filename.split('.')[:-1]) if '.' in output_filename else output_filename
    data.to_csv(base_filename+".tsv", sep='\t')
    with open(base_filename+".pkl", 'w+') as file:
        pickle.dump(data, file)
    file.close()
    print 'Ending at', get_formatted_time_string(time())