#Can add midx to features dump, and then can use that in strategy.py for testing/creating a model from that
#python add_midx_to_features.py ../data/data_live.tsv
from math import log
import sys
import pandas as pd
from datetime import datetime
import multiprocessing

mids_to_add = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30]
input_filename = sys.argv[1]
cpu_count = multiprocessing.cpu_count()

timestamp_format = "%Y-%m-%d %H:%M:%S.%f"

def get_formatted_time_string(this_time):
    return datetime.utcfromtimestamp(this_time).strftime(timestamp_format)

def get_current_time_seconds_utc():
    return (datetime.utcnow()-datetime(1970,1,1)).total_seconds()

def get_future_mid(books, offset, sensitivity):
    '''
    Returns percent change of future midpoints for each data point in DataFrame
    of book data
    '''
    def future(timestamp):
        i = books.index.get_loc(timestamp+offset, method='nearest')
        if abs(books.index[i] - (timestamp+offset)) < sensitivity:
            return books.mid.iloc[i]
    return (books.index.map(future)/books.mid).apply(log)

def worker(params):
    num = params[0]
    data = params[1]
    split_interval = params[2]
    split_start = num*split_interval
    split_end = ((num+1)*split_interval)-1 + mids_to_add[-1]+5
    print "%s - Worker %s starting at %s, ending at %s" % (get_formatted_time_string(get_current_time_seconds_utc()), num, split_start, split_end)
    this_data = data.iloc[split_start:split_end].copy()
    for mid in mids_to_add:
        print "%s - Worker %s getting mid%s" % (get_formatted_time_string(get_current_time_seconds_utc()),num,mid)
        this_data["mid%s"%mid] = get_future_mid(this_data, mid, sensitivity=5)
    return this_data

def handler(data, split_interval):
    splits = range(0, cpu_count)
    parallel_arguments = []
    for split in splits:
        parallel_arguments.append([split, data, split_interval])
    pool = multiprocessing.Pool(cpu_count)
    data_array = pool.map(worker, parallel_arguments)
    pool.close()
    pool.join()
    final_data = pd.concat(data_array)
    final_data = final_data.groupby(final_data.index).max()
    # final_data = final_data[~final_data.index.duplicated(keep='first')]
    subset = []
    for mid in mids_to_add:
        subset.append("mid%s"%(mid))
    final_data = final_data.dropna(axis=0, subset=subset)
    return final_data.sort_index()

if __name__ == '__main__':
    print "%s - Reading data" % (get_formatted_time_string(get_current_time_seconds_utc()))
    data = pd.DataFrame.from_csv(input_filename, sep='\t')
    data = data.groupby(data.index).first()
    data_count = len(data)
    split_interval = data_count / cpu_count
    print "%s - Data length %s, cpu count %s, therefore split interval %s" % (
        get_formatted_time_string(get_current_time_seconds_utc()), data_count, cpu_count, split_interval)

    final_data = handler(data, split_interval)

    base_filename = '.'.join(input_filename.split('.')[:-1]) if '.' in input_filename else input_filename
    dump_filename = base_filename+".with_midx.tsv"
    print "%s - Dumping %s records to %s" % (get_formatted_time_string(get_current_time_seconds_utc()), len(final_data), dump_filename)
    final_data.to_csv(dump_filename, sep='\t')

    print "%s - Done" % (get_formatted_time_string(get_current_time_seconds_utc()))