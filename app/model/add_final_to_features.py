#python add_final_to_features.py ../data/data_live.with_midx.tsv
import sys
import pandas as pd
from datetime import datetime
import numpy as np
import multiprocessing

mids = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30]
threshold_percent = 0.05
threshold = threshold_percent/100
input_filename = sys.argv[1]
cpu_count = multiprocessing.cpu_count()

timestamp_format = "%Y-%m-%d %H:%M:%S.%f"

def get_formatted_time_string(this_time):
    return datetime.utcfromtimestamp(this_time).strftime(timestamp_format)

def get_current_time_seconds_utc():
    return (datetime.utcnow()-datetime(1970,1,1)).total_seconds()

def all_signs_equal(data, mids):
    mids_as_list = []
    for mid in mids:
        mids_as_list.append(data["mid%s"%mid])
    mid_signs = np.sign(mids_as_list)
    all_signs_equal = True
    sign = mid_signs[0]
    for mid_sign in mid_signs:
        if sign != mid_sign:
            all_signs_equal = False
            break
    return all_signs_equal

def average_all_mids(data, mids):
    mids_as_list = []
    for mid in mids:
        mids_as_list.append(data["mid%s"%mid])
    average = np.asarray(mids_as_list).mean()
    return average

def get_final(books):
    def final(book):
        # if not all_signs_equal(book,mids):
        #     return 0
        # else:
        average = average_all_mids(book,mids)

        if average > threshold:
            return 1
        elif average < -threshold:
            return -1
        else:
            return 0

    return books.apply(final,axis=1)

def worker(params):
    num = params[0]
    data = params[1]
    split_interval = params[2]
    split_start = num*split_interval
    split_end = ((num+1)*split_interval)+5
    print "%s - Worker %s starting at %s, ending at %s" % (get_formatted_time_string(get_current_time_seconds_utc()), num, split_start, split_end)
    this_data = data.iloc[split_start:split_end].copy()
    this_data['final'] = get_final(this_data)
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
    #final_data = final_data[~final_data.index.duplicated(keep='first')]
    return final_data.sort_index()

if __name__ == '__main__':
    print "%s - Reading data" % (get_formatted_time_string(get_current_time_seconds_utc()))
    data = pd.DataFrame.from_csv(input_filename, sep='\t')
    #data = data.groupby(data.index).first()
    #data = data.dropna(0)
    data_count = len(data)
    split_interval = data_count / cpu_count
    print "%s - Data length %s, cpu count %s, therefore split interval %s" % (
        get_formatted_time_string(get_current_time_seconds_utc()), data_count, cpu_count, split_interval)

    final_data = handler(data, split_interval)

    base_filename = '.'.join(input_filename.split('.')[:-1]) if '.' in input_filename else input_filename
    dump_filename = base_filename+".with_final.tsv"
    print "%s - Dumping %s records to %s" % (get_formatted_time_string(get_current_time_seconds_utc()), len(final_data), dump_filename)
    final_data.to_csv(dump_filename, sep='\t')

    print "%s - Done" % (get_formatted_time_string(get_current_time_seconds_utc()))