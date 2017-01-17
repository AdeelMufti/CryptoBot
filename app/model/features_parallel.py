#python -W ignore -u features_parallel.py <initial skip> <limit in ascending (earlier to later), 0 for all> <output file>
#python -W ignore -u features_parallel.py 0 1231625 ./data1.pkl && python -W ignore -u features_parallel.py 1231625-150 1231625 ./data2.pkl (-150 on the second command's split because from the first run the last 150 get dropped!
#python -W ignore -u features_parallel.py 0 0 ../data/data.pkl
import datetime
import multiprocessing
import pymongo
import pickle
import sys
import features
import pandas as pd
from time import time

data = pd.concat

client = pymongo.MongoClient()
db = client['cryptobot']
books_db = db['btcc_btccny_books']
cpu_count = multiprocessing.cpu_count()
initial_skip = int(sys.argv[1])
limit = int(sys.argv[2])
if limit == 0:
    limit = books_db.find().count()
output_filename = sys.argv[3]
timestamp_format = "%Y-%m-%d %H:%M:%S.%f"
each_limit = limit/cpu_count

def get_formatted_time_string(this_time):
    return datetime.datetime.utcfromtimestamp(this_time).strftime(timestamp_format)

def worker(num):
    skip = (each_limit * num) + initial_skip
    print 'Worker #%s starting starting at record %s, limiting to %s records' % (num,skip,each_limit)
    if num == cpu_count-1:
        return features.make_data(each_limit, skip)
    else:
        return features.make_data(each_limit + 150, skip)

def handler():
    splits = range(0, cpu_count)
    pool = multiprocessing.Pool(cpu_count)
    data_array = pool.map(worker, splits)
    pool.close()
    pool.join()
    data = pd.concat(data_array)
    data = data[~data.index.duplicated(keep='first')]
    #data = data.groupby(data.index).first()
    return data.sort_index()

if __name__ == '__main__':
    start = time()
    print 'Starting parallel features gen at', get_formatted_time_string(start)
    print cpu_count, 'threads will work on', each_limit, 'records each, totalling to', limit, 'records.'
    data = handler()
    print 'Done generating. Dumping...'
    base_filename = '.'.join(output_filename.split('.')[:-1]) if '.' in output_filename else output_filename
    data.to_csv(base_filename+".tsv", sep='\t')
    with open(base_filename+".pkl", 'w+') as file:
        pickle.dump(data, file)
    file.close()
    print len(data),'Records produced into '+base_filename+".tsv and "+base_filename+".pkl"
    print 'Ending parallel features gen at', get_formatted_time_string(time())
    print 'Took', (time() - start) / 60, 'minutes to run.'
