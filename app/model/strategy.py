#python strategy.gy <data.pkl> <split percent for train/test> <prediction duration in secs> <threshold percent> <dump model/etc true/false>
#python strategy.py ../data/data.pkl 50 30 0.01 true
# https://aws.amazon.com/ec2/pricing/on-demand/ c4.8xlarge = 36 processors, 60gb ram = $1.675/hr
# With 3938164 records, each instance takes memory: virt=9.8gb, res=7.3g =~ 55gb used for 7 in free -g =~ 7.8gb/instance

import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import pickle
import sys
import model
import pandas as pd
from time import time
import multiprocessing
from os import path
import xgboost as xgb

do_parallel_search = False
timestamp_format = "%Y-%m-%d %H:%M:%S.%f"
#input_model_filename = "../data/model_featuresNewAll_1476108655.82-1478514760.62_duration30_times100.pkl"
#input_cols_filename = "../data/cols_featuresNewAll.pkl"
input_model_filename = ""
input_cols_filename = ""
output_cols_filename = "../data/cols_featuresOldAll.pkl" #Todo: create this filename automatically
y_multiplied_by = 100

def get_output_model_filename(features, train_start, train_end, prediction_duration):
    return "../data/model_features%s_%s-%s_duration%s_times%s.pkl"%(features, train_start, train_end, prediction_duration, y_multiplied_by)

def get_formatted_time_string(this_time):
    return datetime.datetime.utcfromtimestamp(this_time).strftime(timestamp_format)

def fit_and_trade(period, data, features, cols, split, split_percent, prediction_duration, threshold, threshold_percent, do_dump):
    '''
    Fits and backtests a theoretical trading strategy
    '''
    X = data[cols]
    # y = data["mid%s"%prediction_duration]
    y = data["final"]
    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]*y_multiplied_by
    y_test = y.iloc[split:]

    print 'Data split at {}%. {}/{} records will be used for training/fitting. Rest used for testing.'.format(split_percent,split,len(data))
    print 'Train data starts at %s (%s), ends at %s (%s)'%(get_formatted_time_string(X_train.index.values[0]),X_train.index.values[0],get_formatted_time_string(X_train.index.values[-1]),X_train.index.values[-1])
    print 'Test data starts at %s (%s), ends at %s (%s)'%(get_formatted_time_string(X_test.index.values[0]),X_test.index.values[0],get_formatted_time_string(X_test.index.values[-1]),X_test.index.values[-1])
    train_start = X_train.index.values[0]
    train_end = X_train.index.values[-1]

    if do_dump == 'true':
        print 'Model output file will be:',get_output_model_filename(features, train_start, train_end, prediction_duration)

    # from sklearn.model_selection import GridSearchCV
    # param_grid = {'learning_rate': [0.1, 0.05, 0.01, 0.005],
    #               'max_depth': [10, 15, 20, 25, 30],
    #               'min_samples_leaf': [100, 250, 500, 750, 1000],
    #               'max_features': [1.0, 0.75, 0.5, 0.25, 0.1]
    #               }
    # est = GradientBoostingRegressor(n_estimators=125, verbose=100)
    # gs_cv = GridSearchCV(est, param_grid, n_jobs=7, verbose=100).fit(X_train, y_train)
    # print gs_cv.best_params_
    # exit(0)

    # regressor, mean_in_sample_score, mean_out_sample_score = model.fit_forest(X_train.iloc[0:].values,y_train.iloc[0:].values)
    # print 'mean_in_sample_score:',mean_in_sample_score
    # print 'mean_out_sample_score:',mean_out_sample_score
    # exit(0)

    if path.isfile(input_model_filename):
        print "Loading model/regressor from",input_model_filename
        with open(input_model_filename, 'r') as file:
            regressor = pickle.load(file)
        file.close()
    else:

        #regressor = RandomForestClassifier(n_estimators=75,min_samples_leaf=750,n_jobs=-1)

        # regressor = RandomForestRegressor(n_estimators=100,
        #                                   min_samples_leaf=500,
        #                                   random_state=42,
        #                                   n_jobs=-1)

        # regressor = GradientBoostingRegressor(n_estimators=250,
        #                           learning_rate=.01,
        #                           min_samples_leaf=500,
        #                           max_depth=20,
        #                           random_state=42, verbose=100)

        # regressor = GradientBoostingRegressor(n_estimators=500,
        #                           learning_rate=.001,
        #                           min_samples_leaf=1000,
        #                           max_depth=30,
        #                           random_state=42,
        #                           verbose=100)

        # regressor = GradientBoostingClassifier(n_estimators=50,verbose=100,min_samples_leaf=1000,max_depth=3)

        # regressor = xgb.XGBRegressor()

        regressor = xgb.XGBClassifier(n_estimators=50,max_depth=3,min_child_weight=5)

        print 'Training started at %s...'%(get_formatted_time_string(time())),
        regressor.fit(X_train.values, y_train.values)
        print 'Training done at %s'%(get_formatted_time_string(time()))

        # print '--Model features importance--'
        # model.get_feature_importances(regressor,cols)
        # print '----'

        if do_dump == 'true':
            with open(get_output_model_filename(features, train_start, train_end, prediction_duration), 'w+') as f:
                pickle.dump(regressor, f)
            f.close()

    print 'r-squared', regressor.score(X_test.values, y_test.values)

    trade(period, X_test.values, y_test.values, X_test.index, regressor, features, cols, split, split_percent, prediction_duration, threshold, threshold_percent, do_dump)

def trade(period, X, y, index, model, features, cols, split, split_percent, prediction_duration, threshold, threshold_percent, do_dump):
    '''
    Backtests a theoretical trading strategy
    '''
    preds = model.predict(X)/y_multiplied_by

    accurate_count = 0
    negative_trades = 0
    positive_trades = 0
    accurate_trades_count = 0
    for i, pred in enumerate(preds):
        if pred == y[i]:
            accurate_count = accurate_count+1
            if pred != 0:
                accurate_trades_count = accurate_trades_count+1
        if pred == -1:
            negative_trades = negative_trades+1
        elif pred == 1:
            positive_trades = positive_trades+1
    accurate_percent = float(accurate_count)/float(len(preds))*100
    print "Total predictions count =",len(preds)
    print "Accurate total predictions =",accurate_count
    print "Accurate total predictions percent =",(float(accurate_count)/float(len(preds))*100),"%"
    print "-1 trades =",negative_trades
    print "1 trades =",positive_trades
    print "Total trades =", (negative_trades+positive_trades)
    print "Accurate trades =",accurate_trades_count
    print "Accurate trades percent =",(float(accurate_trades_count)/float(negative_trades+positive_trades)*100),"%"
    return

    index_as_dates = []
    for value in index:
        index_as_dates.append(datetime.datetime.utcfromtimestamp(value))

    trades = np.zeros(len(preds))
    trade_at = 0
    active = False
    for i, pred in enumerate(preds):
        if active:
            if (index_as_dates[i]-trade_at).total_seconds() >= prediction_duration:
                trade_at = 0
                active = False
                # print 'Trade expired at', index_as_dates[i]
        elif abs(pred) >= threshold:
            active = True
            trades[i] = np.sign(pred)
            trade_at = index_as_dates[i]
            # print 'Trading at',trade_at

    returns = trades*y*100

    # if do_dump == 'true':
    #     with open(base_filename+".human_readable_results.txt", "w") as file:
    #         it = np.nditer(preds, flags=['f_index'])
    #         while not it.finished:
    #             file.write(
    #                 "%s (%s):\t%s\t%s\t%s\t%s\n" %
    #                 (index_as_dates[it.index].strftime(timestamp_format), index[it.index], format(preds[it.index],'.10f'), format(y[it.index],'.10f'), trades[it.index], returns[it.index]))
    #             it.iternext()
    #         file.close()

    trades_only = returns[trades != 0]
    if len(trades_only) == 0:
        print 'No trades were made.'
        return
    mean_return = trades_only.mean()
    accuracy = sum(trades_only > 0)*1./len(trades_only)
    profit = np.cumsum(returns)

    title_text = ('Trading at every {}% prediction. Position held for {} secs.'
              .format(threshold_percent, prediction_duration))
    return_text = 'Average Return: {:.4f} %'.format(mean_return)
    trades_text = 'Total Trades: {:d}'.format(len(trades_only))
    accuracy_text = 'Accuracy: {:.2f} %'.format(accuracy*100)
    print title_text
    print return_text
    print trades_text
    print accuracy_text

    #plt.figure(dpi=100000)
    fig, ax = plt.subplots()
    plt.plot(index_as_dates, profit)
    plt.title(title_text)
    plt.ylabel('Returns')
    plt.xticks(rotation=20)
    x_formatter = mtick.FormatStrFormatter('%.0f%%')
    ax.yaxis.set_major_formatter(x_formatter)
    y_formatter = mdates.DateFormatter("%Y-%m-%d %H:%M")
    ax.xaxis.set_major_formatter(y_formatter)
    plt.text(.05, .85, return_text, transform=ax.transAxes)
    plt.text(.05, .78, trades_text, transform=ax.transAxes)
    plt.text(.05, .71, accuracy_text, transform=ax.transAxes)
    if do_dump or do_parallel_search:
        plt.savefig(get_file_name(period, features, split_percent, prediction_duration, threshold_percent))
    if not do_parallel_search:
        plt.show()

def get_file_name(period,features,split_percent,prediction_duration,threshold_percent):
    #ToDo make this the same as the model output filename
    return '../data/strategy_period%s_features%s_split%s_duration%s_threshold%s.png' % (period, features, split_percent, prediction_duration, threshold_percent)

def parallel(params):
    period = params[0]
    data = params[1]
    features = params[2]
    cols = params[3]
    split = params[4]
    split_percent = params[5]
    prediction_duration = params[6]
    threshold = params[7]
    threshold_percent = params[8]

    print '-------'
    print 'Starting iteration. Period:', period, 'Features:', features, 'Split:', split_percent,'Prediction duration:', prediction_duration, 'Threshold:', threshold_percent
    image_filename = get_file_name(period,features,split_percent,prediction_duration,threshold_percent)
    if path.isfile(image_filename):
        print 'Image file already exists, skipping...'
        return

    fit_and_trade(period, data, features, cols, split, split_percent, prediction_duration, threshold, threshold_percent, False)


if __name__ == '__main__' and len(sys.argv) == 6:
    print "Starting at",get_formatted_time_string(time())

    filename = sys.argv[1]
    split_percent = int(sys.argv[2])
    prediction_duration = int(sys.argv[3])
    threshold_percent = float(sys.argv[4])
    do_dump = sys.argv[5]

    base_filename = '.'.join(filename.split('.')[:-1]) if '.' in filename else filename
    filename_extension = filename.split('.')[-1] if '.' in filename else filename

    threshold = threshold_percent/100

    print "Reading data from disk"
    if filename_extension == 'pkl':
        with open(filename, 'r') as file:
            data = pickle.load(file)
    elif filename_extension == 'tsv':
        data = pd.DataFrame.from_csv(filename, sep='\t')

    if do_dump == 'true' and filename_extension != 'tsv':
        data.to_csv(base_filename+'.tsv', sep='\t')

    if path.isfile(input_cols_filename):
        print "Loading cols from file",input_cols_filename
        with open(input_cols_filename, 'r') as file:
            cols = pickle.load(file)
        file.close()
    else:
        # Original features, work well when trained with books data that doesn't skip, and is fresh roughly every 1-2 seconds
        # cols = [
        #         'width',
        #         'imbalance2',
        #         'imbalance4',
        #         'imbalance8',
        #         'adj_price2',
        #         'adj_price4',
        #         'adj_price8',
        #         't30_count',
        #         't60_count',
        #         't120_count',
        #         't180_count',
        #         't30_av',
        #         't60_av',
        #         't120_av',
        #         't180_av',
        #         'agg30',
        #         'agg60',
        #         'agg120',
        #         'agg180',
        #         'trend30',
        #         'trend60',
        #         'trend120',
        #         'trend180'
        #         ]

        # Features with added windows
        cols = [
                'width',

                'imbalance2',
                'imbalance4',
                'imbalance8',

                'adj_price2',
                'adj_price4',
                'adj_price8',

                't10_count',
                't15_count',
                't30_count',
                't45_count',
                't60_count',
                't75_count',
                't90_count',
                't105_count',
                't120_count',
                't135_count',
                't150_count',
                't165_count',
                't180_count',

                't10_av',
                't15_av',
                't30_av',
                't45_av',
                't60_av',
                't75_av',
                't90_av',
                't105_av',
                't120_av',
                't135_av',
                't150_av',
                't165_av',
                't180_av',

                'agg10',
                'agg15',
                'agg30',
                'agg45',
                'agg60',
                'agg75',
                'agg90',
                'agg105',
                'agg120',
                'agg135',
                'agg150',
                'agg165',
                'agg180',

                'trend10',
                'trend15',
                'trend30',
                'trend45',
                'trend60',
                'trend75',
                'trend90',
                'trend105',
                'trend120',
                'trend135',
                'trend150',
                'trend165',
                'trend180',
                ]
        if do_dump == 'true':
            with open(output_cols_filename, 'w+') as f:
                pickle.dump(cols, f)
            f.close()

    data = data.fillna(0)
    data = data[data.width > 0]
    # ToDo if using scikit-learn
    #data[cols] = data[cols].astype('float32')

    # Search for optimal fits
    if do_parallel_search:
        columns = {}
        columns['OldAll'] = [
                'width',
                'imbalance2',
                'imbalance4',
                'imbalance8',
                'adj_price2',
                'adj_price4',
                'adj_price8',
                't30_count',
                't60_count',
                't120_count',
                't180_count',
                't30_av',
                't60_av',
                't120_av',
                't180_av',
                'agg30',
                'agg60',
                'agg120',
                'agg180',
                'trend30',
                'trend60',
                'trend120',
                'trend180'
                ]
        columns['NewAll'] = [
            'width',
            'imbalance2',
            'imbalance4',
            'imbalance8',
            'imbalance16',
            'imbalance32',
            'imbalance64',
            'adj_price2',
            'adj_price4',
            'adj_price8',
            'adj_price16',
            'adj_price32',
            'adj_price64',
            't30_count',
            't60_count',
            't120_count',
            't180_count',
            't90_count',
            't150_count',
            't210_count',
            't240_count',
            't270_count',
            't300_count',
            't330_count',
            't360_count',
            't390_count',
            't420_count',
            't450_count',
            't30_av',
            't60_av',
            't120_av',
            't180_av',
            't90_av',
            't150_av',
            't210_av',
            't240_av',
            't270_av',
            't300_av',
            't330_av',
            't360_av',
            't390_av',
            't420_av',
            't450_av',
            'agg30',
            'agg60',
            'agg120',
            'agg180',
            'agg90',
            'agg150',
            'agg210',
            'agg240',
            'agg270',
            'agg300',
            'agg330',
            'agg360',
            'agg390',
            'agg420',
            'agg450',
            'trend30',
            'trend60',
            'trend120',
            'trend180',
            'trend90',
            'trend150',
            'trend210',
            'trend240',
            'trend270',
            'trend300',
            'trend330',
            'trend360',
            'trend390',
            'trend420',
            'trend450'
            ]
        parallel_arguments = []
        for period in ['All']:
            if period == 'All':
                None
            for features in columns:
                cols = columns[features]
                for split_percent in [50]:
                    for prediction_duration in [30, 60, 90, 120]:
                        split = int(len(data)*(float(split_percent)/100))
                        parallel_arguments.append([period, data, features, cols, split, split_percent, prediction_duration, threshold, threshold_percent])
        print 'Trying', len(parallel_arguments), 'combinations on multiple threads'
        pool = multiprocessing.Pool(multiprocessing.cpu_count()-1) #Leave one CPU core free so system doesn't lock up
        pool.map(parallel, parallel_arguments)
        pool.close()
        pool.join()

    if not do_parallel_search:
        split = int(len(data)*(float(split_percent)/100))
        fit_and_trade('All', data, 'New', cols, split, split_percent, prediction_duration, threshold, threshold_percent, do_dump)

    print "Done at", get_formatted_time_string(time())