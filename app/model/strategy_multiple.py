#python strategy_multiple.gy <data.pkl> <test after timestamp> <threshold percent>
#python strategy_multiple.py ../data/data.tsv 1479943156.27 0.01

import datetime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import pickle
import sys
import pandas as pd
from time import time
import xgboost as xgb

prediction_duration = 30
prediction_periods = [5, 10, 15, 30, 35, 40, 45]

timestamp_format = "%Y-%m-%d %H:%M:%S.%f"
input_cols_filename = "../data/cols_featuresNewAll.pkl"
model_base_filename = "../data/model_featuresNewAll_1476108655.82-1479943156.27_duration%s_times%s.pkl"
y_multiplied_by = 100
models = {}
for prediction_period in prediction_periods:
    model_filename = model_base_filename%(prediction_period,y_multiplied_by)
    with open(model_filename, 'r') as file:
        models[prediction_period] = pickle.load(file)
    file.close()

def get_formatted_time_string(this_time):
    return datetime.datetime.utcfromtimestamp(this_time).strftime(timestamp_format)

def fit_and_trade(data, cols, threshold, threshold_percent):
    '''
    Fits and backtests a theoretical trading strategy
    '''
    X = data[cols]
    y = {}
    for prediction_period in prediction_periods:
        y[prediction_period] = data["mid%s"%prediction_period].values

    print 'Test data starts at %s (%s), ends at %s (%s)'%(get_formatted_time_string(X.index.values[0]),X.index.values[0],get_formatted_time_string(X.index.values[-1]),X.index.values[-1])

    trade(X.values, y, X.index, threshold, threshold_percent)

def check_preds(i, preds, threshold):
    preds_as_list = []
    for key in preds:
        preds_as_list.append(preds[key][i])
    pred_signs = np.sign(preds_as_list)
    all_signs_equal = True
    sign = pred_signs[0]
    for pred_sign in pred_signs:
        if sign != pred_sign:
            all_signs_equal = False
            break
    all_preds_above_threshold = True
    for key in preds:
        if abs(preds[key][i]) < threshold:
            all_preds_above_threshold = False
            break
    if all_signs_equal and all_preds_above_threshold:
        return True
    else:
        return False

def trade(X, y, index, threshold, threshold_percent):
    '''
    Backtests a theoretical trading strategy
    '''
    preds = {}
    for prediction_period in prediction_periods:
        preds[prediction_period] = models[prediction_period].predict(X) / y_multiplied_by

    index_as_dates = []
    for value in index:
        index_as_dates.append(datetime.datetime.utcfromtimestamp(value))

    trades = np.zeros(len(preds[prediction_duration]))
    trade_at = 0
    active = False
    for i in xrange(len(index)):
        if active:
            if (index_as_dates[i]-trade_at).total_seconds() >= prediction_duration:
                trade_at = 0
                active = False
        elif check_preds(i,preds,threshold):
            active = True
            trades[i] = np.sign(preds[prediction_duration][i])
            trade_at = index_as_dates[i]
            print_string = "%s %s Trading %s at predictions: "%(i,index[i],trades[i])
            for key in preds:
                print_string = print_string+ "%s=%s "%(key,format(preds[key][i], '.10f'))
            print print_string
            print_string = "%s %s                 Real mids: "%(i,index[i])
            for key in preds:
                print_string = print_string + "%s=%s "%(key,format(y[key][i], '.10f'))
            print print_string
            print ""


    returns = trades*y[prediction_duration]*100

    for i in xrange(len(trades)):
        if trades[i] != 0 and returns[i]>0:
            #Check that real future mid (y[x]), x secs later, is in the same direction as pred[x], for x in 5,10,15,30,35,40,45
            this_timestamp = index[i]
            all_predictions_correct = True
            for x in prediction_periods:
                for check_timestamp_index in xrange(i,len(index)):
                    if index[check_timestamp_index]-this_timestamp >= x:
                        print "For trade made at timestamp",this_timestamp,"period of",x,"is found at",index[check_timestamp_index],"with prediction=",format(preds[x][check_timestamp_index],'.10f'),"real=",format(y[x][check_timestamp_index],'.10f')
                        if(np.sign(y[x][check_timestamp_index]) != np.sign(preds[x][check_timestamp_index])):
                            all_predictions_correct = False
                        break
                if not all_predictions_correct:
                    break
            if not all_predictions_correct:
                print "***A prediction for this trade was not correct, reversing return"
                returns[i] = (returns[i]*-1.0)
            else:
                print "+++All predictions were correct! This is a definite positive return"

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
    plt.show()

if __name__ == '__main__':
    print "Starting at",get_formatted_time_string(time())

    filename = sys.argv[1]
    test_after_timestamp = float(sys.argv[2])
    threshold_percent = float(sys.argv[3])

    base_filename = '.'.join(filename.split('.')[:-1]) if '.' in filename else filename
    filename_extension = filename.split('.')[-1] if '.' in filename else filename

    threshold = threshold_percent/100

    print "Reading data from disk"
    if filename_extension == 'pkl':
        with open(filename, 'r') as file:
            data = pickle.load(file)
    elif filename_extension == 'tsv':
        data = pd.DataFrame.from_csv(filename, sep='\t')

    data = data.fillna(0)
    data = data[data.width > 0]
    # ToDo if using scikit-learn
    #data[cols] = data[cols].astype('float32')

    data = data[data.index >= test_after_timestamp]

    print "Loading cols from file",input_cols_filename
    with open(input_cols_filename, 'r') as file:
        cols = pickle.load(file)
    file.close()

    fit_and_trade(data, cols, threshold, threshold_percent)

    print "Done at", get_formatted_time_string(time())