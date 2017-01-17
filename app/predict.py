#python predict.py <threshold>
#python predict.py 0.001
#python -W ignore -u predict.py 0.01 >> data/predict.out.txt 2>&1
import math
from model import features as f
import pymongo
import time
import sys
import pickle
import numpy as np
from math import log
import btcchina
import traceback
from datetime import datetime
import pandas as pd
from StringIO import StringIO
import subprocess

prediction_duration = 15
#prediction_periods = [5, 10, 15, 20, 25, 30]
prediction_periods = [15]
finish_within_one_x_of_duration = 1
periods = 3 # if 3, it means 2 retries and 1 final as market

header = "_id	width	mid	imbalance2	adj_price2	imbalance4	adj_price4	imbalance8	adj_price8	t10_count	t10_av	agg10	trend10	t15_count	t15_av	agg15	trend15	t30_count	t30_av	agg30	trend30	t45_count	t45_av	agg45	trend45	t60_count	t60_av	agg60	trend60	t75_count	t75_av	agg75	trend75	t90_count	t90_av	agg90	trend90	t105_count	t105_av	agg105	trend105	t120_count	t120_av	agg120	trend120	t135_count	t135_av	agg135	trend135	t150_count	t150_av	agg150	trend150	t165_count	t165_av	agg165	trend165	t180_count	t180_av	agg180	trend180"
y_multiplied_by = 100
#model_base_filename = "data/model_featuresNewAll_1476108655.82-1480731470.85_duration%s_times%s.pkl"
model_base_filename = "data/model_featuresNew_1482247313.0-1482762365.28_duration15_times100.pkl"
cols_filename = "data/cols_featuresNewAll.pkl"

client = pymongo.MongoClient()
db = client['cryptobot']
threshold = float(sys.argv[1])/100
# ticks_db = db['btcc_btccny_ticks']
timestamp_format = "%Y-%m-%d %H:%M:%S.%f"
access_key="" #ADD YOUR ACCESS KEY HERE
secret_key="" #ADD YOUR SECRET HERE
max_api_retry = 2

trade_api = btcchina.BTCChina(access_key,secret_key)

with open(cols_filename, 'r') as file:
    cols = pickle.load(file)
file.close()
models = {}
for prediction_period in prediction_periods:
    #model_filename = model_base_filename%(prediction_period,y_multiplied_by)
    model_filename = model_base_filename
    with open(model_filename, 'r') as file:
        models[prediction_period] = pickle.load(file)
    file.close()

def get_formatted_time_string(this_time):
    return datetime.utcfromtimestamp(this_time).strftime(timestamp_format)

def round_down(value, places):
    return math.floor(value * math.pow(10,places)) / math.pow(10,places)

def call_trade_api_with_retries(trade_api_function):
    response = {"error":{"message":"Didn't even try"}}
    for trial in xrange(max_api_retry):
        if trial > 0:
            print_formatted("!!! API request failed (%s). Retrying #%s ..."%(response['error'], trial))
        try:
            response = trade_api_function()
        except Exception as e:
            print_formatted("!!! Exception thrown when calling api: %s"%(str(e)))
            traceback.print_exc()
            sys.exc_clear()
        else:
            if (not isinstance(response, dict)) or (isinstance(response, dict) and 'error' not in response):
                break
            else:
                None #Todo send email on insufficient balance and when on max_api_retry
    return response

def get_live_balances():
    global balance_btc
    global balance_fiat
    response = call_trade_api_with_retries(lambda: trade_api.get_account_info())
    if 'balance' in response:
        balance_btc = float(response['balance']['btc']['amount'])
        balance_fiat = float(response['balance']['cny']['amount'])
    return balance_btc, balance_fiat

# def get_latest_tick():
#     tick = ticks_db.find().sort("$natural", -1).limit(1).next()
#     return tick

def get_current_time_seconds_utc():
    return (datetime.utcnow()-datetime(1970,1,1)).total_seconds()

def print_formatted(string, beginning_tabs_count=0, print_timestamp=True, timestamp=None, max_chars_per_line=125):
    max_chars_count = 0
    if not timestamp:
        timestamp = get_current_time_seconds_utc()
    formatted_string = ""
    timestamp_char_count = 0
    if print_timestamp:
        formatted_string = "%s (%s) - "%(get_formatted_time_string(timestamp),timestamp)
        timestamp_char_count = len(formatted_string)
    formatted_string = formatted_string + "\t"*beginning_tabs_count
    for c in string:
        if max_chars_count == max_chars_per_line:
            formatted_string = formatted_string + "\n" + " "*timestamp_char_count + "\t"*beginning_tabs_count
            max_chars_count = 0
        formatted_string = formatted_string + c
        max_chars_count+=1
    print formatted_string

def check_preds(preds):
    preds_as_list = []
    for key in preds:
        preds_as_list.append(preds[key])
    pred_signs = np.sign(preds_as_list)
    all_signs_equal = True
    sign = pred_signs[0]
    for pred_sign in pred_signs:
        if sign != pred_sign:
            all_signs_equal = False
            break
    all_preds_above_threshold = True
    for key in preds:
        if abs(preds[key]) < threshold:
            all_preds_above_threshold = False
            break
    if all_signs_equal and all_preds_above_threshold:
        return True
    else:
        return False

print_formatted("Running at prediction_duration=%s, threshold=%s ..."%(prediction_duration,threshold*100))

data = pd.DataFrame()
index = 0
last_data_timestamp = 0
position = 0
trade_time = 0
change = 0
previous_price = None
last_trade_midpoint_price = 0
current_order_id = 0
trades_count = 0
accurate_trades_count = 0
trade_position_periods_checked = {}
amount_trade_in_fiat = 0
amount_trade_in_btc = 0

response = call_trade_api_with_retries(lambda: trade_api.get_orders())
if isinstance(response, dict) and 'order' in response:
    orders = response['order']
    for order in orders:
        response = call_trade_api_with_retries(lambda: trade_api.cancel(order['id']))
        if response == True or response == 'true' or response == 'True':
            print_formatted("Order canceled on startup: %s"%(order))
#ToDO redistribute account 50/50, and do it periodically

balance_btc = 0
balance_fiat = 0
balance_btc, balance_fiat = get_live_balances()
balance_btc_initial = balance_btc
balance_fiat_initial = balance_fiat
if balance_btc == 0 or balance_fiat == 0:
    print_formatted('***Unable to get balances. Or one or both of balances (btc/fiat) are zero.')
    exit(0)
print_formatted("Starting balances: balance_btc=%s, balance_fiat=%s"%(balance_btc,balance_fiat))

while True:
    try:
        start = get_current_time_seconds_utc()

        if current_order_id == 0 and position == 0:
            line = subprocess.check_output(['tail', '-1', 'data/data_live.tsv'])
            data_as_tsv_str = header + "\n" + line
            data_as_tsv_StringIO = StringIO(data_as_tsv_str)
            data = pd.DataFrame.from_csv(data_as_tsv_StringIO, sep='\t')
            data = data.fillna(0)
            #ToDo if using scikit-learn
            #data[cols] = data[cols].astype('float32')

            preds = {}

            this_data_timestamp = data.index[index]
            if this_data_timestamp < (start - 3):
                # print "Data hasn't been updated in less than 3 seconds, skipping...",this_data_timestamp,start-3
                None
            elif last_data_timestamp != 0 and last_data_timestamp == this_data_timestamp:
                # print "Last data timestamp is equal to this data timestamp, skipping..."
                None
            else:
                last_data_timestamp = this_data_timestamp

                for prediction_period in prediction_periods:
                    preds[prediction_period] = models[prediction_period].predict(data[cols].values)[index] / y_multiplied_by

        else:
            data = f.get_book_df(1,live=True)
            data['width'], data['mid'] = f.get_width_and_mid(data)
            preds = {}
            # print "In a position or order, skipping features", data.index[index], data.width.iloc[index], data.mid.iloc[index]

        current_midpoint_price = data.mid.iloc[index]

        # If we can execute a new trade, and no pending orders
        if data.width.iloc[index] > 0 and current_order_id==0 and position == 0 and len(preds)>0 and check_preds(preds):
            position = np.sign(preds[prediction_duration])

            if position < 0:
                price = current_midpoint_price
                amount_trade_in_btc = balance_btc
                trade_type = 'ask'
            elif position > 0:
                price = current_midpoint_price
                amount_trade_in_fiat = balance_fiat
                amount_trade_in_btc = round_down(amount_trade_in_fiat / price, 4)
                trade_type = 'bid'
            trades_count+=1
            print_formatted("-----------Trade #%s----------"%(trades_count))
            trade_time = get_current_time_seconds_utc()
            print_string = "1) Trading %s at predictions: "%(position)
            for key in preds:
                print_string = print_string+"%s=%s "%(key,format(preds[key], '.10f'))
            print_formatted(print_string,timestamp=trade_time)
            print_formatted("Current midpoint price: %s" % (current_midpoint_price),1)
            if position < 0:
                print_formatted("Going from btc->fiat: selling %s btc, buying %s fiat." % (amount_trade_in_btc, amount_trade_in_fiat),2)
                print_formatted("+Hopefully buyback price will be lower than current sell price, for a profit",2)
                response = call_trade_api_with_retries(lambda: trade_api.sell(price, amount_trade_in_btc))
            elif position > 0:
                print_formatted("Going from fiat->btc: buying %s btc, selling %s fiat." % (amount_trade_in_btc, amount_trade_in_fiat),2)
                print_formatted("+Hopefully sellback price will be higher than current buy price, for a profit",2)
                response = call_trade_api_with_retries(lambda: trade_api.buy(price, amount_trade_in_btc))
            if isinstance(response, int):
                last_trade_midpoint_price = current_midpoint_price
                current_order_id = response
                trade_position_periods_checked = {}
                print_formatted("@@@*** Order successfully made, order ID: %s" % (current_order_id),1)
                print_formatted("Holding position for %s seconds"%(prediction_duration))
            else:
                print_formatted("@@@!!! Error making order: %s" % (str(response['error'] if isinstance(response, dict) and 'error' in response else response)), 1)
                position = 0
                last_trade_midpoint_price = 0
                amount_trade_in_fiat = 0
                amount_trade_in_btc = 0

        # If an open position has expired, and no pending orders
        if current_order_id==0 and position != 0 and (start - trade_time) >= prediction_duration+1:
            print_formatted("2) Position %s expired after %s secs. " % (position, (start-trade_time)))
            trade_time = get_current_time_seconds_utc()
            print_formatted("Current midpoint price: %s" % (current_midpoint_price),1)
            if position < 0:
                price = current_midpoint_price
                # amount_trade_in_fiat = balance_fiat-fiat_balance_before_btc_sale
                amount_trade_in_fiat = balance_fiat*0.50
                amount_trade_in_btc = round_down(amount_trade_in_fiat / price, 4)
                print_formatted("Going from fiat->btc: selling %s fiat, buying %s btc" % (amount_trade_in_fiat, amount_trade_in_btc),2)
                if price < last_trade_midpoint_price:
                    print_formatted("+Buyback price did go down, so should be profiting!",2)
                    accurate_trades_count = accurate_trades_count+1
                else:
                    print_formatted("-Buyback price didn't go down, should be a loss",2)
                response = call_trade_api_with_retries(lambda: trade_api.buy(price, amount_trade_in_btc))
            elif position > 0:
                price = current_midpoint_price
                # amount_trade_in_btc = amount_trade_in_btc
                amount_trade_in_btc = balance_btc*0.5
                amount_trade_in_fiat = round_down(amount_trade_in_btc * price, 2)
                print_formatted("Going from btc->fiat: buying %s fiat, selling %s btc" % (amount_trade_in_fiat, amount_trade_in_btc),2)
                if last_trade_midpoint_price < price:
                    print_formatted("+Sellback price did go up, so should be profiting!",2)
                    accurate_trades_count = accurate_trades_count + 1
                else:
                    print_formatted("-Sellback price didn't go up, should be a loss",2)
                response = call_trade_api_with_retries(lambda: trade_api.sell(price, amount_trade_in_btc))
            if isinstance(response, int):
                last_trade_midpoint_price = current_midpoint_price
                current_order_id = response
                trade_position_periods_checked = {}
                print_formatted("@@@*** Order successfully made, order ID: %s" % (current_order_id),1)
            else:
                print_formatted("@@@!!! Error making order: %s" % (str(response['error'] if isinstance(response, dict) and 'error' in response else response)), 1)
                last_trade_midpoint_price = 0
                amount_trade_in_fiat = 0
                amount_trade_in_btc = 0
            print_formatted("------------------------------")
            position = 0

        if current_order_id != 0:
            response = call_trade_api_with_retries(lambda: trade_api.get_orders(current_order_id))
            print_formatted("@@@ Pending order (ID %s) status: "%(current_order_id),1)
            print_formatted(str(response).replace("u\"","\"").replace("u\'","\'").replace("\'","\""),7,False,None,75)

            seconds_elapsed_since_trade = start - trade_time
            order_status = response['order']['status']
            order_type = response['order']['type']
            order_price = float(response['order']['price'])
            amount_unfilled = float(response['order']['amount'])
            amount_filled_total = amount_trade_in_btc - amount_unfilled

            if amount_unfilled != 0 or order_status == 'closed':
                current_order_id = 0
                balance_btc, balance_fiat = get_live_balances()
                print_formatted("+++ Order completed after %s s. Current balances: balance_btc=%s, balance_fiat=%s"%(seconds_elapsed_since_trade,balance_btc,balance_fiat), 1)
                if position == 0:
                    # ToDo % change from previous trade. Send email if going too low and kill
                    btc_percent_comparison = balance_btc / balance_btc_initial * 100
                    fiat_percent_comparison = balance_fiat / balance_fiat_initial * 100
                    total_percent_comparison = (btc_percent_comparison-100.0) + (fiat_percent_comparison-100.0)
                    print_formatted("+++ Current bals in %% comparison w/ initial: btc=%s%%, fiat=%s%%, total=%s%%" % (btc_percent_comparison, fiat_percent_comparison, total_percent_comparison), 1)
                    print_formatted("+++ Current accuracy percentage: %s%%"%((float(accurate_trades_count)/float(trades_count))*100), 1)
            else:
                period = 1
                period_duration = prediction_duration / finish_within_one_x_of_duration / periods
                for key in trade_position_periods_checked:
                    if key >= period:
                        period = key+1
                this_period_duration = period_duration * period
                if seconds_elapsed_since_trade >= this_period_duration and period not in trade_position_periods_checked:
                    if order_status != 'cancelled':
                        print_formatted("@@@*** Position %s order still not filled at period %s after %s seconds" % (position, period, seconds_elapsed_since_trade), 1)
                        response = call_trade_api_with_retries(lambda: trade_api.cancel(current_order_id))
                        print_formatted("@@@*** Order canceled.", 1)
                    else:
                        trade_position_periods_checked[period] = True
                        current_order_id = 0

                        if period < periods and seconds_elapsed_since_trade < prediction_duration:
                            amount = amount_unfilled
                            print_formatted("@@@*** Retry #%s. Will remake order as %s at current midpoint price %s for unfilled btc amount: %s" % (period, order_type, current_midpoint_price, amount), 1)
                            if order_type == 'bid':
                                if last_trade_midpoint_price != current_midpoint_price and position != 0:
                                    #Recalculate amount based on new price. Only necessary for initial (position take) orders which leave nothing left in fiat balance
                                    balance_btc, balance_fiat = get_live_balances()
                                    balance_fiat_remaining = balance_fiat
                                    amount = round_down(balance_fiat_remaining / current_midpoint_price, 4)
                                    amount_trade_in_btc = amount_filled_total + amount
                                    print_formatted("@@@*** Price changed for initial buy order. Recalc w/ remaining fiat bal: %s, unfilled btc amount: %s"%(balance_fiat_remaining, amount),1)
                                    print_formatted("@@@*** Total trade in btc amount adjusted to: %s" % (amount_trade_in_btc), 1)
                                response = call_trade_api_with_retries(lambda: trade_api.buy(current_midpoint_price, amount))
                            elif order_type == 'ask':
                                response = call_trade_api_with_retries(lambda: trade_api.sell(current_midpoint_price, amount))
                            if isinstance(response, int):
                                current_order_id = response
                                last_trade_midpoint_price = current_midpoint_price
                                print_formatted("@@@*** Order successfully made, order ID: %s" % (current_order_id), 1)
                            else:
                                print_formatted("@@@!!! Error making order: %s" % (str(response['error'] if isinstance(response,dict) and 'error' in response else response)),1)

                        else:
                            print_formatted("@@@*** Final period reached. Filled btc amount: %s. Unfilled btc amount: %s"%(amount_filled_total, amount_unfilled),1)

                            if position!=0 and amount_filled_total==0: #Initial (position take) trade not filled at all, all good
                                balance_btc, balance_fiat = get_live_balances()
                                print_formatted("@@@*** Position take order not filled at all, moving on. Current balances: balance_btc=%s, balance_fiat=%s"%(balance_btc,balance_fiat), 1)
                                position = 0
                                trades_count = trades_count - 1

                            else:
                                if position != 0: #Initial (position take) trade
                                    if (amount_filled_total / amount_trade_in_btc) < 0.5: #Less than 50% filled, so reverse it!
                                        reverse = True
                                        amount = amount_filled_total #Reverse the filled amount
                                        if order_type == 'bid':
                                            this_order_type = 'ask'
                                        elif order_type == 'ask':
                                            this_order_type = 'bid'
                                    else: #More than 50% filled, so forge ahead
                                        amount = amount_unfilled
                                        reverse = False
                                        this_order_type = order_type
                                elif position == 0:
                                    amount = amount_unfilled
                                    reverse = False
                                    this_order_type = order_type

                                if not reverse:
                                    print_formatted("@@@*** Will fill remainder as %s at market, the unfilled btc amount: %s" % (this_order_type, amount), 2)
                                elif reverse:
                                    trades_count = trades_count - 1
                                    print_formatted("@@@*** Position take order filled only < 50 percent. Canceling position / reversing trade as %s at market. Reversing filled btc amount: %s" % (this_order_type, amount), 2)
                                if this_order_type == 'bid':
                                    response = call_trade_api_with_retries(lambda: trade_api.buy(None, amount))
                                elif this_order_type == 'ask':
                                    response = call_trade_api_with_retries(lambda: trade_api.sell(None, amount))
                                if isinstance(response, int):
                                    current_order_id = response
                                    print_formatted("@@@*** Order successfully made, order ID: %s" % (current_order_id),1)
                                else:
                                    print_formatted("@@@!!! Error making order: %s" % (str(response['error'] if isinstance(response,dict) and 'error' in response else response)),1)
                                if position != 0 and reverse:
                                    position = 0

        time_delta = get_current_time_seconds_utc()-start
        if time_delta < 1.0:
            time.sleep(1-time_delta)

    except Exception as e:
        print_formatted("!!! Exception: %s"%str(e))
        traceback.print_exc()
        sys.exc_clear()
