import sys
from pyspark import SparkContext
import csv
import json
import numpy as np
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import time


def get_stats():
    with open(val_file) as in_file:
    	ground_truth  = in_file.readlines()[1:]

    with open(output_file) as in_file:
    	prediction = in_file.readlines()[1:]

    error_dist = {'0-1': 0, '1-2': 0, '2-3': 0, '3-4': 0, '4-5': 0}

    rmse = 0
    for pred in range(len(prediction)):
    	abs_diff = abs(float(prediction[pred].split(',')[2]) - float(ground_truth[pred].split(',')[2]))
    	rmse += abs_diff ** 2
    	if abs_diff < 1:
    		error_dist['0-1'] += 1
    	elif 2 > abs_diff >= 1:
    		error_dist['1-2'] += 1
    	elif 3 > abs_diff >= 2:
    		error_dist['2-3'] += 1
    	elif 4 > abs_diff >= 3:
    		error_dist['3-4'] += 1
    	else:
    		error_dist['4-5'] += 1

    print('Error Distribution:')
    for key in error_dist:
    	print(key, ':',error_dist[key])

    rmse = (rmse / len(prediction)) ** (1/2)
    print('\nRMSE:')
    print(rmse, '\n')


def create_features():
	review_rdd = sc.textFile(folder_path + '/review_train.json')
	user_rdd = sc.textFile(folder_path + '/user.json')
	business_rdd = sc.textFile(folder_path + '/business.json')
	checkin_rdd = sc.textFile(folder_path + '/checkin.json')

	average_dict = {}

	review_dict = review_rdd.map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], (float(x['useful']), float(x['funny']), float(x['cool'])))).groupByKey().mapValues(get_average).collectAsMap()

	user_dict = user_rdd.map(lambda x: json.loads(x)).map(lambda x: (x['user_id'], (float(x['average_stars']), float(x['review_count']), float(x['fans']), float(x['useful']), float(x['compliment_note']), float(x['compliment_hot']), float(x['compliment_more']), float(x['compliment_profile']), float(x['compliment_cute']), float(x['compliment_list']), float(x['compliment_plain']), float(x['compliment_cool']), float(x['compliment_funny']), float(x['compliment_writer']), float(x['compliment_photos'])))).collectAsMap()

	business_dict = business_rdd.map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], (float(x['stars']), float(x['review_count']), float(x['is_open']), float(x['latitude']) if ('latitude' in x and x['latitude'] != None) else 0, float(x['longitude']) if ('longitude' in x and x['longitude'] != None) else 0, 1 if ('attributes' in x and x['attributes'] != None and 'WiFi' in x['attributes'] and x['attributes']['WiFi'] != None and x['attributes']['WiFi'] == 'no') else 0, 0 if ('attributes' in x and x['attributes'] != None and 'WiFi' in x['attributes'] and x['attributes']['WiFi'] != None and (x['attributes']['WiFi'] == 'no' or x['attributes']['WiFi'] == 'paid' or x['attributes']['WiFi'] == 'free')) else 1, 1 if ('attributes' in x and x['attributes'] != None and 'GoodForKids' in x['attributes'] and (x['attributes']['GoodForKids'] == 'True' or x['attributes']['GoodForKids'] is True)) else 0, 1 if ('attributes' in x and x['attributes'] != None and 'RestaurantsGoodForGroups' in x['attributes'] and (x['attributes']['RestaurantsGoodForGroups'] == 'True' or x['attributes']['RestaurantsGoodForGroups'] is True)) else 0))).collectAsMap()

	days_open_dict = business_rdd.map(lambda x: json.loads(x)).filter(lambda x: ('hours' in x) and (x['hours'])).map(lambda x: (x['business_id'], len(x['hours']))).collectAsMap()
	average_dict['days_open'] = sum([days for days in days_open_dict.values()]) / len(days_open_dict)

	checkin_dict = checkin_rdd.map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], len(x['time'].values()))).collectAsMap()
	average_dict['checkin'] = sum([checkin for checkin in checkin_dict.values()]) / len(days_open_dict)

	return review_dict, user_dict, business_dict, average_dict, days_open_dict, checkin_dict


def get_features(x, review_dict, user_dict, business_dict, average_dict, days_open_dict, checkin_dict):
	user_id = x[0]
	business_id = x[1]

	default_feature = (None, None, None)

	if business_id in review_dict:
		review_feature = review_dict[business_id]
	else:
		review_feature = default_feature

	if user_id in user_dict:
		user_feature = user_dict[user_id]
	else:
		user_feature = default_feature

	if business_id in business_dict:
		business_feature = business_dict[business_id]
	else:
		business_feature = default_feature

	if business_id in days_open_dict:
		days_open_feature = days_open_dict[business_id]
	else:
		days_open_feature = average_dict['days_open']

	if business_id in checkin_dict:
		checkin_feature = checkin_dict[business_id]
	else:
		checkin_feature = average_dict['checkin']

	return [*review_feature, *user_feature, *business_feature, checkin_feature]


def get_average(rating):
	useful = 0
	funny = 0
	cool = 0

	for u, f, c in rating:
		useful += u
		funny += f
		cool += c

	n = len(rating)
	if n:
		return (useful / n, funny / n, cool / n)
	else:
		return (None, None, None)


def get_xgb_output(train_x, train_y, val_x):
	param = {
        'lambda': 9.92724463758443,
        'alpha': 0.2765119705933928,
        'colsample_bytree': 0.5,
        'subsample': 0.8,
        'learning_rate': 0.02,
        'max_depth': 17,
        'random_state': 2020,
        'min_child_weight': 101,
        'n_estimators': 300,
    	}

	xgb = XGBRegressor(**param)
	xgb.fit(train_x, train_y)
	val_y = xgb.predict(val_x)

	return val_y


def get_catboost_output(train_x, train_y, val_x):
	param = {
		'random_state': 1, 
		'learning_rate': 0.05, 
		'n_estimators': 1000, 
		'max_depth': 10, 
		'max_bin': 256, 
		'verbose': 0, 
		'colsample_bylevel': 0.8
		}

	catboost = CatBoostRegressor(**param)
	catboost.fit(train_x, train_y)
	val_y = catboost.predict(val_x)

	return val_y


def model_based_collaborative_filtering(yelp_train, yelp_val):
	review_dict, user_dict, business_dict, average_dict, days_open_dict, checkin_dict = create_features()
	
	train_x = np.array([get_features(x, review_dict, user_dict, business_dict, average_dict, days_open_dict, checkin_dict) for x in yelp_train], dtype = 'float32')
	train_y = np.array([float(x[2]) for x in yelp_train], dtype = 'float32')
	val_x = np.array([get_features(x, review_dict, user_dict, business_dict, average_dict, days_open_dict, checkin_dict) for x in yelp_val], dtype = 'float32')

	xgb_output = get_xgb_output(train_x, train_y, val_x)
	catboost_output = get_catboost_output(train_x, train_y, val_x)

	xgb_weight = 0.5
	catboost_weight = 0.5
	model_cf_output = []
	for xgb, catboost in zip(xgb_output, catboost_output):
		model_cf_output.append(xgb_weight * xgb + catboost_weight * catboost)

	return model_cf_output


folder_path = sys.argv[1]
val_file = sys.argv[2]
output_file = sys.argv[3]


begin = time.time()


sc = SparkContext('local[*]', 'competition')
sc.setLogLevel('WARN')


yelp_train = sc.textFile(folder_path + '/yelp_train.csv').zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0].split(',')).collect()
yelp_val = sc.textFile(val_file).zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0].split(',')).collect()


model_cf_output = model_based_collaborative_filtering(yelp_train, yelp_val)


output = []
for ids, pred in zip(yelp_val, model_cf_output):
	if pred < 1:
		pred = 1.0
	elif pred > 5:
		pred = 5.0

	temp_output = []
	temp_output.append(ids[0])
	temp_output.append(ids[1])
	temp_output.append(pred)
	output.append(temp_output)


with open(output_file, 'w') as out_file:
	writer = csv.writer(out_file)
	writer.writerows([['user_id', 'business_id', 'prediction']])
	writer.writerows(output)


get_stats()
end = time.time()
print('Execution Time:')
print(round(end - begin, 2), 's')