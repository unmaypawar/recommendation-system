'''
Method Description:
After experimenting with different techniques, I have chosen to use weighted hybrid recommendation system to improve RMSE from what I achieved in homework 3.
Feature engineering is important as a lot of data is available but choosing right attributes is crucial.
From 'user.json', I have used attributes about user by use of binary encoding or directly used their numerical values.
From 'business.json', majority of features were binary encoded and some of them were directly used.
From 'review_train.json', total of 'useful' , 'funny' and 'cool' for each business were used.
From 'checkin.json', I have combined checkins recorded for each of the business and their aggregate was used.
From 'photo.json', number of photos for each business was used.
From 'tip.json', likes of tips for each business was used.
All attributes were used in generating a heatmap of correlation between features and ratings.
Each of the attribute was added one by one to check if it was actually improving RMSE on the validation data provided. 
Final attributes were selected based on whether they passed the above conditions or not.
I chose XGBoost and CatBoost as two regressor models for my recommendation system.
Hyperparameter tuning was done by using Optuna and manual tweaking.
I combined output of XGBoost and CatBoost using linear formula and choosing the individual weight by manual testing.
P.S. Code for hyperparameter tuning is at the end and commented out because of time constraints and also optuna isn't available on vocareum.

Error Distribution:
0-1 : 102408
1-2 : 32735
2-3 : 6110
3-4 : 791
4-5 : 0

RMSE:
0.9757425782700765 

Execution Time:
395.0 s
'''


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


def get_total(rating):
    useful = 0
    funny = 0
    cool = 0

    for u, f, c in rating:
        useful += u
        funny += f
        cool += c

    return (useful, funny, cool)


def create_business_features(business_data):
    if 'latitude' in business_data and business_data['latitude'] != None:
        latitude = float(business_data['latitude'])
    else:
        latitude = 0

    if 'longitude' in business_data and business_data['longitude'] != None:
        longitude = float(business_data['longitude'])
    else:
        longitude = 0

    if 'stars' in business_data and business_data['stars'] != None:
        stars = float(business_data['stars'])
    else:
        stars = 0

    if 'review_count' in business_data and business_data['review_count'] != None:
        review_count = float(business_data['review_count'])
    else:
        review_count = 0

    if 'is_open' in business_data and business_data['is_open'] != None:
        is_open = float(business_data['is_open'])
    else:
        is_open = 0

    if 'categories' in business_data and business_data['categories'] != None:
        categories = int(len(business_data['categories']))
    else:
        categories = 0

    if 'attributes' in business_data and business_data['attributes'] != None:
        if 'GoodForKids' in business_data['attributes'] and business_data['attributes']['GoodForKids'] != None:
            GoodForKids = 1 if (business_data['attributes']['GoodForKids'] == 'True' or business_data['attributes']['GoodForKids'] is True) else 0
        else:
            GoodForKids = 0        

        if 'RestaurantsGoodForGroups' in business_data['attributes'] and business_data['attributes']['RestaurantsGoodForGroups'] != None:
            RestaurantsGoodForGroups = 1 if (business_data['attributes']['RestaurantsGoodForGroups'] == 'True' or business_data['attributes']['RestaurantsGoodForGroups'] is True) else 0
        else:
            RestaurantsGoodForGroups = 0

        if 'BusinessAcceptsCreditCards' in business_data['attributes'] and business_data['attributes']['BusinessAcceptsCreditCards'] != None:
            BusinessAcceptsCreditCards = 1 if (business_data['attributes']['BusinessAcceptsCreditCards'] == 'True' or business_data['attributes']['BusinessAcceptsCreditCards'] is True) else 0
        else:
            BusinessAcceptsCreditCards = 0

        if 'BikeParking' in business_data['attributes'] and business_data['attributes']['BikeParking'] != None:
            BikeParking = 1 if (business_data['attributes']['BikeParking'] == 'True' or business_data['attributes']['BikeParking'] is True) == True else 0
        else:
            BikeParking = 0

        if 'OutdoorSeating' in business_data['attributes'] and business_data['attributes']['OutdoorSeating'] != None:
            OutdoorSeating = 1 if (business_data['attributes']['OutdoorSeating'] == 'True' or business_data['attributes']['OutdoorSeating'] is True) else 0
        else:
            OutdoorSeating = 0

        if 'RestaurantsDelivery' in business_data['attributes'] and business_data['attributes']['RestaurantsDelivery'] != None:
            RestaurantsDelivery = 1 if (business_data['attributes']['RestaurantsDelivery'] == 'True' or business_data['attributes']['RestaurantsDelivery'] is True) else 0
        else:
            RestaurantsDelivery = 0

        if 'Caters' in business_data['attributes'] and business_data['attributes']['Caters'] != None:
            Caters = 1 if (business_data['attributes']['Caters'] == 'True' or business_data['attributes']['Caters'] is True) else 0
        else:
            Caters = 0

        if 'HasTV' in business_data['attributes'] and business_data['attributes']['HasTV'] != None:
            HasTV = 1 if (business_data['attributes']['HasTV'] == 'True' or business_data['attributes']['HasTV'] is True) else 0
        else:
            HasTV = 0

        if 'RestaurantsReservations' in business_data['attributes'] and business_data['attributes']['RestaurantsReservations'] != None:
            RestaurantsReservations = 1 if (business_data['attributes']['RestaurantsReservations'] == 'True' or business_data['attributes']['RestaurantsReservations'] is True) else 0
        else:
            RestaurantsReservations = 0

        if 'RestaurantsTableService' in business_data['attributes'] and business_data['attributes']['RestaurantsTableService'] != None:
            RestaurantsTableService = 1 if (business_data['attributes']['RestaurantsTableService'] == 'True' or business_data['attributes']['RestaurantsTableService'] is True) else 0
        else:
            RestaurantsTableService = 0

        if 'ByAppointmentOnly' in business_data['attributes'] and business_data['attributes']['ByAppointmentOnly'] != None:
            ByAppointmentOnly = 1 if (business_data['attributes']['ByAppointmentOnly'] == 'True' or business_data['attributes']['ByAppointmentOnly'] is True) else 0
        else:
            ByAppointmentOnly = 0

        if 'RestaurantsTakeOut' in business_data['attributes'] and business_data['attributes']['RestaurantsTakeOut'] != None:
            RestaurantsTakeOut = 1 if (business_data['attributes']['RestaurantsTakeOut'] == 'True' or business_data['attributes']['RestaurantsTakeOut'] is True) else 0
        else:
            RestaurantsTakeOut = 0

        if 'AcceptsInsurance' in business_data['attributes'] and business_data['attributes']['AcceptsInsurance'] != None:
            AcceptsInsurance = 1 if (business_data['attributes']['AcceptsInsurance'] == 'True' or business_data['attributes']['AcceptsInsurance'] is True) else 0
        else:
            AcceptsInsurance = 0

        if 'WheelchairAccessible' in business_data['attributes'] and business_data['attributes']['WheelchairAccessible'] != None:
            WheelchairAccessible = 1 if (business_data['attributes']['WheelchairAccessible'] == 'True' or business_data['attributes']['WheelchairAccessible'] is True) else 0
        else:
            WheelchairAccessible = 0

        if 'RestaurantsPriceRange2' in business_data['attributes'] and business_data['attributes']['RestaurantsPriceRange2'] != None:
            RestaurantsPriceRange2 = float(business_data['attributes']['RestaurantsPriceRange2'])
        else:
            RestaurantsPriceRange2 = 0

        if 'WiFi' in business_data['attributes'] and business_data['attributes']['WiFi'] != None:
            no_wifi = 1 if business_data['attributes']['WiFi'] == 'no' else 0
        else:
            no_wifi = 0

        if 'WiFi' in business_data['attributes'] and business_data['attributes']['WiFi'] != None:
            no_wifi_info = 0 if business_data['attributes']['WiFi'] == 'no' or  business_data['attributes']['WiFi'] == 'free' or business_data['attributes']['WiFi'] == 'paid' else 1
        else:
            no_wifi_info = 1

        if 'Ambience' in business_data['attributes'] and business_data['attributes']['Ambience'] != None:
            Ambience_dict = json.loads(business_data['attributes']['Ambience'].replace('\'', '"').replace('False', '"False"').replace('True', '"True"'))
                                          
            if 'romantic' in Ambience_dict and (Ambience_dict['romantic'] == 'True' or Ambience_dict['romantic'] is True):
                romantic = 1
            else:
                romantic = 0

            if 'intimate' in Ambience_dict and (Ambience_dict['intimate'] == 'True' or Ambience_dict['intimate'] is True):
                intimate = 1
            else:
                intimate = 0

            if 'classy' in Ambience_dict and (Ambience_dict['classy'] == 'True' or Ambience_dict['classy'] is True):
                classy = 1
            else:
                classy = 0

            if 'casual' in Ambience_dict and (Ambience_dict['casual'] == 'True' or Ambience_dict['casual'] is True):
                casual = 1
            else:
                casual = 0

            if 'hipster' in Ambience_dict and (Ambience_dict['hipster'] == 'True' or Ambience_dict['hipster'] is True):
                hipster = 1
            else:
                hipster = 0

            if 'divey' in Ambience_dict and (Ambience_dict['divey'] == 'True' or Ambience_dict['divey'] is True):
                divey = 1
            else:
                divey = 0

            if 'touristy' in Ambience_dict and (Ambience_dict['touristy'] == 'True' or Ambience_dict['touristy'] is True):
                touristy = 1
            else:
                touristy = 0

            if 'trendy' in Ambience_dict and (Ambience_dict['trendy'] == 'True' or Ambience_dict['trendy'] is True):
                trendy = 1
            else:
                trendy = 0

            if 'upscale' in Ambience_dict and (Ambience_dict['upscale'] == 'True' or Ambience_dict['upscale'] is True):
                upscale = 1
            else:
                upscale = 0

            ambience = romantic + intimate + classy + casual + hipster + divey + touristy + trendy + upscale
        else:
            romantic = 0
            intimate = 0
            classy = 0
            casual = 0
            hipster = 0
            divey = 0
            touristy = 0
            trendy = 0
            upscale = 0
            ambience = 0

    else:
        GoodForKids = 0
        RestaurantsGoodForGroups = 0
        BusinessAcceptsCreditCards = 0
        BikeParking = 0
        OutdoorSeating = 0
        RestaurantsDelivery = 0
        Caters = 0
        HasTV = 0
        RestaurantsReservations = 0
        RestaurantsTableService = 0
        ByAppointmentOnly = 0
        RestaurantsTakeOut = 0
        AcceptsInsurance = 0
        WheelchairAccessible = 0
        RestaurantsPriceRange2 = 0
        no_wifi = 0
        no_wifi_info = 1
        romantic = 0
        intimate = 0
        classy = 0
        casual = 0
        hipster = 0
        divey = 0
        touristy = 0
        trendy = 0
        upscale = 0
        ambience = 0

    business_feature = (business_data['business_id'], (latitude, longitude, stars, review_count, is_open, GoodForKids, RestaurantsGoodForGroups, BusinessAcceptsCreditCards, OutdoorSeating, HasTV, RestaurantsTableService, RestaurantsTakeOut, AcceptsInsurance, WheelchairAccessible, RestaurantsPriceRange2, no_wifi, no_wifi_info))

    return business_feature


def create_features():
    review_rdd = sc.textFile(folder_path + '/review_train.json')
    user_rdd = sc.textFile(folder_path + '/user.json')
    business_rdd = sc.textFile(folder_path + '/business.json')
    checkin_rdd = sc.textFile(folder_path + '/checkin.json')
    photo_rdd = sc.textFile(folder_path + '/photo.json')
    tip_rdd = sc.textFile(folder_path + '/tip.json')

    average_dict = {}

    review_dict = review_rdd.map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], (float(x['useful']), float(x['funny']), float(x['cool'])))).groupByKey().mapValues(get_total).collectAsMap()

    user_dict = user_rdd.map(lambda x: json.loads(x)).map(lambda x: (x['user_id'], (float(x['average_stars']), float(x['review_count']), float(x['fans']), float(x['useful']), float(x['compliment_note']), float(x['compliment_hot']), float(x['compliment_more']), float(x['compliment_profile']), float(x['compliment_cute']), float(x['compliment_list']), float(x['compliment_plain']), float(x['compliment_cool']), float(x['compliment_funny']), float(x['compliment_writer']), float(x['compliment_photos'])))).collectAsMap()

    friends_dict = user_rdd.map(lambda x: json.loads(x)).map(lambda x: (x['user_id'], x['friends'].split(', ')) if x['friends'] else (x['user_id']['None'])).map(lambda x: (x[0], 0) if x[1] == ['None'] else (x[0], len(x[1]))).collectAsMap()
    average_dict['friends'] = sum([friend for friend in friends_dict.values()]) / len(friends_dict)

    yelping_since_dict = user_rdd.map(lambda x: json.loads(x)).map(lambda x: (x['user_id'], int((x['yelping_since'].split('-')[0])))).collectAsMap()
    
    business_dict = business_rdd.map(lambda x: json.loads(x)).map(create_business_features).collectAsMap()
    
    days_open_dict = business_rdd.map(lambda x: json.loads(x)).filter(lambda x: ('hours' in x) and (x['hours'])).map(lambda x: (x['business_id'], len(x['hours']))).collectAsMap()
    average_dict['days_open'] = sum([days for days in days_open_dict.values()]) / len(days_open_dict)

    checkin_dict = checkin_rdd.map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], len(x['time'].values()))).collectAsMap()
    average_dict['checkin'] = sum([checkin for checkin in checkin_dict.values()]) / len(days_open_dict)

    photo_dict = photo_rdd.map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x, y: x + y).collectAsMap()
    average_dict['photo'] = sum([photo for photo in photo_dict.values()]) / len(photo_dict)

    tip_dict = tip_rdd.map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], x['likes'])).reduceByKey(lambda x, y: x + y).collectAsMap()
    average_dict['tip'] = sum([tip for tip in tip_dict.values()]) / len(tip_dict)

    return review_dict, user_dict, friends_dict, average_dict, yelping_since_dict, business_dict, days_open_dict, checkin_dict, photo_dict, tip_dict


def get_features(x, review_dict, user_dict, friends_dict, average_dict, yelping_since_dict, business_dict, days_open_dict, checkin_dict, photo_dict, tip_dict):
    user_id = x[0]
    business_id = x[1]

    if business_id in review_dict:
        review_feature = review_dict[business_id]
    else:
        review_feature = (0,) * 3

    if user_id in user_dict:
        user_feature = user_dict[user_id]
    else:
        user_feature = (0,) * 15

    if user_id in friends_dict:
        friends_feature = friends_dict[user_id]
    else:
        friends_feature = average_dict['friends']

    if user_id in yelping_since_dict:
        yelping_since_feature = yelping_since_dict[user_id]
    else:
        yelping_since_feature = 0

    if business_id in business_dict:
        business_feature = business_dict[business_id]
    else:
        business_feature = (0,) * 17

    if business_id in days_open_dict:
        days_open_feature = days_open_dict[business_id]
    else:
        days_open_feature = 0

    if business_id in checkin_dict:
        checkin_feature = checkin_dict[business_id]
    else:
        checkin_feature = average_dict['checkin']

    if business_id in photo_dict:
        photo_feature = photo_dict[business_id]
    else:
        photo_feature = 0

    if business_id in tip_dict:
        tip_feature = tip_dict[business_id]
    else:
        tip_feature = average_dict['tip']

    return [*review_feature, *user_feature, *business_feature, checkin_feature]


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
        'n_estimators': 300
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
    review_dict, user_dict, friends_dict, average_dict, yelping_since_dict, business_dict, days_open_dict, checkin_dict, photo_dict, tip_dict = create_features()
    
    train_x = np.array([get_features(x, review_dict, user_dict, friends_dict, average_dict, yelping_since_dict, business_dict, days_open_dict, checkin_dict, photo_dict, tip_dict) for x in yelp_train], dtype = 'float32')
    train_y = np.array([float(x[2]) for x in yelp_train], dtype = 'float32')
    val_x = np.array([get_features(x, review_dict, user_dict, friends_dict, average_dict, yelping_since_dict, business_dict, days_open_dict, checkin_dict, photo_dict, tip_dict) for x in yelp_val], dtype = 'float32')
    
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


sc = SparkContext('local[*]', 'competition_project')
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


'''
def xgb_tuning(trial):
	param = {
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'random_state': 2020
    	}
    
    xgb = XGBRegressor(**param)
    kf = KFold(n_splits=5, shuffle=True)
    rmse_scores = -1 * cross_val_score(xgb, train_x, train_y, cv=kf, scoring='neg_root_mean_squared_error')
     
    return rmse_scores.mean()


def catboost_tuning(trial):
	param = {
        'random_state': trial.suggest_int('random_state', 1, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'max_bin': trial.suggest_int('max_bin', 10, 255),
        'verbose': 0,
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.1, 1.0)
    	}
    
    catboost = CatBoostRegressor(**param)
    kf = KFold(n_splits=5, shuffle=True)
    rmse_scores = -1 * cross_val_score(catboost, train_x, train_y, cv=kf, scoring='neg_root_mean_squared_error')
     
    return rmse_scores.mean()


xgb_study = optuna.create_study(direction='minimize')
xgb_study.optimize(xgb_tuning, n_trials=50)
xgb_best_params = xgb_study.best_params
print(xgb_best_params)


catboost_study = optuna.create_study(direction='minimize')
catboost_study.optimize(catboost_tuning, n_trials=50)
catboost_best_params = catboost_study.best_params
print(catboost_best_params)
'''