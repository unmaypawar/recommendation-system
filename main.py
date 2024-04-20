import sys
from pyspark import SparkContext
import csv
import json
import numpy as np
from xgboost import XGBRegressor
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

    rmse = (rmse / len(prediction)) ** (1/2)
    for key in error_dist:
    	print(key, ':',error_dist[key])
    print('RMSE:', rmse)


def get_pearson_similarity(b1, b2, co_rated_users):
	b1_ratings = business_user_ratings.value[b1]
	b2_ratings = business_user_ratings.value[b2]
	b1_average_rating = business_average_ratings.value[b1]
	b2_average_rating = business_average_ratings.value[b2]

	if len(co_rated_users) <= 1:
		difference = abs(b1_average_rating - b2_average_rating)
		if difference <= 1:
			w = 1
		elif difference <= 2:
			w = 0.5
		else:
			w = 0

		return w
	
	numerator = 0
	denominator_u1 = 0
	denominator_u2 = 0
	for user in co_rated_users:
		normalised_u1 = b1_ratings[user] - b1_average_rating
		normalised_u2 = b2_ratings[user] - b2_average_rating
		numerator += normalised_u1 * normalised_u2

		d_u1 = (b1_ratings[user] - b1_average_rating) ** 2
		d_u2 = (b2_ratings[user] - b2_average_rating) ** 2
		denominator_u1 += d_u1
		denominator_u2 += d_u2

	if numerator == 0:
		w = 0
	else:
		w = numerator / ((denominator_u1) ** (1/2) * (denominator_u2) ** (1/2))

	return w


def get_predicted_rating(business_id, user_id):
	if business_id == -1 and user_id == -1:
		predicted_rating = default_rating

	elif business_id == -1:
		predicted_rating = user_average_ratings.value[user_id]

	elif user_id == -1:
		predicted_rating = business_average_ratings.value[business_id]

	else:
		neighbors = []
		user_history = output_user_ratings.value[user_id]
		
		for business, rating in user_history.items():
			co_rated_users = set(business_user_ratings.value[business_id].keys()).intersection(set(business_user_ratings.value[business].keys()))
			w = get_pearson_similarity(business_id, business, co_rated_users)
			if w > 0:
				neighbors.append((w, rating))

		if not neighbors:
			predicted_rating = default_rating

		else:
			numerator = 0
			denominator = 0

			neighbors = sorted(neighbors, key = lambda x: -x[0])[:neighbors_threshold]
			for w, rating in neighbors:
				numerator += w * rating
				denominator += abs(w)

			if numerator == 0:
				predicted_rating = default_rating
			else:
				predicted_rating = numerator / denominator

	return predicted_rating


def index_val(business_id, user_id, business_train_index, users_train_index):
	if business_id in business_train_index:
		business_index = business_train_index[business_id]
	else:
		business_index = -1

	if user_id in users_train_index:
		user_index = users_train_index[user_id]
	else:
		user_index = -1

	return (business_index, user_index)


def get_features(x):
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

	return [*review_feature, *user_feature, *business_feature]


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


folder_path = sys.argv[1]
val_file = sys.argv[2]
output_file = sys.argv[3]


begin = time.time()


sc = SparkContext('local[*]', 'task2')
sc.setLogLevel('WARN')


yelp_train = sc.textFile(folder_path + '/yelp_train.csv').zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0].split(','))
yelp_val = sc.textFile(val_file).zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0].split(','))


users_train = yelp_train.map(lambda x: x[0]).distinct().zipWithIndex()
users_train_index = users_train.map(lambda x: (x[0], x[1])).collectAsMap()
business_train = yelp_train.map(lambda x: x[1]).distinct().zipWithIndex()
business_train_index = business_train.map(lambda x: (x[0], x[1])).collectAsMap()
yelp_train_index = yelp_train.map(lambda x: ((business_train_index[x[1]], users_train_index[x[0]]), float(x[2])))
yelp_val_index = yelp_val.map(lambda x: (index_val(x[1], x[0], business_train_index, users_train_index), (x[1], x[0])))


business_user_ratings = yelp_train_index.map(lambda x: (x[0][0], (x[0][1], x[1]))).groupByKey().mapValues(dict).collectAsMap()
business_user_ratings = sc.broadcast(business_user_ratings)
user_average_ratings = yelp_train_index.map(lambda x: (x[0][1], x[1])).mapValues(lambda x: (x, 1)).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])).mapValues(lambda x: x[0] / x[1]).collectAsMap()
user_average_ratings = sc.broadcast(user_average_ratings)
business_average_ratings = yelp_train_index.map(lambda x: (x[0][0], x[1])).mapValues(lambda x: (x, 1)).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])).mapValues(lambda x: x[0] / x[1]).collectAsMap()
business_average_ratings = sc.broadcast(business_average_ratings)


output_user_index = yelp_val_index.map(lambda x: x[0][1]).distinct().map(lambda x: (x, None))
output_business_index = yelp_val_index.map(lambda x: x[0][0]).distinct()
output_user_ratings = output_user_index.leftOuterJoin(yelp_train_index.map(lambda x: (x[0][1], (x[0][0], x[1])))).map(lambda x: (x[0], x[1][1])).groupByKey().mapValues(list)
output_user_ratings = sc.broadcast(output_user_ratings.map(lambda x: (x[0], dict(x[1]) if x[1][0] is not None else (x[0], {}))).collectAsMap())


default_rating = 3.75
neighbors_threshold = 20
cf_output = yelp_val_index.map(lambda x: (x[1][1], x[1][0], get_predicted_rating(x[0][0], x[0][1]))).collect()


yelp_train = yelp_train.collect()
yelp_val = yelp_val.collect()
review_dict = sc.textFile(folder_path + '/review_train.json').map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], (float(x['useful']), float(x['funny']), float(x['cool'])))).groupByKey().mapValues(get_average).collectAsMap()
user_dict = sc.textFile(folder_path + '/user.json').map(lambda x: json.loads(x)).map(lambda x: (x['user_id'], (float(x['average_stars']), float(x['review_count']), float(x['fans'])))).collectAsMap()
business_dict = sc.textFile(folder_path + '/business.json').map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], (float(x['stars']), float(x['review_count'])))).collectAsMap()


train_x = np.array([get_features(x) for x in yelp_train], dtype = 'float32')
train_y = np.array([float(x[2]) for x in yelp_train], dtype = 'float32')
xgb_param = {'n_estimators': 300, 'max_depth': 10, 'learning_rate': 0.02, 'random_state': 1, 'subsample': 0.8, 'colsample_bytree': 0.5}
xgb = XGBRegressor(**xgb_param)
xgb.fit(train_x, train_y)


val_x = np.array([get_features(x) for x in yelp_val], dtype = 'float32')
val_y = xgb.predict(val_x)


output = []
alpha = 0.1
for item, model in zip(cf_output, val_y):
	final_rating = alpha * item[2] + (1 - alpha) * model
	temp_output = []
	temp_output.append(item[0])
	temp_output.append(item[1])
	temp_output.append(final_rating)
	output.append(temp_output)


with open(output_file, 'w') as out_file:
	writer = csv.writer(out_file)
	writer.writerows([['user_id', 'business_id', 'prediction']])
	writer.writerows(output)


get_stats()
end = time.time()
print('Duration:', end - begin)