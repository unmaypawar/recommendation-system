import sys
from pyspark import SparkContext
import csv
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


def get_pearson_similarity(b1, b2, co_rated_users, business_user_ratings, business_average_ratings):
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


def get_predicted_rating(business_id, user_id, business_user_ratings, business_average_ratings, user_average_ratings, output_user_ratings, default_rating, neighbors_threshold):
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
			w = get_pearson_similarity(business_id, business, co_rated_users, business_user_ratings, business_average_ratings)
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


def item_based_collaborative_filtering():
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
	item_cf_output = yelp_val_index.map(lambda x: (x[1][1], x[1][0], get_predicted_rating(x[0][0], x[0][1], business_user_ratings, business_average_ratings, user_average_ratings, output_user_ratings, default_rating, neighbors_threshold))).collect()

	return item_cf_output


def model_based_collaborative_filtering():


	return model_cf_output


def get_final_output(item_cf_output, model_cf_output):

	return output


folder_path = sys.argv[1]
val_file = sys.argv[2]
output_file = sys.argv[3]


begin = time.time()


sc = SparkContext('local[*]', 'competition')
sc.setLogLevel('WARN')


yelp_train = sc.textFile(folder_path + '/yelp_train.csv').zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0].split(','))
yelp_val = sc.textFile(val_file).zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0].split(','))


item_cf_output = item_based_collaborative_filtering()
#model_cf_output = model_based_collaborative_filtering()
#output = get_final_output(item_cf_output, model_cf_output)


with open(output_file, 'w') as out_file:
	writer = csv.writer(out_file)
	writer.writerows([['user_id', 'business_id', 'prediction']])
	writer.writerows(item_cf_output)


get_stats()
end = time.time()
print('Execution Time:')
print(round(end - begin, 2), 's')