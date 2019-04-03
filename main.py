import os
import math
import numpy as np
from time import time
import logging
from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel

conf = SparkConf().setAppName("Advertiser Recommendation System")
sc = SparkContext(conf=conf)

DATA_PATH = "data/csv/"
RESULTS_PATH = "results/"
FILE_USERS_NAME = "user_20_type"
FILE_ITEMS_NAME = "item_industry_types"
# item_industry_types_and_hour (chon cai file nay neu users_name khong co chu: type)

list_files = ["user_20", "user_50", "user_100", "user_count_20", "user_count_50", "user_count_100"]
list_files_type = ["user_20_type", "user_50_type", "user_100_type", "user_count_20_type", "user_count_50_type", "user_count_100_type"]


#### Load files into rdd (spark)
## 1. Load clicks data file (users file)
raw_clicks_data = sc.textFile("file:///" + os.getcwd() + "/data/csv/" + FILE_USERS_NAME + ".csv")
raw_clicks_data_header = raw_clicks_data.take(1)[0]
clicks_dataset = raw_clicks_data.filter(lambda line: line != raw_clicks_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),int(float(tokens[2])))).cache()

logging.basicConfig(filename=RESULTS_PATH + "log_file-" + FILE_USERS_NAME + ".log", filemode='w', level=logging.INFO)
logging.info("File size is %s", clicks_dataset.count())
logging.info('\nColumns are: %s', raw_clicks_data_header)
logging.info("\n3 first records: \n%s\n", clicks_dataset.take(3))


## 2. load the items file
items_file = os.path.join(DATA_PATH, FILE_ITEMS_NAME + ".csv")
items_raw_data = sc.textFile(items_file)
items_raw_data_header = items_raw_data.take(1)[0]
# Parse
items_dataset = items_raw_data.filter(lambda line: line != items_raw_data_header) \
    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]), tokens[1])).cache()
logging.info("There are %s items in the complete dataset", items_dataset.count())


## 3. Split data into train, validation and test datasets
rddTraining, rddValidating, rddTesting = clicks_dataset.randomSplit([6,2,2], seed=1001)
validation_for_predict_RDD = rddValidating.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = rddTesting.map(lambda x: (x[0], x[1]))

## 4. Add user clicks in the training model
nbValidating = rddValidating.count()
nbTesting    = rddTesting.count()
logging.info("Training: %d, validation: %d, test: %d" , rddTraining.count(), nbValidating, rddTesting.count())

## 5. Training with parameters
seed = 1002
iterations = [10, 50, 100]                         # main para
ranks = [4, 8, 12, 16, 20]                         # main para
regularization_parameters = [0.01, 0.1, 0.2]       # main para
tolerance = 0.02

train_list_paras = []
for iteration in iterations:
    for rank in ranks:
        for rp in regularization_parameters:
            train_list_paras.append([iteration, rank, rp])

errors = []
train_log = []
min_error = float('inf')
best_iter, best_rank, best_rp = -1, -1, -1
for paras in train_list_paras:
    model = ALS.train(rddTraining, paras[1], seed=seed, iterations=paras[0], lambda_=paras[2])
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = rddValidating.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())
    errors.append(error)
    log = ["iter_" + str(paras[0]) + "-rank_" + str(paras[1]) + "-rp_" + str(paras[2]), error]
    train_log.append(log)
    logging.info('For iter: %s, rank: %s, rp: %s the RMSE: %s', paras[0], paras[1], paras[2], error)
    if error < min_error:
        min_error = error
        best_iter = paras[0]
        best_rank = paras[1]
        best_rp = paras[2]
logging.info('The best model was trained with iter: %s, rank: %s, rp: %s the RMSE: %s', best_iter, best_rank, best_rp, min_error)

## 6. Testing
model = ALS.train(rddTraining, best_rank, seed=seed, iterations=best_iter, lambda_=best_rp)
predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = rddTesting.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error_test = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())
test_log = ["The best model with iter_" + str(best_iter) + "-rank_: "+ str(best_rank) + "-rp_: " + str(best_rp), error_test]
train_log.append(test_log)
logging.info('\nFor testing data the RMSE is: %s\n', error_test)

## 7. Save results
np.savetxt(RESULTS_PATH + "train_file-" + FILE_USERS_NAME + ".csv", np.array(train_log), fmt='%s', delimiter=',', newline='\n', comments='')

# Give recommendations of items with a certain minimum number of click, count the number of click per item.
def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings)

item_ID_with_clicks_RDD = (clicks_dataset.map(lambda x: (x[1], x[2])).groupByKey())
item_ID_with_avg_clicks_RDD = item_ID_with_clicks_RDD.map(get_counts_and_averages)
item_rating_counts_RDD = item_ID_with_avg_clicks_RDD.map(lambda x: (x[0], x[1][0]))


## 8. Add new user with some information, then using our recommendation system to recommend
new_user_ID = 0
# The format of each line is (userID, ,itemID, clicks_count)
new_user_clicks = [
     (0,1,21),
     (0,5,3),
     (0,7,7),
     (0,13,80),
     (0,16,36),
     (0,21,1),
    ]
new_user_clicks_RDD = sc.parallelize(new_user_clicks)
logging.info('\nNew user clicks: \n%s\n', new_user_clicks_RDD.take(10))

## 9. Add them to the data we will use to train our recommender model
full_clicks_dataset = clicks_dataset.union(new_user_clicks_RDD)

## 10. Finally we train the ALS model using all the parameters we selected before
t0 = time()
new_clickings_model = ALS.train(full_clicks_dataset, best_rank, seed=seed, iterations=best_iter, lambda_=best_rp)
tt = time() - t0
logging.info("\nNew model trained in %s seconds.\n", round(tt,3))


##### Getting top recommendations
new_user_clicks_ids = map(lambda x: x[1], new_user_clicks)    # get just movie IDs
# keep just those not on the ID list
new_user_unclicked_items_RDD = (items_dataset.filter(lambda x: x[0] not in new_user_clicks_ids).map(lambda x: (new_user_ID, x[0])))
# Use the input RDD, new_user_unclicked_items_RDD, with new_clickings_model.predictAll() to predict new ratings for the movies
new_user_recommendations_RDD = new_clickings_model.predictAll(new_user_unclicked_items_RDD)


# Transform new_user_recommendations_RDD into pairs of the form (Items ID, Predicted Clicking)
new_user_recommendations_clicking_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))
new_user_recommendations_clicking_title_and_count_RDD = \
    new_user_recommendations_clicking_RDD.join(items_dataset).join(item_rating_counts_RDD)
logging.info("\n5 first of advertiser recommend to user is: \n%s\n", new_user_recommendations_clicking_title_and_count_RDD.take(5))


# (Title, Rating, Ratings Count In All)
new_user_recommendations_clicking_title_and_count_RDD = \
    new_user_recommendations_clicking_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))

# Get the highest clicked recommendations for the new user, filtering out items with less than 20 clicking.
top_items = new_user_recommendations_clicking_title_and_count_RDD.filter(lambda r: r[2]>=20).takeOrdered(10, key=lambda x: -x[1])
logging.info('\nTOP recommended items (with more than 20 clicks):\n%s', '\n'.join(map(str, top_items)))


##### Gettings individual clicks (Predict the number of clicks of a user for an item)
user_id = 0
item_id = 10
my_movie = sc.parallelize([(user_id, item_id)])        # user:0, item: 10
individual_item_click_RDD = new_clickings_model.predictAll(new_user_unclicked_items_RDD)
logging.info("\nPredict the number of clicks of user_id: %s to item_id: %s\n", user_id, item_id)
logging.info(individual_item_click_RDD.take(1))

##### Save and load model
model.save(sc, RESULTS_PATH + "model-" + FILE_USERS_NAME)
# same_model = MatrixFactorizationModel.load(sc, RESULTS_PATH + "model-" + FILE_USERS_NAME)