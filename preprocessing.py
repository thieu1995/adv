
import pickle
import numpy as np
import pandas as pd

DATA_PATH = "data/final/"
DATA_PATH_FINAL = "data/csv/"
list_files = ["user_20", "user_20_type", "user_50", "user_50_type", "user_100", "user_100_type",
              "user_count_20", "user_count_20_type", "user_count_50", "user_count_50_type",
              "user_count_100", "user_count_100_type"]
for it in list_files:
    with open(DATA_PATH + it + ".pkl", 'rb') as f:
        data = pickle.load(f)
        train, test = data["train"], data["test"]
        data = np.concatenate((train, test), axis=0)

    dataset = data[:, [0, 1, 2]]
    dataset = pd.DataFrame(dataset)
    dataset.to_csv(DATA_PATH_FINAL + it + ".csv", header=["user_id", "item_id", "click_count"], sep=',', index=False, encoding='utf-8')
