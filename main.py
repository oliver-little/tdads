import importlib
import time
import json
from pprint import pprint
import copy

import numpy as np
import keras.datasets as kd

import config

(train_x, train_y), (test_x, test_y) = kd.mnist.load_data()
modules = config.modules

def calculate_metrics(test_y, predictions, train_time, predict_time):
    # return dictionary
    mets = {}
    # calculate accuracy by summing truth values
    mets["accuracy"] = (test_y == predictions).sum()/len(predictions)

    confusion_matrix = np.zeros((10, 10), dtype=int)
    np.add.at(confusion_matrix, (predictions, test_y), 1)
    mets["confusion_matrix"] = confusion_matrix.tolist()


    mets['predict_time'] = predict_time
    mets["train_time"] = str(train_time) + "s"

    return mets

metrics = {}

for module_string, kwargs in modules.items():
    module = importlib.import_module(module_string)

    print(f"Fitting model {module_string}")
    clock = time.perf_counter()
    fit_result = None
    if kwargs["fit"] is not None:
        fit_result = module.fit(copy.deepcopy(train_x), copy.deepcopy(train_y), **kwargs["fit"])
    else:
        fit_result = module.fit(copy.deepcopy(train_x), copy.deepcopy(train_y))
    train_time = time.perf_counter() - clock

    print(f"Predicting from model {module_string}")
    clock = time.perf_counter()
    predictions = None
    if kwargs["predict"] is not None:
        predictions = module.predict(copy.deepcopy(test_x), fit_result, **kwargs["predict"])
    else:
        predictions = module.predict(copy.deepcopy(test_x), fit_result)
    predict_time = time.perf_counter() - clock

    print(f"Calculating {module_string} metrics")
    metrics[module_string] = calculate_metrics(test_y, predictions, train_time, predict_time)
    print(f"Finished {module_string}\n")

    with open("results.json", "w") as json_file:
        json.dump(metrics, json_file)

pprint(metrics)
