
""" This script takes the predictions of a model, adds redundant labels and writes the result as a .csv file. """

import numpy as np
import pandas as pd

def add_missing(predictions):
    """Adds zeroes for columns of redundant labels and writes the result into a a file 'predictions_KuuKiviKin.csv'.
    
    Args:
    predictions: Numpy array of size n * 103, holding the predicted labels as multi-hot binary vectors, with n as sample size.
    """
    n = np.shape(predictions)[0]
    final = np.hstack((
        np.zeros((n,11)),
        predictions[:,:60],
        np.zeros((n,9)),
        predictions[:,60:75],
        np.zeros((n,1)),
        predictions[:,75:],
        np.zeros((n,2))
    ))

    columns = []
    with open('../data/topic_codes.txt', 'r') as f:
        for line in f:
            if line.startswith(";"):
                continue
            code = line.split("\t",1)[0]
            columns.append(code)

    ids = pd.read_csv("../data/final_test.csv", header=0, names=["id","a","b"])
    ids = ids["id"]
    data = pd.DataFrame(final, columns=columns)
    result = pd.concat([ids,data], axis=1)   
    result.set_index("id", inplace=True)    
    print(result.head())
    result.to_csv("predictions_KuuKiviKin.csv", sep=" ", index_label="id")
