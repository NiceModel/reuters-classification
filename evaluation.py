import numpy as np
import pandas as pd
import sklearn.metrics as skm
from ast import literal_eval

# Build a dictionary with topic codes as keys and running numbers as values
num_code = 0
code_dict = {}
with open('REUTERS_CORPUS_2/topic_codes.txt', 'r') as f:
    for line in f:
        if line.startswith(";"):
            continue
        code = line.split("\t",1)[0]
        code_dict[num_code] = code
        num_code += 1      

def add_missing(predictions):
    """Adds zeroes for columns of redundant labels and writes the result into a a file 'pred_numpy.npy'.
    
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
    return final

# A function for adding a list of labels for each data point. 
def labelize(predictions):
    """A function for translating vectorized predictions into actual labels.

    Args:
    predictions: A numpy array of dimension N * 103, where N is the number of rows (predictions).
    
    Returns:
    result: A single column DataFrame, where labels have been added as lists.
    """

    predictions = add_missing(predictions)  # Add missing columns
        
    rows = []
    for row in predictions:
        label_list = []
        for i, x in enumerate(row):
            if x == 1:
                label_list.append(code_dict[i])
        rows.append({"codes": label_list})

    new_column = pd.DataFrame(rows)
    return new_column

# Create random label data
def create_random():
    y_true = np.zeros((100,103))
    nrows = (len(y_true[:,0]))
    for i in range(nrows):
        random_indexes = np.random.choice(103, size=(8), replace=False)
        for x in random_indexes:
            y_true[i][x] += np.random.randint(2)

    y_pred = np.copy(y_true)
    for i in range(nrows):
        random_indexes = np.random.choice(103, size=(2), replace=False)
        for x in random_indexes:      
            y_pred[i][x] = 1 - y_pred[i][x]
    return (y_true, y_pred)

# Create evaluation statistics
def evaluate(y_true, y_pred):
    prec, rec, f1, _ = skm.precision_recall_fscore_support(y_true, y_pred, average="weighted") 
    print(f"Macro F1-score is: {f1}.")
    print(f"Precision is: {prec}.")
    print(f"Recall is: {rec}.")

# Uncomment the two rows below to generate and evaluate random data
# true, pred = create_random()
# results = evaluate(true, pred)


# Read predictions of BERT and labelize them
bert = np.load("../BERT_validation.npy")
bert_pred = labelize(bert)

# Read true labels for validation set
df = pd.read_csv("../data/valid.csv")
true = df["codes"].apply(lambda x: literal_eval(x))

# Check first 5 rows of each
print(bert_pred.head())
print(true.head())
