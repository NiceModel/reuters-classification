
""" Checks the strafied split by plotting a bar diagram of the relative frequencies of labels in each class. """

from pathlib import Path
from ast import literal_eval
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data from .csv files
print("\nReading files")
df_train = pd.read_csv("../data/train.csv")
df_valid = pd.read_csv("../data/valid.csv")
df_test = pd.read_csv("../data/test.csv")
print(f"Lengths: {len(df_train)}, {len(df_valid)}, {len(df_test)}")

train_codes = df_train["labels"].apply(lambda x: (literal_eval(x)))
valid_codes = df_valid["labels"].apply(lambda x: (literal_eval(x)))
test_codes = df_test["labels"].apply(lambda x: (literal_eval(x)))

# Count the frequencies of each label in the training set
print("\nCounting frequencies for the training set")
codes = len(train_codes[0])
tr_freq = [0]*codes
tr_n = len(train_codes)

def counts(output, line):
    for i,x in enumerate(line):
        if x == 1:
            output[i] += 1

train_codes.apply(lambda x: counts(tr_freq,x))

tr_p = [0]*codes
for i in range(codes):
    tr_p[i] = tr_freq[i]/tr_n

# Count the frequencies of each label in the validation set
print("Counting frequencies for the validation set")
vl_freq = [0]*codes
vl_n = len(valid_codes)

valid_codes.apply(lambda x: counts(vl_freq,x))

vl_p = [0]*codes
for i in range(codes):
    vl_p[i] = vl_freq[i]/vl_n

# Count the frequencies of each label in the test set
print("Counting frequencies for the test set")
te_freq = [0]*codes
te_n = len(test_codes)

test_codes.apply(lambda x: counts(te_freq,x))

te_p = [0]*codes
for i in range(codes):
    te_p[i] = te_freq[i]/te_n

# Generate the distribution of different codes and plot a bar diagram
print("Plotting the diagram.")
distribution = pd.DataFrame({"train": tr_p, "validation": vl_p, "test": te_p})
distribution.plot.bar()
plt.tight_layout()
plt.title("Distribution of codes")
plt.show()
