
"""This script was used in early data exploration of the corpus."""

from pathlib import Path
from ast import literal_eval
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data from .csv file
df = pd.read_csv("../data/documents.csv")

# Check the distribution of the number of tokens 
df["num_tokens"] = df["text_tokenized"].apply(lambda x: len(literal_eval(x)))
print(df.describe())
sns.displot(data=df, x="num_tokens")
plt.title("Distribution of number of tokens in news text body")
plt.show()

# Build a dictionary with topic codes as keys and running numbers as values
num_code = 0
code_dict = {}
with open('../REUTERS_CORPUS_2/topic_codes.txt', 'r') as f:
    for line in f:
        if line.startswith(";"):
            continue
        code = line.split("\t",1)[0]
        code_dict[code] = num_code
        num_code += 1      

# Count the frequency of each code and form the distribution
codecounts = [0] * 126
code_names = code_dict.keys()
no_labels = {"count": 0}

def code_add(l):
    if not l:
        no_labels["count"] += 1
    for code in l:
        codecounts[code_dict[code]] += 1

df["codes"].apply(lambda x: code_add(literal_eval(x)))

print("Number of news with no labels:", no_labels)

# Generate the distribution of different codes and plot a bar diagram
code_df = pd.DataFrame({"code":code_names, "count":codecounts})
sns.barplot(data=code_df, x="code", y="count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.title("Distribution of codes")
plt.show()

# List redundant codes (with frequency = 0)
mask = code_df["count"]==0
redundant = code_df[mask]
print(redundant["code"])

# Aggregate codes beginning with the same letter.
# CCAT = 44, other C-codes = 11-43
# ECAT = 70, other E-codes = 45-70
# GCAT = 90, other G-codes = 71-113
# MCAT = 123, other G-codes = 114-122

ccat = code_df["count"][44]
ecat = code_df["count"][70]
gcat = code_df["count"][90]
mcat = code_df["count"][123]
cats = [ccat, ecat, gcat, mcat]

counts = np.zeros((8,1), dtype=int)  # Used to store other results

def c_codes(l, x, code_range):
    for code in l:
        if code_dict[code] in code_range:
            counts[x,0] += 1
            break

c_range = range(11,44)   # codes beginning with C (not including CCAT)
c_range2 = range(11,45)  # codes beginning with C (including CCAT)
e_range = range(45,71)   # etc...
e_range2 = range(45,72)
g_range = list(range(71,90)) + list(range(91,114))
g_range2 = range(71,114)
m_range = range(114,123)
m_range2 = range(114,124)
ranges = [c_range, c_range2, e_range, e_range2, g_range, g_range2, m_range, m_range2]

for i, r in enumerate(ranges):
    df["codes"].apply(lambda x: c_codes(literal_eval(x), i, r))

# Create printouts for different combinations

names = ['C', 'E', 'G', 'M']
for i in range(4):
    print("Number of news with code "+names[i]+"CAT:",cats[i])
    print("Number of news with any other code beginning with "+names[i]+":",counts[2*i, 0])
    print("Number of all news with a code beginning with "+names[i]+":",counts[2*i+1, 0],"\n")

# List sorted codes with frequency > 0 and 
# calculate weight for each code (negative / positive)

mask2 = code_df["count"]!=0
freqs = code_df[mask2].reset_index(drop=True)
freqs["weights"] = freqs["count"].apply(lambda x: (len(df)-x)/x)
sorted_counts = freqs.sort_values("count", ascending=False)
output = sorted_counts.to_string()
print(output)

