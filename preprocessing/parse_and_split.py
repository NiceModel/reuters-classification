 
""" This script parses the whole Reuters corpus, performs stratified splitting and stores the results in files train.csv, valid.csv and test.csv"""

from lxml import etree
from pathlib import Path
from skmultilearn.model_selection import IterativeStratification
import pandas as pd
import numpy as np
import spacy
import re
import time

nlp = spacy.load('en_core_web_sm', disable=['parser','tagger','ner','lemmatizer'])

# Some helper functions
def total_time(start_time, end_time):
    """Returns time in minutes and seconds between two checkpoints."""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



def cleaner(text):
    """Returns text with all but alphanumeric characters replaced with white space."""
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text) 
    return text.strip()


def tokenizer(sentence): 
    """Uses Spacy pipeline to tokenize a sentence."""
    return [w.text.lower() for w in nlp(cleaner(sentence))]
   

# Set directory paths
corpus_dir = Path(__file__).parent / "REUTERS_CORPUS_2"
print(f"Parsing data from: {corpus_dir}\n")

train_file = Path(__file__).parent / "../data/train.csv"
valid_file = Path(__file__).parent / "../data/valid.csv"
test_file = Path(__file__).parent / "../data/test.csv"

# Delete existing results file.
train_file.unlink(missing_ok=True)
valid_file.unlink(missing_ok=True)
test_file.unlink(missing_ok=True)

# Process each document in the corpus
processed_docs = 0
documents = []
failed = []

start_time = time.time()

for file_path in corpus_dir.glob("**/*.xml"):
    if processed_docs > 0 and processed_docs % 100 == 0:
        print(f"Number of documents processed: {processed_docs}", end="\r", flush=True)
    try:
        # Parse a single .xml document
        tree = etree.parse(str(file_path))
        sent_list = [tree.find("headline").text]  # The headline is always the first sentence of the sentence list.
        token_list = tokenizer(sent_list[0])

        for element in tree.find("text"):
            sentence = element.text
            sent_list.append(sentence)
            token_list += tokenizer(sentence)
        
        metadata = tree.find("metadata")
        code_list = []

        for codes in metadata.iter():
            if codes.get("class") == "bip:topics:1.0":
                for code in codes.iter():
                    if code.get("code"):
                        code_list.append(code.get("code"))

        # Gather relevant data from the file into a dictionary...
        document = {
            "sentences": sent_list,
            "tokens": token_list,
            "codes": code_list
        }

        # ...and append it to a list of dictionaries.
        documents.append(document)
        processed_docs += 1

    except Exception as e:
        print(f"Failed processing document {file_path}")
        print(e)
        failed.append(file_path)

parse_time = time.time()
mins, secs = total_time(start_time, parse_time)

# Create a Pandas dataframe out of a list of dictionaries
print(f"\nAll documents parsed and tokenized in {mins} min {secs} sec.")
print("Processing information...")
df = pd.DataFrame(documents)
n_samples = len(df)

# Build a dictionary with topic codes as keys and running numbers as values
num_code = 0
code_dict = {}
with open('REUTERS_CORPUS_2/topic_codes.txt', 'r') as f:
    for line in f:
        if line.startswith(";"):
            continue
        code = line.split("\t",1)[0]
        code_dict[code] = num_code
        num_code += 1      

# Count the frequency of each code and form the distribution of codes
codecounts = [0] * 126
code_names = code_dict.keys()

def code_add(l):
    for code in l:
        codecounts[code_dict[code]] += 1

df["codes"].apply(lambda x: code_add(x))

code_df = pd.DataFrame({"code":code_names, "count":codecounts})

# List codes with frequency > 0, calculate 'default' weight for each code (negative / positive) and save them in a .csv file

mask = code_df["count"] != 0
code_df = code_df[mask].reset_index(drop=True)
code_df["weights"] = code_df["count"].apply(lambda x: (n_samples-x)/x)
code_df.to_csv("../data/codecounts.csv")

# Update code dictionary to only hold non-redundant codes
code_dict = dict(zip(code_df["code"], list(code_df.index.values)))

# Vectorize labels (as multi-hot binary vectors)
num_labels = len(code_dict)

def vectorize_classes(x, code_dict={}):
    label_vector = np.zeros((num_labels), dtype=int)
    non_zero_idx = [code_dict[label] for label in x]
    label_vector[non_zero_idx] = 1
    label_vector = label_vector.tolist()
    return label_vector

df["labels"] = df["codes"].apply(vectorize_classes, code_dict=code_dict)

# Perform iterative stratified sampling to form train, validation and test sets (with sizes 80 %, 10 % and 10 % respectively).
X = df["sentences"]
y = df["labels"]
y_dummy = pd.DataFrame(y.tolist())

strat_start = time.time()

print("\nAll samples processed. Performing iterative stratified sampling to form train, validation and test sets.")
print("Forming training set. This might take up to 15 minutes. Please wait patiently...")
train_tv_fold = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[0.2, 0.8])
tr_idx, vt_idx = next(train_tv_fold.split(X, y_dummy))

strat_mid = time.time()
mins, secs = total_time(strat_start, strat_mid)
print(f"Forming the training set took: {mins} min, {secs} sec")

print("\nForming validation and test sets. This will take a short while.")
X_vt, y_vt = X[vt_idx], y[vt_idx]
y_vt_dummy = pd.DataFrame(y_vt.tolist())

valid_test_fold = IterativeStratification(n_splits=2, order=2)
val_idx, test_idx = next(valid_test_fold.split(X_vt, y_vt_dummy))

strat_end = time.time()
mins, secs = total_time(strat_start, strat_end)
print(f"Forming the validation and test sets took: {mins} min, {secs} sec")

df_train = df.iloc[tr_idx,:]
df_valid = df.iloc[val_idx,:]
df_test = df.iloc[test_idx,:]    

stop_time = time.time()
mins, secs = total_time(start_time, stop_time)

print(f"\nTotal time to parse all documents and perform stratified splits: {mins} min, {secs} sec")

# Write results to appropriate files
df_train.to_csv(train_file, index=False)
df_valid.to_csv(valid_file, index=False)
df_test.to_csv(test_file, index=False)

print(f"\nSaved results to files: \n{train_file}, \n{valid_file}, \n{test_file}")
print(f"Failed files: {failed}")
print(f"Numbers of samples are train: {len(df_train)}, validation: {len(df_valid)}, test: {len(df_test)}")
