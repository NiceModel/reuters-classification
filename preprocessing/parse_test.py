
""" This script parses the final test set and stores the results in a single file final_test.csv"""

from lxml import etree
from pathlib import Path
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


def get_path_to_resource_file(file_basename):
    """Returns relative path of a given file."""
    filepath_relative_to_this_file = Path(__file__).parent / file_basename
    return filepath_relative_to_this_file


def cleaner(text):
    """Returns text with all but alphanumeric characters replaced with white space."""
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text) 
    return text.strip()


def tokenizer(sentence): 
    """Uses Spacy pipeline to tokenize a sentence."""
    return [w.text.lower() for w in nlp(cleaner(sentence))]
   

# Set directory paths
corpus_dir = Path(__file__).parent / "text-test-corpus-STRIPPED"
print(f"Parsing data from: {corpus_dir}\n")

final_test_file = Path(__file__).parent / "../data/final_test.csv"

# Delete existing results file.
final_test_file.unlink(missing_ok=True)

# Process each document in the corpus
processed_docs = 0
documents = []
failed = []

start_time = time.time()

for file_path in corpus_dir.glob("**/*.xml"):
    if processed_docs > 0 and processed_docs % 100 == 0:
        print(f"Number of documents processed: {processed_docs}", end="\r", flush=True)
    
    # Parse a single .xml document
    tree = etree.parse(str(file_path))
    try:
        # The headline is always the first sentence of the sentence list...
        sent_list = [tree.find("headline").text]  
        token_list = tokenizer(sent_list[0])
    except Exception as e:
        # ...unless the headline is missing.
        print(f"A missing headline for file: {file_path}")
        print(e)
        sent_list = []  
        token_list = []
    
    for element in tree.find("text"):
        sentence = element.text
        sent_list.append(sentence)
        token_list += tokenizer(sentence)
    
    # Gather relevant data from the file into a dictionary...
    document = {
        "file": file_path.name,
        "sentences": sent_list,
        "tokens": token_list
    }

    # ...and append it to a list of dictionaries.
    documents.append(document)
    processed_docs += 1

parse_time = time.time()
mins, secs = total_time(start_time, parse_time)

# Create a Pandas dataframe out of a list of dictionaries
print(f"\nAll documents parsed and tokenized in {mins} min {secs} sec.")
print("Processing information...")
df = pd.DataFrame(documents)

# Write results to appropriate files
df.to_csv(final_test_file, index=False)

print(f"\nSaved results to file: {final_test_file}")
print(f"Failed files: {failed}")
print(f"Numbers of samples: {len(df)}")

print(df.head())