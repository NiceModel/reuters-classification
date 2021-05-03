 
""" This script parses the whole Reuters corpus and stores the results in a single documents.csv file """

from lxml import etree
from pathlib import Path
import pandas as pd
import re
import spacy

# Spacy pipeline definitions.
nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner", "lemmatizer"])
stopwords = nlp.Defaults.stop_words  # Default stopwords (326) for spacy.
stopwords.add("umph")  # You can add more stopwords if you like.

# Helper functions.
def tokenizer(text): 
    """Performs tokenization using the Spacy pipeline."""
    tokens = [token.text.lower() for token in nlp(cleaner(text))]
    # tokens = [word for word in tokens if not word in stopwords]  # Use this instead if you want to remove stop words.
    return tokens
    

def cleaner(text):
    """Replaces non alphanumeric characters with white space from a given sequence."""
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text) 
    return text.strip()


# Setup for parsing
corpus_dir = Path(__file__).parent / "REUTERS_CORPUS_2"
print(f"\nParsing data from: {corpus_dir}\n")

processed_docs = 0
documents = []
failed = []
test_run = False

results_file = Path(__file__).parent / "../data/documents.csv"
results_file.unlink(missing_ok=True)  # Delete (potential) existing results file.

# Parse all .xml files in the corpus
first_save = True
for file_path in corpus_dir.glob("**/*.xml"):
    if processed_docs > 0 and processed_docs % 500 == 0:
        # Write docs to file in batches of 500 docs
        df = pd.DataFrame(documents)
        df.to_csv(results_file, index=False, mode="a", header=first_save)
        first_save = False
        documents = []
        print(f"Number of processed docs: {processed_docs}")
        if test_run:
            break
    try:
        tree = etree.parse(str(file_path))

        headline = tree.find("headline").text

        sent_list = []
        for element in tree.find("text"):
            sent_list.append(element.text)
        whole_text = " ".join(sent_list)

        metadata = tree.find("metadata")
        code_list = []

        for codes in metadata.iter():
            if codes.get("class") == "bip:topics:1.0":
                for code in codes.iter():
                    if code.get("code"):
                        code_list.append(code.get("code"))

        # Store results in a list of dictionaries.
        document = {
            "file": file_path.name,
            "headline": headline,
            "head_tokens": tokenizer(headline),
            "text": whole_text,
            "text_tokenized": tokenizer(whole_text),
            "codes": code_list,
        }
        documents.append(document)
        processed_docs += 1

    except Exception as e:
        print(f"Failed processing document {file_path}")
        print(e)
        failed.append(file_path)

if not test_run:
    # Write possibly remaining docs
    df = pd.DataFrame(documents)
    df.to_csv(results_file, index=False, mode="a", header=first_save)

print(f"Saved results to {results_file}")
print("Failed files:")
print(failed)