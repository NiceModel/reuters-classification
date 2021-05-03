import gensim.downloader as api
import numpy as np
import pandas as pd
from collections import Counter
import re

#Load pretrained embeddings
pretrained_embeddings = api.load('word2vec-google-news-300')

#Read the data
train = pd.read_csv("../data/train.csv", usecols = ["tokens", "labels"],
                    converters={"tokens": lambda x: x.strip("[]").replace("'","").split(", ")})
valid = pd.read_csv("../data/valid.csv", usecols = ["tokens", "labels"],
                    converters={"tokens": lambda x: x.strip("[]").replace("'","").split(", ")})
test = pd.read_csv("../data/test.csv", usecols = ["tokens", "labels"],
                    converters={"tokens": lambda x: x.strip("[]").replace("'","").split(", ")})

#Count text tokens
counter = Counter()
for tokens in pd.concat([train["tokens"], valid["tokens"], test["tokens"]], ignore_index = True):
    counter.update(tokens)

#Calculate unknown words, excluding tokens including numbers
num_unknown_words = len(set([word for word in counter if word not in pretrained_embeddings.vocab and not re.match(r"\d+", word)]))
num_not_number = len(set([word for word in counter if not re.match(r"\d+", word)]))
print(f"number of unknown words: {num_unknown_words} ({round((num_unknown_words/num_not_number)*100, 0)}%)")

#Create list of embeddings for words in vocabulary
embeds = [(word, pretrained_embeddings[word].tolist()) for word in counter if word in pretrained_embeddings.vocab]

#Add padding and unknown tokens
pad = np.random.uniform(low = -1, high = 1, size = (300)).tolist()
unk = np.random.uniform(low = -1, high = 1, size = (300)).tolist()
embeds.insert(0,("<pad>", pad))
embeds.insert(0,("<unk>", unk))

#Save embeddings
embeds = pd.DataFrame(embeds, columns = ["word", "embedding"])
embeds.to_csv("embeddings_drop_unknown.csv")
del embeds

#Create list of embeddings, randomize unknown words
embeds = [(word, pretrained_embeddings[word].tolist()) if word in pretrained_embeddings.vocab 
                                              else (word, np.random.uniform(-1,1, (300)).tolist()) for word in counter 
                                              if not re.match(r"\d+", word)]
num_embedding = np.random.uniform(low = -1, high = 1, size = (300)).tolist()

#Add numeric, padding and unknown tokens
embeds.append(("<num>", num_embedding))
embeds.insert(0,("<pad>", pad))
embeds.insert(0,("<unk>", unk))

#Save embeddings
embeds = pd.DataFrame(embeds, columns = ["word", "embedding"])
embeds.to_csv("embeddings.csv")

