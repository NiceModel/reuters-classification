# -*- coding: utf-8 -*-

# File containing BERT model and the process to train the model.

# Part of the code in this file has been adapted from the following resources:
# https://colab.research.google.com/github/abhimishra91/transformers-tutorials
# https://github.com/abhimishra91/transformers-tutorials

from ast import literal_eval
from pathlib import Path
import time
import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
import arrow

import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader

# Note: needs transformers 3
import transformers
from transformers import BertTokenizer
from transformers import BertTokenizer

device = 'cuda' if cuda.is_available() else 'cpu'

MAX_LEN = 512
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 1e-05
DEBUG_RUN = False
FINAL_TRAIN = True
USE_POS_WEIGHTS = True

model_filename = f"bert__{arrow.now().format('MMM-Do-YYYY-HH_mm_ss')}.pth"

print(f"MAX_LEN: {MAX_LEN}")
print(f"TRAIN_BATCH_SIZE: {TRAIN_BATCH_SIZE}")
print(f"VALID_BATCH_SIZE: {VALID_BATCH_SIZE}")
print(f"EPOCHS: {EPOCHS}")
print(f"LEARNING_RATE: {LEARNING_RATE:.6f}")
print(f"USE_POS_WEIGHTS: {USE_POS_WEIGHTS}\n")
print(f"MODEL WILL BE SAVED IN: {model_filename}\n")
print(f"FINAL TRAIN: {FINAL_TRAIN}\n")

def log(msg):
    """Helper function to write stuff both to std output and a separate file.
    """
    print(msg)
    file.write(f"{msg}\n")

def time_str(msg, end, start):
    """Helper function for creating time delta strings for logging execition
    etc. times.
    """
    delta = end-start 
    time_str = f"{msg}: {delta:.2f} seconds ({delta/3600:.2f} hours)"
    return time_str

root = Path(__file__).absolute().parents[1]

output_file = root / "log.txt"
if output_file.exists():
    output_file.unlink()

file = open(output_file, "w")

code_counts_file = root / "data" / "codecounts.csv"
df_codes = pd.read_csv(code_counts_file)
pos_weights=torch.log1p(torch.tensor(df_codes["weights"].values, device=device))

if not (Path(root) / "data" / "train.pickle").exists():
    log("Reading data from csv files and creating pickle files...")
    train_file = root / "data" / "train.csv"
    valid_file = root / "data" / "valid_bert.csv"
    more_training = root / "data" / "rest_bert.csv"
    test_file = root / "data" / "test.csv"

    train_df = pd.read_csv(train_file)
    # We have more training data in a separate file.
    more_train_df = pd.read_csv(more_training)
    train_df = train_df.append(more_train_df)

    valid_df = pd.read_csv(valid_file)
    test_df = pd.read_csv(test_file)

    # Turn string representations of lists into actual lists.
    train_df["codes"] = train_df["codes"].apply(lambda x: literal_eval(x))
    valid_df["codes"] = valid_df["codes"].apply(lambda x: literal_eval(x))
    test_df["codes"] = test_df["codes"].apply(lambda x: literal_eval(x))

    train_df["labels"] = train_df["labels"].apply(lambda x: literal_eval(x))
    valid_df["labels"] = valid_df["labels"].apply(lambda x: literal_eval(x))
    test_df["labels"] = test_df["labels"].apply(lambda x: literal_eval(x))

    train_df["sentences"] = train_df["sentences"].apply(lambda x: literal_eval(x))
    valid_df["sentences"] = valid_df["sentences"].apply(lambda x: literal_eval(x))
    test_df["sentences"] = test_df["sentences"].apply(lambda x: literal_eval(x))

    with open(Path(root) / "data" / "train.pickle", 'wb') as handle:
        pickle.dump(train_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(Path(root) / "data" / "valid.pickle", 'wb') as handle:
        pickle.dump(valid_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(Path(root) / "data" / "test.pickle", 'wb') as handle:
        pickle.dump(test_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved data to pickle files.")

else:
    print("Loading data from pickle files...")
    with open(Path(root) / "data" / "train.pickle", 'rb') as handle:
        train_df = pickle.load(handle)
    with open(Path(root) / "data" / "valid.pickle", 'rb') as handle:
        valid_df = pickle.load(handle)
    with open(Path(root) / "data" / "test.pickle", 'rb') as handle:
        test_df = pickle.load(handle)
    print("Data loaded.")

if FINAL_TRAIN:
    train_df = train_df.append(test_df)

def trim_string(x, **kwargs):

    first_n_words = kwargs["first_n_words"]

    try:
        x = x.split(maxsplit=first_n_words)
        x = " ".join(x[:first_n_words])
    except AttributeError as e:
        # Catch nan values and use empty string instead.
        x = ""

    return x

def join_sents(sents, first_n_words):
  try:
    concatenated = " ".join(sents)
  except TypeError as e:
    concatenated = " ".join(sents[1:first_n_words+1])
  trimmed = trim_string(concatenated, first_n_words=first_n_words)

  return trimmed

train_df["title_text"] = train_df["sentences"].apply(join_sents, first_n_words=MAX_LEN)
valid_df["title_text"] = valid_df["sentences"].apply(join_sents, first_n_words=MAX_LEN)
test_df["title_text"] = test_df["sentences"].apply(join_sents, first_n_words=MAX_LEN)


target_size = len(df_codes)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.title_text = dataframe.title_text
        self.targets = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.title_text)

    def __getitem__(self, index):
        # set_trace()
        title_text = str(self.title_text[index])
        title_text = " ".join(title_text.split())

        inputs = self.tokenizer.encode_plus(
            title_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

# Creating the dataset and dataloader for the neural network
if DEBUG_RUN:
    size_train = 2000
    size_valid = 1000
    size_test = 1000
else:
    # Use all the data (except smaller set for validation to speed up training)
    size_train = train_df.shape[0]
    size_valid = valid_df.shape[0]
    size_test = test_df.shape[0]

train_dataset = train_df[0:size_train].reset_index(drop=True)
valid_dataset = valid_df[0:size_valid].reset_index(drop=True)
test_dataset = test_df[0:size_test].reset_index(drop=True)

# unused_valid = valid_df[size_valid:valid_df.shape[0]].reset_index(drop=True)
# Extend training set with unused dev set.
# train_dataset = train_dataset.append(unused_valid).reset_index(drop=True)

log("TRAIN Dataset: {}".format(train_dataset.shape))
log("VALID Dataset: {}".format(valid_dataset.shape))
if not FINAL_TRAIN:
    log("TEST Dataset: {}".format(test_dataset.shape))

training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
valid_set = CustomDataset(valid_dataset, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

calculate_dev_acc_every = (size_train//2)//TRAIN_BATCH_SIZE
calculate_dev_acc_every

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
valid_loader = DataLoader(valid_set, **test_params)
testing_loader = DataLoader(testing_set, **test_params)

# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        # Bert base has 768 hidden units.
        self.l3 = torch.nn.Linear(768, target_size)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

model = BERTClass()
_tmp = model.to(device)

def loss_fn(outputs, targets):
    if USE_POS_WEIGHTS:
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)(outputs, targets)
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

def validation(dataset_loader):
    model.eval()  

    fin_targets = np.empty((0,103), int)
    fin_outputs = np.empty((0,103), int)

    with torch.no_grad():
        for data in dataset_loader:
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            outputs = torch.sigmoid(model(ids, mask, token_type_ids)).cpu().detach().numpy()
            targets = data["targets"].cpu().detach().numpy()

            fin_outputs = np.append(fin_outputs, outputs, axis=0)
            fin_targets = np.append(fin_targets, targets, axis=0)
        
        outputs = fin_outputs > 0.5

        stats = {}
        stats['acc'] = metrics.accuracy_score(fin_targets, outputs)
        stats['hamming'] = metrics.hamming_loss(fin_targets, outputs)
        stats['f1_micro'] = metrics.f1_score(fin_targets, outputs, average='micro', zero_division=0)
        stats['prec'], stats['rec'], stats['f1_macro'], _ = metrics.precision_recall_fscore_support(fin_targets, outputs, average='macro', zero_division=0)
        
        return stats

def print_results(stats):
    print(f"Accuracy Score = {stats['acc']}")
    print(f"Hamming loss = {stats['hamming']}")
    print(f"F1 Score (Micro) = {stats['f1_micro']}")
    print(f"F1 Score (Macro) = {stats['f1_macro']}")
    print(f"Precision (Macro) = {stats['prec']}")
    print(f"Recall (Macro) = {stats['rec']}")   

best_f1_score = 0

def train(epoch):
    global best_f1_score
    best_f1_score = 0
    model.train()
    epoch_start_time = time.time()
    for i,data in enumerate(training_loader):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)
        #outputs = model(**ids, return_dict=False)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if i == calculate_dev_acc_every:
            log(f'\n--- Training loss:  {loss.item()} (Epoch: {epoch}, batch num: {i}): ---')

            valid_start_time = time.time()
            with torch.no_grad():
                model.eval()
                results = validation(valid_loader)
                print(f"\n--- Results for the validation set (Epoch: {epoch}, batch num: {i}): ---")
                print_results(results)
                if results['f1_macro'] > best_f1_score: 
                        best_f1_score = results['f1_macro']
                        # Save the best model so far.
                        torch.save(model.state_dict(), model_filename)
            model.train()
            valid_end_time = time.time()
            log(time_str("Validation with dev set took", valid_end_time, valid_start_time))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_end_time = time.time()
    log(time_str("Training for this epoch took", epoch_end_time, epoch_start_time))

log("Training starts")
training_start_time = time.time()
for epoch in range(EPOCHS):
    train(epoch)
    results = validation(valid_loader)
    print(f"\n--- Results for the validation set (Epoch: {epoch}): ---")
    print_results(results)
    if results['f1_macro'] > best_f1_score: 
        best_f1_score = results['f1_macro']
        # Save the best model so far.
        torch.save(model.state_dict(), model_filename) 
training_end_time = time.time()
log(time_str("Training ended. Training took", training_end_time, training_start_time))

log(f"Using model with a macro f1 score of {best_f1_score} for the dev set.")
model.load_state_dict(torch.load(model_filename))
if FINAL_TRAIN:
    print("This was the final training. We don't have test data to evaluate the model :)")
else:
    log("Evaluating model with the test set.")
    test_results = validation(testing_loader)
    print_results(test_results)
file.close()
