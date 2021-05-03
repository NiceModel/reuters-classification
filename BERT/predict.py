# -*- coding: utf-8 -*-
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

device = 'cuda' if cuda.is_available() else 'cpu'

MAX_LEN = 512
TEST_BATCH_SIZE = 16
DEBUG_RUN = False
USE_VALID_DATA = False

print(f"DEBUG_RUN: {DEBUG_RUN}")
print(f"USE_VALID_DATA: {USE_VALID_DATA}")

root = Path(__file__).absolute().parents[1]

# This is somewhat hacky method to enable importing functionalities, but
# gets the job done in the scope of this project.
# We could avoid this by having a different structure for our code and by utilizing packages.
import sys
sys.path.append(str(root))
print(root)
from finalizer import add_missing

code_counts_file = root / "data" / "codecounts.csv"
df_codes = pd.read_csv(code_counts_file)
target_size = len(df_codes)

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


model_filename = f"bert__{arrow.now().format('MMM-Do-YYYY-HH_mm_ss')}.pth"



def time_str(msg, end, start):
    """Helper function for creating time delta strings for logging execition
    etc. times.
    """
    delta = end-start 
    time_str = f"{msg}: {delta:.2f} seconds ({delta/3600:.2f} hours)"
    return time_str


# code_counts_file = root / "data" / "codecounts.csv"
# df_codes = pd.read_csv(code_counts_file)
# pos_weights=torch.log1p(torch.tensor(df_codes["weights"].values, device=device))

if USE_VALID_DATA:
    file_base_name = "valid"
else:
    file_base_name = "final_test"
if not (Path(root) / "data" / f"{file_base_name}.pickle").exists():
    print("Reading final test data from csv files and creating pickle files...")
    test_file = root / "data" / f"{file_base_name}.csv"

    test_df = pd.read_csv(test_file)
    test_df["sentences"] = test_df["sentences"].apply(lambda x: literal_eval(x))

    with open(Path(root) / "data" / f"{file_base_name}.pickle", 'wb') as handle:
        pickle.dump(test_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved data to pickle files.")

else:
    print("Loading data from pickle files...")
    with open(Path(root) / "data" / f"{file_base_name}.pickle", 'rb') as handle:
        test_df = pickle.load(handle)
    print("Data loaded.")


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

test_df["title_text"] = test_df["sentences"].apply(join_sents, first_n_words=MAX_LEN)

# target_size = len(df_codes)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.title_text = dataframe.title_text
        self.max_len = max_len

    def __len__(self):
        return len(self.title_text)

    def __getitem__(self, index):
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
            'index': torch.tensor(index, dtype=torch.long)
        }

# Creating the dataset and dataloader for the neural network
if DEBUG_RUN:
    size_test = 30
else:
    size_test = test_df.shape[0]

test_dataset = test_df[0:size_test].reset_index(drop=True)

print("TEST Dataset: {}".format(test_dataset.shape))

testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)


test_params = {'batch_size': TEST_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

testing_loader = DataLoader(testing_set, **test_params)

# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 


def predict(dataset_loader):
    model.eval()  

    fin_targets = np.empty((0,103), int)
    fin_outputs = np.empty((0,103), int)

    indices = torch.tensor([], dtype=torch.int).to(device)
    
    with torch.no_grad():
        for data in dataset_loader:
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            batch_indices = data['index'].to(device, dtype = torch.long)
            indices = torch.cat((torch.flatten(indices), batch_indices))

            # ids.to(device)
            # mask.to(device)
            # token_type_ids.to(device)
            outputs = torch.sigmoid(model(ids, mask, token_type_ids)).cpu().detach().numpy()

            fin_outputs = np.append(fin_outputs, outputs, axis=0)
        
        outputs = fin_outputs > 0.5

    indices = indices.cpu().detach().numpy()
    predictions = outputs[indices.argsort()]
    return predictions


model_filename = root / "BERT" / "bert__Apr-9th-2021-00_10_39.pth" 
print(f"Loading model for predicting the dataset.")
model = BERTClass()
model.to(device)
model.load_state_dict(torch.load(model_filename))

print("Generating predictions.")
results = predict(testing_loader)

if USE_VALID_DATA:
    file_name_to_save = "BERT_pred_numpy_no_zeros_valid.npy"
else:
    file_name_to_save = "BERT_pred_numpy_no_zeros_final.npy"

if Path(file_name_to_save).exists():
    Path(file_name_to_save).unlink()

# Add missing zeros for labels that were not present in training and save the final result.
add_missing(results)