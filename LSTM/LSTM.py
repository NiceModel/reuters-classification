from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import re
from sklearn import metrics
import random
from torch.utils.data.sampler import Sampler

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        return self.dataframe.iloc[index]

def run_model(name, batch_size, drop_unknown, hidden_size, weights):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #DROP_UNKNOWN drops words not in pre-trained embeddings
    DROP_UNKNOWN = drop_unknown
    BATCH_SIZE = batch_size

    #Load train and validation sets
    train = pd.read_csv("../data/train.csv", usecols = ["tokens", "labels"],
                        converters={"tokens": lambda x: x.strip("[]").replace("'","").split(", "),
                                    "labels": lambda x: list(map(int, x.strip("[]").replace("'","").split(", ")))})
    valid = pd.read_csv("../data/valid.csv", usecols = ["tokens", "labels"],
                        converters={"tokens": lambda x: x.strip("[]").replace("'","").split(", "),
                                    "labels": lambda x: list(map(int, x.strip("[]").replace("'","").split(", ")))})
    
    #Load code weights
    code_weights = pd.read_csv("../data/codecounts.csv", usecols = ["weights"])

    w_default = code_weights["weights"].to_numpy()
    w_log = np.log10(w_default)
    w_log = torch.from_numpy(w_log).to(device)                               

    #Load embeddings:
    if DROP_UNKNOWN:
        embeds = pd.read_csv("../Embeddings/embeddings_drop_unknown.csv",
                            converters={"embedding": lambda x: list(map(float, x.strip("[]").split(", ")))})
        words_list = set(embeds["word"].tolist())
        train["tokens"] = train["tokens"].apply(lambda x: [word for word in x if word in words_list])
        valid["tokens"] = valid["tokens"].apply(lambda x: [word for word in x if word in words_list])
        del words_list
    else:
        embeds = pd.read_csv("../Embeddings/embeddings.csv",
                            converters={"embedding": lambda x: list(map(float, x.strip("[]").split(", ")))})
        words_list = set(embeds["word"].tolist())
        train["tokens"] = train["tokens"].apply(lambda x: [word if word in words_list else
                                                        "<num>" if re.match(r"\d+", word) else "<unk>"
                                                        for word in x])
        valid["tokens"] = valid["tokens"].apply(lambda x: [word if word in words_list else
                                                        "<num>" if re.match(r"\d+", word) else "<unk>"
                                                        for word in x])                                                
        del words_list

    #Define model
    class LSTM(nn.Module):
        def __init__(self, vocab_size, hidden_size, n_layers, bidirectional = False, linear_dropout = 0, LSTM_dropout = 0):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 300)
            self.lstm = nn.LSTM(input_size = 300, hidden_size = hidden_size, num_layers = n_layers, bidirectional = bidirectional, dropout = LSTM_dropout, batch_first = False)
            if bidirectional:
                self.linear = nn.Linear(hidden_size*2, 103)
            else:
                self.linear = nn.Linear(hidden_size, 103)
            self.dropout = nn.Dropout(linear_dropout)

        def forward(self, x):
            embeds = self.embedding(x)
            embeds = self.dropout(embeds)
            lstm_out, _ = self.lstm(embeds)
            out = self.linear(lstm_out[-1,:,:])
            return out

    #Create model and update model embeddings
    model = LSTM(vocab_size = torch.tensor(embeds.shape[0], device = device), hidden_size = hidden_size, n_layers = 2, bidirectional = True, linear_dropout = 0.4, LSTM_dropout = 0.4)
    model.embedding.weight.data.copy_(torch.Tensor(embeds["embedding"]))

    #Text and label processing for batching with dataloader
    word_to_index = {word: index for index, word in enumerate(embeds["word"])}
    text_transform = lambda x: [word_to_index[token] for token in x]

    train_dataset = CustomDataset(train)
    valid_dataset = CustomDataset(valid)
    indice_lengths_train = [(i, len(s)) for i, s in enumerate(train["tokens"])]
    indice_lengths_valid = [(i, len(s)) for i, s in enumerate(valid["tokens"])]
    
    #Delete unneeded variable to free up some memory
    del embeds
    del train
    del valid

    #Transform words to indices and pad batches. Code adapted from torchtext migration guide:
    #https://colab.research.google.com/github/pytorch/text/blob/master/examples/legacy_tutorial/migration_tutorial.ipynb
    def collate_batch(batch):
        label_list, text_list = [], []
        for text, label in batch:
                label_list.append(label)
                processed_text = torch.tensor(text_transform(text))
                text_list.append(processed_text)
        return torch.tensor(label_list, dtype = torch.float).to(device), pad_sequence(text_list, padding_value=1).to(device)

    #Custom sampler for sampling groups with similar lengths. Code adapted from torchtext migration guide:
    #https://colab.research.google.com/github/pytorch/text/blob/master/examples/legacy_tutorial/migration_tutorial.ipynb
    class batch_sampler(Sampler):
        def __init__(self, indice_lengths):
            self.indice_lengths = indice_lengths
            self.pooled_indices = []

        def __iter__(self):
            random.shuffle(self.indice_lengths)
            self.pooled_indices = []
            # create pool of indices with similar lengths 
            for i in range(0, len(self.indice_lengths), BATCH_SIZE * 100):
                self.pooled_indices.extend(sorted(self.indice_lengths[i:i + BATCH_SIZE * 100], key=lambda x: x[1]))

            self.pooled_indices = [x[0] for x in self.pooled_indices]

            for i in range(0, len(self.pooled_indices), BATCH_SIZE):
                yield self.pooled_indices[i:i + BATCH_SIZE]

        def __len__(self) -> int:
            return int(len(self.indice_lengths)/BATCH_SIZE)

    train_dataloader = DataLoader(train_dataset, batch_sampler = batch_sampler(indice_lengths_train),
                                collate_fn=collate_batch, num_workers=0, pin_memory=False)
    valid_dataloader = DataLoader(valid_dataset, batch_sampler = batch_sampler(indice_lengths_valid),
                                collate_fn=collate_batch, num_workers=0, pin_memory=False)

    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01)

    #If weights True, weight each observation according to their proportion
    if weights:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=w_log)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    # --- Train Loop ---
    print("\nBeginning training")
    for epoch in range(50):
        start_time = time.time()
        epoch_loss = 0
        outputs = torch.empty((0,103)).to(device)
        targets = torch.empty((0,103)).to(device)

        model.train()
        batch_fraction = int(len(train_dataloader)/10)
        len_dataloader = len(train_dataloader)
        for batch_num, batch in enumerate(train_dataloader):

            optimizer.zero_grad()

            output = model(batch[1])
            loss = criterion(output, batch[0])

            outputs = torch.cat((outputs, output))
            targets = torch.cat((targets, batch[0]))

            epoch_loss += loss

            loss.backward()
            optimizer.step()

            if batch_num % batch_fraction == 0:
                print(f"Epoch {epoch+1:02}: {round((batch_num+1)/len_dataloader*100, 0)}% done", end="\r", flush=True)
        
        #Tensors to numpy arrays
        outputs = (torch.sigmoid(outputs) >= 0.5).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()

        #Calculate scores
        accuracy = metrics.accuracy_score(targets, outputs)
        precision, recall, f1_score_macro, _ = metrics.precision_recall_fscore_support(targets, outputs, average='macro', zero_division = 0)
        epoch_loss_avg = epoch_loss/(batch_num+1)

        print("\n--- Training loss and accuracy: ---")
        print(f"Epoch loss: {epoch_loss_avg}")
        print(f"Accuracy Score = {accuracy}")
        print(f"Precision (Macro) = {precision}")
        print(f"Recall (Macro) = {recall}")
        print(f"F1 Score (Macro) = {f1_score_macro}")

        end_time = time.time()

        #save results
        results = np.hstack((epoch+1, epoch_loss_avg.cpu().detach().numpy(), accuracy, precision, recall, f1_score_macro, end_time - start_time))
        np.save(f"./Grid search results/Train/{name}Train_epoch{epoch+1}", np.array(results))

        epoch_loss = 0
        outputs = torch.empty((0,103)).to(device)
        targets = torch.empty((0,103)).to(device)

        model.eval()
        with torch.no_grad():
            for batch_num, batch in enumerate(valid_dataloader):
                output = model(batch[1])
                loss = criterion(output, batch[0])

                outputs = torch.cat((outputs, output))
                targets = torch.cat((targets, batch[0]))

                epoch_loss += loss

            #Tensors to numpy arrays
            outputs = (torch.sigmoid(outputs) >= 0.5).cpu().detach().numpy()
            targets = targets.cpu().detach().numpy()

            #Calculate scores
            accuracy = metrics.accuracy_score(targets, outputs)
            precision, recall, f1_score_macro, _ = metrics.precision_recall_fscore_support(targets, outputs, average='macro', zero_division = 0)
            epoch_loss_avg = epoch_loss/(batch_num+1)

            print("\n--- Validation loss and accuracy: ---")
            print(f"Epoch loss: {epoch_loss_avg}")
            print(f"Accuracy Score = {accuracy}")
            print(f"Precision (Macro) = {precision}")
            print(f"Recall (Macro) = {recall}")
            print(f"F1 Score (Macro) = {f1_score_macro}")

            #save results
            results = np.hstack((epoch+1, epoch_loss_avg.cpu().detach().numpy(), accuracy, precision, recall, f1_score_macro))
            np.save(f"./Grid search results/Valid/{name}Valid_epoch{epoch+1}", np.array(results))

        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n')


if __name__ == '__main__':
    #name is used for saving results
    run_model(name = "no_weights_BS_16", batch_size = 16, drop_unknown = True, hidden_size = 200, weights = False)