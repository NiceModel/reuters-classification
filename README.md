# Multi-label classification on the Reuters RCV1 corpus

In this project we experiment on two different deep learning architectures in an effort to do multi-label classification on the Reuters corpus. We built two models to tackle the classification challenge. One of these was an LSTM model and the other a BERT model.

## Structure of this repository

The original data was given as a collection of .xml files. These files have been parsed and preprcosessed with the tools found in the folder `preprocessing`.

As we did not want to bloat the version control system with large files that are not subject to change, the data folder has to be added manually to the project in order to reproduce our work. To do this, complete the following steps:  

- Download our datafolder from https://www.dropbox.com/s/4gsztwl2xf7fuau/data.tar.gz?dl=0 to the root of this project
- Untar the file in the project root. In *nix based systems you can do this by running `tar -zxvf data.tar.gz`

After these steps the data folder structure should be as follows:

```
data
├── codecounts.csv
├── rest_bert.csv
├── test.csv
├── topic_codes.txt
├── train.csv
├── valid.csv
└── valid_bert.csv
```

All the scripts needed to build, train and evaluate the models can be found in the folders 'LSTM' and 'BERT'.

## Working with BERT

To train BERT for producing predictions for the test set obtained by course administrators complete the following steps:  
- Go inside `BERT` folder
- Run `python bert_data_loader_real_data_strat.py`
- As a result a model file with timestamp and a `.pth` extension will be created

To predict BERT for the test set obtained by course administrators complete the following steps:  
- Go inside `BERT` folder
- Make sure the test data exists in the folder with a name `final_test.csv`
- Run `python predict.py`
- The process produces csv file with the predictions
## Working with LSTM

To train LSTM on the training set and validate against validation set, you need train.csv, valid.csv and codecounts.csv in the data folder. You also need to run the Embeddings/Embeddings.py to create the embeddings LSTM uses. After this the LSTM/LSTM.py script should work correctly.
