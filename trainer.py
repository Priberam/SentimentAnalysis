
import os
import sys, traceback
import argparse
import json
import array
from tqdm import tqdm
import numpy as np

import torch
cuda_available = torch.cuda.is_available()
from torchtext import data, vocab
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from nn_modules import GaussianNoise, RNN
from sklearn.metrics import f1_score
import random

import core
import __main__
train_config=__main__.train_config

def train_all():
    for instance_to_train in train_config["to_train_instances"]:
        train(instance_to_train)

def train(config):
    # Load pre-trained embeddings
    itos, stoi, vectors, dim = core.load_embeddings(config["embeddings_path"])
    #id to string: itos
    #string to id: stoi
    vectors = torch.Tensor(vectors).view(-1, dim)

    # Load datasets
    TEXT = data.Field()
    LABEL = data.RawField()
    train = data.TabularDataset(path=config["train_dataset_path"], format='tsv',
                                fields=[('text', TEXT),
                                        ('label', LABEL)])

    valid = data.TabularDataset(path=config["dev_dataset_path"], format='tsv',
                                fields=[('text', TEXT),
                                        ('label', LABEL)])

    test = data.TabularDataset(path=config["test_dataset_path"], format='tsv',
                               fields=[('text', TEXT),
                                       ('label', LABEL)])
    TEXT.build_vocab(train)
    TEXT.vocab.set_vectors(stoi, vectors, dim)

    labels_vect = config["labels"]

    # parameters from DataStories system
    # http://aclweb.org/anthology/S17-2126
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = dim
    HIDDEN_DIM = config["hidden_dim"]
    OUTPUT_DIM = len(labels_vect.keys())
    N_LAYERS = config["num_layers"]
    BIDIRECTIONAL = config["bidirectional"]
    ATTENTION = config["attention"]
    NOISE = config["noise"]
    FINAL_LAYER= config["final_layer"]
    DROPOUT_FINAL= config["dropout_final"]
    DROPOUT_ATTENTION= config["dropout_attention"]
    DROPOUT_WORDS= config["dropout_words"]
    DROPOUT_RNN= config["dropout_rnn"]
    DROPOUT_RNN_U= config["dropout_rnn_u"]
    LR= config["lr"]
    GRAD_CLIP= config["grad_clip"]
    N_EPOCHS = config["n_epochs"]
    BATCH_SIZE = config["batch_size"]

    model = RNN(INPUT_DIM, 
                EMBEDDING_DIM, 
                HIDDEN_DIM, 
                OUTPUT_DIM, 
                N_LAYERS, 
                BIDIRECTIONAL,     
                ATTENTION,            
                DROPOUT_FINAL,
                DROPOUT_ATTENTION,
                DROPOUT_WORDS,
                DROPOUT_RNN,
                DROPOUT_RNN_U,
                NOISE,
                FINAL_LAYER)

    # copy the pre-trained embeddings. Just the embeddings for the vocabulary in TEXT (train,dev and test)
    model.embedding.weight.data.copy_(TEXT.vocab.vectors)

    device = torch.device('cuda' if cuda_available else 'cpu')
    model = model.to(device)

    criterion = nn.NLLLoss()
    criterion = criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Batches!    
    train_iter, valid_iter, test_iter = data.BucketIterator.splits(
        (train, valid, test), 
        batch_size=BATCH_SIZE, 
        sort_key=lambda x: len(x.text),
        shuffle=True,
        repeat=False)
    
    highest_value_f1=0.0
    best_epoch=0
    best_model=""
    for epoch in range(N_EPOCHS):

        # Train dataset
        model = model.train()
        train_loss = []    
        pred_list = []
        labels_list = []
        for batch in train_iter:
        
            optimizer.zero_grad()
        
            predictions = model(batch.text)
        
            labels = np.zeros([len(batch.label)])
            for index, label in enumerate(batch.label):    
                labels[index] = labels_vect[label]                                
            labels_list.append(labels)        
        
            labels = torch.Tensor(labels).type(torch.LongTensor).cuda() if cuda_available else \
                     torch.Tensor(labels).type(torch.LongTensor)    
            loss = criterion(predictions, labels)
            loss.backward() 
        
            # grad clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        
            train_loss.append(loss.item())
        
            top_n, top_i = predictions.topk(1)
            pred = top_i.squeeze(-1)
            pred_list.append(pred.cpu().numpy())

        pred = np.hstack(pred_list)
        labels = np.hstack(labels_list)
    
        # Official f-score from SemEval. Same as used by data stories
        # https://github.com/cbaziotis/datastories-semeval2017-task4/blob/  faeee5fd1c2cf38e32179a4676dff53ae01adfa8/models/nn_task_message.py#L107
        train_f1 = f1_score(labels, pred, average='macro', labels=[labels_vect['positive'], labels_vect['negative']])
        
        # Evaluation dataset. 
        # model.eval() disables dropout, for example.    
        model = model.eval()    
        pred_list = []
        labels_list = []    
        with torch.no_grad():    
            for batch in valid_iter:

                predictions = model(batch.text)
                top_n, top_i = predictions.topk(1)
                pred = top_i.squeeze(-1)
                pred_list.append(pred.cpu().numpy())
            
                labels = np.zeros([len(batch.label)])
                for index, label in enumerate(batch.label):    
                    labels[index] = labels_vect[label]
                labels_list.append(labels)
            

        pred = np.hstack(pred_list)
        labels = np.hstack(labels_list)    
        val_f1 = f1_score(labels, pred, average='macro', labels=[labels_vect['positive'], labels_vect['negative']])
    
        print(f'Epoch: {epoch+1:02}, Train Loss: {sum(train_loss)/len(train_loss):.3f}, Train F1:   {train_f1:.3f}, Val F1:{val_f1:.3f}')
        if val_f1 >highest_value_f1:
            #save current best model
            torch.save(model, 'Models/'+config["name"]+f'-epoch{epoch+1:02}-{val_f1:.3f}.pt')
            highest_value_f1=val_f1
            best_epoch=epoch
            #delete previous best model
            if best_model!="" and os.path.isfile(best_model):
                os.remove(best_model)
            best_model='Models/'+config["name"]+f'-epoch{epoch+1:02}-{val_f1:.3f}.pt'

    print("\n\nbest_model: ", best_model, "\n")

    if config["save_in_REST_config"]==True:
        if os.path.isfile(config["target_REST_config_path"]):   
            with open(config["target_REST_config_path"], 'r') as fp:
                target_config = json.load(fp) 
            new_instance={}
            new_instance["name"]=config["name"]
            new_instance["language"]=config["language"]
            new_instance["embeddings_path"]=config["embeddings_path"]
            new_instance["preprocessing_style"]=config["preprocessing_style"]
            new_instance["labels"]=config["labels"]
            new_instance["model_path"]=best_model

            found=None
            for index, config_inst in enumerate(target_config["REST_instances"]):
                if config_inst["name"] ==  new_instance["name"]:
                    found=index
                    break
            if found != None:
                del(target_config["REST_instances"][index])
            target_config["REST_instances"].append(new_instance)
            
            with open(config["target_REST_config_path"], 'w') as fp:
                json.dump(target_config, fp, indent=2)

    # Evaluation dataset. 
    model = torch.load(best_model)
    model = model.eval()
    pred_list = []
    labels_list = []
    with torch.no_grad():    
        for batch in test_iter:

            predictions = model(batch.text)
            top_n, top_i = predictions.topk(1)
            pred = top_i.squeeze(-1).cpu().numpy()
            pred_list.append(pred)

            labels = np.zeros([len(batch.label)])
            for index, label in enumerate(batch.label):    
                labels[index] = labels_vect[label]        
            labels_list.append(labels)
        
        pred = np.hstack(pred_list)
        labels = np.hstack(labels_list)

        f1 = f1_score(labels, pred, average='macro', labels=[labels_vect['positive'], labels_vect['negative']])
        # should achieve 0.675 (state-of-the-art)
        print(f'F1-score: {f1:.3f}')
