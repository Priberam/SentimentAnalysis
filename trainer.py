
import os
import sys, traceback
import argparse
import json
import array
from tqdm import tqdm
import numpy as np

import torch
from torchtext import data, vocab
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from nn_modules import GaussianNoise, RNN
from sklearn.metrics import f1_score
import random

import __main__
train_config=__main__.train_config

def train():
    EMBEDDINGS_PATH = args.embeddings
    # Load pre-trained embeddings
    itos, vectors, dim = [], array.array(str('d')), None
    count_embeddings= 0
    max_vectors=1000000
    with open(EMBEDDINGS_PATH, encoding="utf8") as fp:    
        for line in fp:
            count_embeddings+=1
    with open(EMBEDDINGS_PATH, encoding="utf8") as fp:    
        for index, line in enumerate(tqdm(fp, total=count_embeddings)):
            if index > max_vectors:
                break
            # Explicitly splitting on " " is important, so we don't
            # get rid of Unicode non-breaking spaces in the vectors.
            entries = line.rstrip().split(" ")

            word, entries = entries[0], entries[1:]
            if dim is None and len(entries) > 1:
                dim = len(entries)
            elif dim != len(entries):
                raise RuntimeError(
                    "Vector for token {} has {} dimensions, but previously "
                    "read vectors have {} dimensions. All vectors must have "
                    "the same number of dimensions.".format(word, len(entries), dim))

            vectors.extend(float(x) for x in entries)
            itos.append(word)
                
    #id to string: itos
    #string to id: stoi
    stoi = {word: i for i, word in enumerate(itos)}
    vectors = torch.Tensor(vectors).view(-1, dim)


    # Load datasets
    TEXT = data.Field()
    LABEL = data.RawField()
    train = data.TabularDataset(path=args.train_dataset, format='tsv',
                                fields=[('text', TEXT),
                                        ('label', LABEL)])


    valid = data.TabularDataset(path=args.dev_dataset, format='tsv',
                                fields=[('text', TEXT),
                                        ('label', LABEL)])

    test = data.TabularDataset(path=args.test_dataset, format='tsv',
                               fields=[('text', TEXT),
                                       ('label', LABEL)])

    TEXT.build_vocab(train)
    TEXT.vocab.set_vectors(stoi, vectors, dim)



    #ClassResclalingWeights = torch.Tensor([x, y, z])
    #labels_vect = {'positive': 0, 'neutral': 1, 'negative': 2}


    # parameters from DataStories system
    # http://aclweb.org/anthology/S17-2126
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 150
    OUTPUT_DIM = 3
    N_LAYERS = 2
    BIDIRECTIONAL = True
    ATTENTION = "simple" #None, "simple"
    NOISE = 0.3
    FINAL_LAYER=False
    DROPOUT_FINAL=0.5
    DROPOUT_ATTENTION=0.5
    DROPOUT_WORDS=0.3
    DROPOUT_RNN=0.3
    DROPOUT_RNN_U=0.3
    LR=0.001
    GRAD_CLIP=1

    N_EPOCHS = 15
    BATCH_SIZE = 128


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

    criterion = nn.NLLLoss()#(weight=ClassResclalingWeights)
    criterion = criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)


    # Batches!    
    train_iter, valid_iter, test_iter = data.BucketIterator.splits(
        (train, valid, test), 
        batch_size=BATCH_SIZE, 
        sort_key=lambda x: len(x.text),
        shuffle=True,
        repeat=False)

    labels_vect = {'positive': 0, 'neutral': 1, 'negative': 2}
    

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
        torch.save(model, f'Models/english-epoch{epoch+1:02}-{val_f1:.3f}.pt')
        if val_f1 >highest_value_f1:
            highest_value_f1=val_f1
            best_epoch=epoch
            best_model=f'Models/english-epoch{epoch+1:02}-{val_f1:.3f}.pt'


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
