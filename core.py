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
from collections import defaultdict

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

def _default_unk_index():
    return 0

# Load pre-trained embeddings
def load_embeddings(embeddings_path):
    itos, vectors, dim = [], array.array(str('d')), None
    count_embeddings= 0
    max_vectors=1000000
    with open(embeddings_path, encoding="utf8") as fp:    
        for line in fp:
            count_embeddings+=1
    with open(embeddings_path, encoding="utf8") as fp:    
        for index, line in enumerate(tqdm(fp, total=min(max_vectors,count_embeddings))):
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
    stoi = defaultdict(_default_unk_index)
    # stoi is simply a reverse dict for itos
    stoi.update({token: i for i, token in enumerate(itos)})
    return itos, stoi, vectors, dim


class Instance(object):
    def __init__(self, 
                name,
                language,
                embeddings_path, 
                preprocessing_style,
                model_path,
                labels):
        self.name = name
        self.language = language
        self.embeddings_path = embeddings_path
        self.preprocessing_style = preprocessing_style
        self.model_path = model_path
        self.labels={v:k for k,v in labels.items()}

def load_instances(config, instances):
    for instance_config in config["REST_instances"]:
        instance = Instance(instance_config["name"],
                            instance_config["language"],
                            instance_config["embeddings_path"],
                            instance_config["preprocessing_style"],
                            instance_config["model_path"],
                            instance_config["labels"])

        instance.text_processor = TextPreProcessor(
            # terms that will be normalized
            normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                'time', 'url', 'date', 'number'],
            # terms that will be annotated
            annotate={"hashtag", "allcaps", "elongated", "repeated",
                'emphasis', 'censored'},
            fix_html=True,  # fix HTML tokens
    
            # corpus from which the word statistics are going to be used 
            # for word segmentation 
            segmenter=instance_config["preprocessing_style"], 
    
            # corpus from which the word statistics are going to be used 
            # for spell correction
            corrector=instance_config["preprocessing_style"], 
    
            unpack_hashtags=True,  # perform word segmentation on hashtags
            unpack_contractions=True,  # Unpack contractions (can't -> can not)
            spell_correct_elong=False,  # spell correction for elongated words
    
            # select a tokenizer. You can use SocialTokenizer, or pass your own
            # the tokenizer, should take as input a string and return a list of tokens
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
            # list of dictionaries, for replacing tokens extracted from the text,
            # with other expressions. You can pass more than one dictionaries.
            dicts=[emoticons]
        )

        instance.itos, instance.stoi, instance.vectors, instance.embeddings_size = \
            load_embeddings(instance.embeddings_path)

        instance.text = data.Field()
        instance.text.build_vocab(instance.itos)
        instance.text.vocab.set_vectors(instance.stoi, instance.vectors, instance.embeddings_size)
    
        instance.model = torch.load(instance.model_path, map_location='cpu' if not cuda_available else None)
        instance.model = instance.model.eval()   
        instances[instance_config["name"]]=instance

def batch_predict(instance, batch_text, batch_size=128):
    model = instance.model
    text_processor = instance.text_processor
    processed_batch_texts = [" ".join(text_processor.pre_process_doc(text)) for text in batch_text]
    
    Dataset_input = [data.Example.fromlist(data=[processed_batch_text], fields=[('text', instance.text)]) for processed_batch_text in processed_batch_texts]
    batch_data = data.Dataset(Dataset_input, fields=[('text', instance.text)])

  
    (batch_data_iter,) = data.BucketIterator.splits([batch_data], 
        batch_size=batch_size, 
        sort_key=lambda x: len(x.text),
        shuffle=False,
        repeat=False)

    with torch.no_grad():  
        pred_list = []
        for batch in batch_data_iter:
            predictions = model(batch.text)
            top_n, top_i = predictions.topk(1)
            pred = top_i.squeeze(-1).cpu().numpy()
            pred_list.append(pred)
        pred = np.hstack(pred_list)
        return [instance.labels[p] for p in pred]

def predict(instance, text):
    return batch_predict(instance, [text])[0]