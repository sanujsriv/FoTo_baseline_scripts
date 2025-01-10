import seaborn as sb
import numpy as np  

import torch.nn as nn
import plotly.graph_objects as go
from collections import Counter
import bz2
import _pickle as cPickle
import pickle5
import pickle
from termcolor import colored
import torch
import math
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import matplotlib.pyplot as plt
from time import time
import gc
import pandas as pd
from nltk import bigrams
import random

seed=2022
random.seed(2022)
np.random.seed(seed=2022)


def DESM_score_Corpus(query_list, train_vec, vocab, embeddings):
  sim_list = torch.zeros(train_vec.shape[0])
  index = 0
  word_list = np.asarray(sorted(vocab))
  for d in train_vec:
    words_in_d = np.where(d>0)[0]
    all_words_tensor = get_embedding_tensor(np.repeat(word_list[words_in_d],d[words_in_d]),embeddings)
    doc_bar = (all_words_tensor/torch.norm(all_words_tensor,dim=1).unsqueeze(-1)).sum(0) / d.sum(0)
    D_bar = doc_bar.unsqueeze(0).expand(len(query_list),doc_bar.shape[0])
    q = get_embedding_tensor(query_list,embeddings)
    norm_div = torch.norm(q,dim=1) * torch.norm(D_bar,dim=1)
    sim_list[index]=(torch.mm(q,D_bar.T)[:,0]/norm_div).sum()/len(query_list)
    index +=1
  return sim_list

cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-8)

def cosine_keywords(keywords,words_tensor,embeddings):
  all_keywords_score = []
  for k in keywords:
    keyword_torch = torch.from_numpy(embeddings[k])
    keyword_torch = keyword_torch.unsqueeze(0).expand(words_tensor.shape[0],words_tensor.shape[1])
    cosine_sim_score = cos_sim(keyword_torch,words_tensor)
    all_keywords_score.append(cosine_sim_score)
  return all_keywords_score

def vocab_filtered_data(doc,vocab):
  doc = word_tokenize(doc)
  doc = filter(lambda x: x in vocab, doc) 
  doc = ' '.join(e for e in doc)
  return doc


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_obj_pkl5(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle5.load(f)

def compressed_pickle(data,title):
  with bz2.BZ2File(title + '.pbz2', 'w') as f: 
    cPickle.dump(data, f)

def decompress_pickle(file):
 data = bz2.BZ2File(file+'.pbz2', 'rb')
 data = cPickle.load(data)
 return data

  
cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-8)

def generate_co_occurrence_matrix(corpus):
    vocab = set(corpus)
    vocab = list(vocab)
    vocab_index = {word: i for i, word in enumerate(vocab)}
 
    # Create bigrams from all words in corpus
    bi_grams = list(bigrams(corpus))
 
    # Frequency distribution of bigrams ((word1, word2), num_occurrences)
    bigram_freq = nltk.FreqDist(bi_grams).most_common(len(bi_grams))
 
    # Initialise co-occurrence matrix
    # co_occurrence_matrix[current][previous]
    co_occurrence_matrix = np.zeros((len(vocab), len(vocab)))
 
    # Loop through the bigrams taking the current and previous word,
    # and the number of occurrences of the bigram.
    for bigram in bigram_freq:
        current = bigram[0][1]
        previous = bigram[0][0]
        count = bigram[1]
        pos_current = vocab_index[current]
        pos_previous = vocab_index[previous]
        co_occurrence_matrix[pos_current][pos_previous] = count
    co_occurrence_matrix = np.matrix(co_occurrence_matrix)
 
    # return the matrix and the index
    return co_occurrence_matrix, vocab_index


def flatten_list(user_list): return [item for sublist in user_list for item in sublist]
def get_embedding_tensor(word_list,embeddings): return torch.tensor(np.asarray([embeddings[w] for w in word_list]))
# def get_doc_word_embeddings(id_vocab,embeddings):
#   sorted_id_word_vocab = sorted(id_vocab.items(), key=lambda x: x[1]) ### alphabetically sorted
#   word_list = [s[1] for s in sorted_id_word_vocab]
#   words_tensor = get_embedding_tensor(word_list,embeddings)
#   embedding_tensor_sorted_alp = get_embedding_tensor(word_list,embeddings)
#   emb_size = embedding_tensor_sorted_alp.shape[1]
#   return embedding_tensor_sorted_alp,emb_size

def get_topwords(beta, id_vocab,topwords):
    topic_indx = 0
    topwords_topic = []
    topic_topwords = topwords
    for i in range(len(beta)):
        topwords_topic.append( str(topic_indx)+": "+ " ".join([id_vocab[j] for j in beta[i].argsort()[:-topic_topwords - 1:-1]]))
        topic_indx+=1
    return topwords_topic

def plot_loss(y,name):
  figure = go.Figure()
  figure.add_trace(go.Scatter(x=[i for i in range(1,epochs+1)], y=y,mode='lines',name=name))
  figure.show(renderer='colab')


def print_Topics(beta,id_vocab,no_of_topwords):
  print("---"*10)
  topword_topics = get_topwords(beta, id_vocab,no_of_topwords)
  topword_topics_list=[]
  for topwords in topword_topics:
      topword_topics_list.append(topwords.split())
      print(topwords)
  print("---"*10) 
 
    
def getall_tensor_size(): 
  for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size(),get_mem_size(obj)) 
    except:
        pass
