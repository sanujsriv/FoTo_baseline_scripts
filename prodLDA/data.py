import gc,torch
from sklearn.feature_extraction.text import CountVectorizer
import math,os
import numpy as np
import nltk
from nltk.corpus import stopwords
from utils import load_obj_pkl5,load_obj,save_obj,vocab_filtered_data,decompress_pickle
from nltk import word_tokenize

paper = 'emnlp2022'
model='prodLDA'
home_dir = '/home/grad16/sakumar/'+paper+'/'+model
data_dir = '/home/grad16/sakumar/'+paper+'/dataset'

def load_data(data,dtype,data_dir,qs,ext,skipgram_embeddings):

  data = data.lower()
  dtype = dtype.lower()
  dir ='/content/data_'+data+'/'+dtype
  os.chdir(data_dir+dir)

  data_preprocessed=load_obj_pkl5("data_preprocessed_"+data+"_"+dtype)
  data_preprocessed_labels=load_obj_pkl5("data_preprocessed_labels_"+data+"_"+dtype)
  if skipgram_embeddings: embeddings=load_obj_pkl5("generated_embeddings_"+data+"_"+dtype)
  else: embeddings=load_obj_pkl5("embeddings_"+data+"_"+dtype)
  queries_data_dict = decompress_pickle("queries_"+data)#+"_"+str(th)
  qs = str(qs)
  keywords = queries_data_dict[qs]['query']
  extend_each_by = queries_data_dict[qs]['extend_each_by']
  extended_keywords_list = [[k] for k in keywords]
  os.chdir(home_dir)
  return data_preprocessed,data_preprocessed_labels,embeddings,data,keywords,extend_each_by,extended_keywords_list,queries_data_dict[qs]
  
def get_data_label_vocab(data,labels):
  
  preprossed_data = data
  train_label = labels
  vectorizer = CountVectorizer(min_df=0,dtype=np.uint8)
  train_vec = vectorizer.fit_transform(preprossed_data).toarray()
  vocab = vectorizer.vocabulary_
  id_vocab = dict(map(reversed, vocab.items()))
  print("train_input_vec_shape :",train_vec.shape)
  preprossed_data_non_zeros = list(preprossed_data)
  return train_vec,train_label,id_vocab,preprossed_data,vocab
