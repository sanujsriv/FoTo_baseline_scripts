import gc,torch
from sklearn.feature_extraction.text import CountVectorizer
import math,os
import numpy as np
import nltk
from nltk.corpus import stopwords
from utils import load_obj_pkl5,load_obj,save_obj,vocab_filtered_data,decompress_pickle
from nltk import word_tokenize

# nltk.download('punkt')
paper = 'emnlp2022'
model='WTM'
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
  

def get_data_label_vocab(data,lables):
  
  preprossed_data = data
  train_label = lables
  vectorizer = CountVectorizer(min_df=0,dtype=np.uint8)
  train_vec = vectorizer.fit_transform(preprossed_data).toarray()
  vocab = vectorizer.vocabulary_
  id_vocab = dict(map(reversed, vocab.items()))
  print("train_input_vec_shape :",train_vec.shape)

  return train_vec,train_label,id_vocab,preprossed_data,vocab
  
# def get_data_label_vocab(data,lables):
#   preprossed_data = data
#   train_label = lables

#   vectorizer = CountVectorizer(min_df=0,dtype=np.uint8)
#   train_vec = vectorizer.fit_transform(preprossed_data).toarray()
#   vocab = vectorizer.vocabulary_
#   id_vocab = dict(map(reversed, vocab.items()))
#   nonzeros_indexes = np.where(train_vec.any(1))[0]
#   train_vec_non_zeros = [train_vec[i] for i in nonzeros_indexes]
#   preprossed_data_non_zeros = [preprossed_data[i] for i in nonzeros_indexes]
#   preprossed_data_non_zeros_vocab_filtered = [vocab_filtered_data(preprossed_data[i],vocab) for i in nonzeros_indexes]

#   train_label = [train_label[i] for i in nonzeros_indexes]

#   train_vec = np.asanyarray(train_vec_non_zeros)
#   tensor_train_w = torch.from_numpy(train_vec).float()
#   # num_input = train_vec.shape[1]
#   # keywords = get_keywords(d_data)
#   train_label = np.asarray(train_label)
#   # for v in vocab: embeddings.pop(v)
#   print("train_input_vec_shape :",train_vec.shape)
#   # print(train_label)
#   # print(vocab)
#   return tensor_train_w,train_label,id_vocab,preprossed_data_non_zeros,vocab