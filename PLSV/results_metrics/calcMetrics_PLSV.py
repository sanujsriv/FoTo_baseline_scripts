# STTM TSNE (FoTo)

import seaborn as sb
import numpy as np  
import os
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
import itertools

from sklearn.metrics import average_precision_score as AP_score
from sklearn.metrics import precision_recall_curve,auc
from sklearn.neighbors import KNeighborsClassifier
# from utils import flatten_list,get_topwords,get_embedding_tensor,cosine_keywords

cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-8)

def flatten_list(user_list): return [item for sublist in user_list for item in sublist]
def get_embedding_tensor(word_list,embeddings): return torch.tensor(np.asarray([embeddings[w] for w in word_list]))

def list_of_tensors_to_tensor(loT):
  stacked_tensor = torch.stack(loT)
  return stacked_tensor

def toT(a): return torch.tensor(a)

def cosine_keywords(keywords,words_tensor,embeddings):
  all_keywords_score = []
  for k in keywords:
    keyword_torch = torch.from_numpy(embeddings[k])
    keyword_torch = keyword_torch.unsqueeze(0).expand(words_tensor.shape[0],words_tensor.shape[1])
    cosine_sim_score = cos_sim(keyword_torch,words_tensor)
    all_keywords_score.append(cosine_sim_score)
  return all_keywords_score

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

def get_bigram_coocurring_word_list(data_preprocessed,keywords):
  nlargestVal =  20
  text_data = [d.split(' ') for d in data_preprocessed]
  data = list(itertools.chain.from_iterable(text_data))
  matrix, vocab_index = generate_co_occurrence_matrix(data)
  
  data_matrix = pd.DataFrame(matrix, index=vocab_index,
                              columns=vocab_index)

  bigram_coocurring_word_list = []

  for k in keywords: 
    bigram_coocurring_word = np.array(data_matrix.nlargest(nlargestVal, k, keep='first').index)
    bigram_coocurring_word_list.extend(bigram_coocurring_word)
  return bigram_coocurring_word_list

## Quantitative

## 1) KNN
## Quantitative (clustering quality of visualization)

def cal_knn(coordinate, label):
    output = []
    for n_neighbors in [10, 20, 30, 40, 50]:
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        neigh.fit(coordinate, label)
        output.append(neigh.score(coordinate, label))
    return output

## 2) AUPR
## Quantitative (average precision score for documents close to any of the keywords)

def cal_AUPR_f(num_docs,num_keyword,num_coordinate,relv_docs_in_vis_idx,x_list,query_center,not_pseudo_idx):
  map_true_ranking = torch.zeros(num_docs) 
  map_true_ranking[relv_docs_in_vis_idx] = 1.0
  map_true_ranking = map_true_ranking[not_pseudo_idx]

  x_list = x_list[not_pseudo_idx]
  
  doc_query_size = (len(x_list), num_keyword, num_coordinate)

  x_q = toT(x_list).view(len(x_list),1,num_coordinate).expand(doc_query_size)
  q_x = toT(query_center).view(1, num_keyword,num_coordinate).expand(doc_query_size)
  dist_x_q = (x_q - q_x).pow(2).sum(-1)
  minDist_x_q = torch.min(dist_x_q,-1).values
  
  precision, recall, _ = precision_recall_curve(map_true_ranking,-minDist_x_q)
  map_true_ranking = map_true_ranking.numpy()
  minDist_x_q = minDist_x_q.numpy()
  return map_true_ranking,minDist_x_q,auc(recall,precision)


## 3) sum of cosine
def get_cosine_sum_topics(topics_wordlist,embeddings,keywords):
  np_topics_wordsScores_all = []
  for topics in topics_wordlist:
    topics_wordtensors = get_embedding_tensor(topics,embeddings)
    topics_wordsScores = cosine_keywords(keywords,topics_wordtensors,embeddings)
    np_topics_wordsScores = list_of_tensors_to_tensor(topics_wordsScores)
    np_topics_wordsScores_all.append(np_topics_wordsScores)
  
  sum_avg_cos = []
  for np_topics_wordsScores in np_topics_wordsScores_all:
    sum_avg_cos.append((np_topics_wordsScores.mean(-1)).sum(-1).item())
  sum_avg_cos = np.array(sum_avg_cos)
  return sum_avg_cos

# map_true_ranking, minDist_x_q, aupr = cal_AUPR_f(len(X),len(keywords_as_labels),2,relv_docs_in_vis_idx,X,query_center,not_pseudo_idx)
# knn=cal_knn(X,labels)
# sum_avg_cos = get_cosine_sum_topics(topics_wordlist,embeddings,keywords)

## Qualitative

######### Focused Visualization ######## 
## Qualitative (how close are the relv docs to the ground truth)
def get_colored_topics(topics_wordlist,keywords,bigram_coocurring_word_list,extended_keywords_list):
  flattened_ext_keylist = flatten_list(extended_keywords_list)
  for t in range(len(topics_wordlist)):
    topic=topics_wordlist[t]  
    for i in range(len(topic)):
      if topic[i] in keywords: 
        topic[i] = "<r> "+topic[i]+" </r>"
      elif topic[i] in bigram_coocurring_word_list:
        topic[i] = "<b> "+topic[i]+" </b>"
      elif topic[i] in flattened_ext_keylist:
        topic[i]= "<g> "+topic[i]+" </g>"
    topics_wordlist[t] = ' '.join(topic)
  return topics_wordlist


def plot_relv_irrelv_docs(model_name,filename,zx1,zx2,l1,l2, zphi, query_center,query_words,hv_qwords,showtopic,keywords,lim,save,contour='No'):
       
    fig, ax = plt.subplots( figsize=(20, 20))
    # if contour=='yes':
    #    get_Contour(ax,zx,lim)
    label_colors_dict = {'direct': 'C1','indirect':'C2',
                         'relevant(T)': 'C1','irrelevant(F)': 'C2'}
    sb.scatterplot(ax=ax,x=zx1[:,0],y=zx1[:,1],hue=l1,palette=label_colors_dict,alpha=0.8,s=50)
    sb.scatterplot(ax=ax,x=zx2[:,0],y=zx2[:,1],hue=l2,palette=label_colors_dict,alpha=0.8,s=50)
    
    ax.set(ylim=(-lim,lim))
    ax.set(xlim=(-lim,lim))

    ax.text(query_center[0],query_center[1], 'X' ,c='black',weight='bold',fontsize=20)
    # ax.text(0,0, 'X' ,c='black')
    
    if hv_qwords:
      for i in range(len(query_words)):
        if (i==len(query_words)-1):
          ax.text(query_words[i][0],query_words[i][1], 'X'+keywords[i] ,c='black',weight='bold',fontsize=13)
        else:
          ax.text(query_words[i][0],query_words[i][1], 'X'+keywords[i] ,c='black',weight='bold',fontsize=13)

    if showtopic:      
      ax.scatter(zphi[:, 0], zphi[:, 1], alpha=1.0,  edgecolors='black', facecolors='none', s=30)

    for indx, topic in enumerate(zphi):
        ax.text(zphi[indx, 0], zphi[indx, 1], 'topic'+str(indx),fontsize=13)
    if save:
      plt.savefig(model_name+'_vis_'+filename+".png", bbox_inches='tight')


def get_docs_idx_in_vis(relv_docs,preprossed_data_non_zeros,doc_ids):
  d = {item: idx for idx, item in enumerate(preprossed_data_non_zeros)} 
  doc_ids_list = list(doc_ids)
  relv_docs_idx = [d.get(item) for item in relv_docs]
  docs_in_vis_idx = [doc_ids_list.index(r_i) for r_i in relv_docs_idx]
  assert (doc_ids[docs_in_vis_idx] == relv_docs_idx).all() == True
  return docs_in_vis_idx

def get_labels_dict(unique_labels):
  labels_dict = {}
  for l in unique_labels:
    labels_dict[l] = 'C'+str(unique_labels.index(l))
  return labels_dict

def plot_fig(model_name,zx, labels_list, zphi,lim,sorted_unique_labels,query_words,keywords,hv_qwords=True,showtopic=False
            ,bold_topics=True,remove_legend=False,show_axis=True,save=False,figname="plot"):
       
    fig, ax = plt.subplots( figsize=(20, 20))
    # if contour=='yes':
    #    get_Contour(ax,zx,lim)

    label_colors_dict = get_labels_dict(sorted_unique_labels+['keywords'])
    # sb.scatterplot(ax=ax,x=zx[:,0],y=zx[:,1],hue=labels_list,alpha=0.8,palette='deep')
    g = sb.scatterplot(ax=ax,x=zx[:,0],y=zx[:,1],hue=labels_list,alpha=0.8,palette=label_colors_dict,s=50)
    
    ax.set(ylim=(-lim,lim))
    ax.set(xlim=(-lim,lim))
    
    if showtopic:
      ax.scatter(zphi[:, 0], zphi[:, 1], alpha=1.0,  edgecolors='black', facecolors='none', s=30)
   
      for indx, topic in enumerate(zphi):
        if bold_topics:
          ax.text(zphi[indx, 0], zphi[indx, 1], 'topic'+str(indx),fontsize=13,fontweight='bold')
        else: ax.text(zphi[indx, 0], zphi[indx, 1], 'topic'+str(indx),fontsize=13)

    if hv_qwords:
      for i in range(len(query_words)):
        if (i==len(query_words)-1):
          ax.text(query_words[i][0],query_words[i][1], 'X'+keywords[i] ,c='black',weight='bold',fontsize=13)
        else:
          ax.text(query_words[i][0],query_words[i][1], 'X'+keywords[i] ,c='black',weight='bold',fontsize=13)

    plt.setp(g.get_legend().get_texts(), fontsize='20') # for legend text
    plt.setp(g.get_legend().get_title(), fontsize='20') # for legend title
    plt.tight_layout()
    
    if remove_legend:
      g.legend_.remove()
    if not show_axis:
      plt.axis('off')
    if save:
      plt.savefig(model_name+"_vis_"+figname+".png", bbox_inches='tight')
    ax.text(0,0, 'X' ,c='black',weight='bold',fontsize=20)
    return ax
    

from MulticoreTSNE import MulticoreTSNE as TSNE
import os
from time import time
import argparse
import numpy as np  
from sklearn.neighbors import KNeighborsClassifier
import pickle5
import bz2
import pickle
import _pickle as cPickle

def compressed_pickle(data,title):
  with bz2.BZ2File(title + '.pbz2', 'w') as f:
    cPickle.dump(data, f)

def decompress_pickle(file):
 data = bz2.BZ2File(file+".pbz2", 'rb')
 data = cPickle.load(data)
 return data

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_obj_pkl5(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle5.load(f)

model_name ='PLSV'
dataset_dir = '/home/grad16/sakumar/ICDM_Experiments_2022/dataset'
model_results_dir = '/home/student_no_backup/sakumar/ICDM_Experiments_2022/'+model_name
model_data_dir= '/home/grad16/sakumar/ICDM_Experiments_2022/dataset'


# data_names=['opinions_twitter'] # 'nfcorpus', 'opinions_twitter','searchsnippet','yahooanswers'
# num_topics = ['10']#,'20','30','40','50']
# qss = ['1']#,'2']
# # fracs = ['0.05']#,'0.1','0.15','0.2']
# fracs = ['0.2']
# runs = [1]#,2,3,4,5]

# for data_name in data_names:
#   for num_topic in num_topics:
#     for qs in qss:
#       for r in runs:
#         for frac in fracs:

parser = argparse.ArgumentParser(description='STTM')
parser.add_argument('--data_name', type=str, default='bbc', help='bbc,searchsnippet, wos')
parser.add_argument('--num_runs', type=str, default='1', help='# run')
parser.add_argument('--num_topic', type=str, default='10', help='number of topic')
parser.add_argument('--queryset', type=int, default=1, help='queryset used to run the model')
parser.add_argument('--frac', type=float, default=0.10, help='injection fraction')

args = parser.parse_args()

data_name = args.data_name
dtype = 'short'
r = str(args.num_runs)
qs = str(args.queryset)
frac = str(args.frac)
num_topic = str(args.num_topic) 

os.chdir(dataset_dir+'/content/data_'+data_name+"/short")
embeddings = load_obj_pkl5("embeddings_"+data_name+"_short")
print(model_name,data_name,num_topic,qs,frac,r)
os.chdir(dataset_dir+'/content/data_'+data_name+"/short/injected_docs/"+str(frac))
files = os.listdir('.')
if data_name == 'bbc' or data_name =="searchsnippet":
  data_dict = decompress_pickle(files[int(int(qs)%2)].split('.')[0]) # int(int(qs)%2) # int(qs)-1
else:
  data_dict = decompress_pickle(files[int(qs)-1].split('.')[0]) # int(int(qs)%2) # int(qs)-1
new_docs = data_dict['new_docs']
new_labels = data_dict['new_labels']
extended_keywords_list = data_dict['extended_keywords_list']
ground_truth_docs = data_dict['docs_injected']
ground_truth_labels = data_dict['labels_injected']
keywords = data_dict['query']

bigram_coocurring_word_list = get_bigram_coocurring_word_list(new_docs,keywords)
keywords_as_labels = []
if data_name =='yahooanswers':
  extended_keywords_list_joined = np.asarray([list(extended_keywords_list[x]) + (list(extended_keywords_list[x+1])) for x in range(len(extended_keywords_list)) if x%2==0])
  keywords_as_labels = np.asarray(['p{'+keywords[x]+'&'+keywords[x+1]+'}' for x in range(len(keywords)) if x%2==0] )
  for e in range(len(extended_keywords_list_joined)):
    new_docs = np.append(new_docs,' '.join(extended_keywords_list_joined[e]))
    new_labels = np.append(new_labels,keywords_as_labels[e])
  extended_keywords_list = extended_keywords_list_joined
else: 
  for ext_kws in extended_keywords_list:
    keyword=ext_kws[0]
    kws = ' '.join(ext_kws)
    pseudo_label = 'p{'+keyword+'}'
    new_docs = np.append(new_docs,kws)
    keywords_as_labels.append(pseudo_label)
    new_labels = np.append(new_labels,pseudo_label)

keywords_as_labels = np.asarray(keywords_as_labels)
save_dir_no_bkp = '/SavedOutput/'+data_name+"/short/topics_"+num_topic+"/qs_"+qs+"/run_"+str(r)+"/frac_"+frac
os.chdir(model_results_dir+save_dir_no_bkp)
# all_results = decompress_pickle(model_name+"_"+data_name+"_short_"+str(num_topic)+"_"+str(r)+"_all_results")
all_results = decompress_pickle(data_name+"_short_topics_"+str(num_topic)+"_qs_"+str(qs)+"_run_"+str(r)+"_frac_"+str(frac)+"_all_results")
all_metrics = {}

all_indices = all_results['all_indices']

doc_ids = [batch_ndx.numpy() for batch_ndx in all_indices]
doc_ids = np.asanyarray(flatten_list(doc_ids))

X = all_results['X_w_pseudo']
labels = new_labels[doc_ids]

pseudo_idx = np.argwhere(np.isin(labels,keywords_as_labels)).ravel()
not_pseudo_idx = [i for i in range(len(X)) if i not in pseudo_idx]
query_center = X[pseudo_idx]

if isinstance(ground_truth_docs[0],str): relv_docs = ground_truth_docs
else: relv_docs = flatten_list(ground_truth_docs)

# doc_id_in_seq = np.asanyarray([np.where(doc_ids==i)[0][0] for i in range(len(doc_ids))])
# X_original_seq = X[doc_id_in_seq]

d = {item: idx for idx, item in enumerate(new_docs)} 
doc_ids_list = list(doc_ids)
relv_docs_idx = [d.get(item) for item in relv_docs]
relv_docs_in_vis_idx = [doc_ids_list.index(r_i) for r_i in relv_docs_idx]

map_true_ranking, minDist_x_q, aupr = cal_AUPR_f(len(X),len(keywords_as_labels),2,relv_docs_in_vis_idx,X,query_center,not_pseudo_idx)
os.chdir(model_results_dir+save_dir_no_bkp)


all_metrics['X'] = X
all_metrics['labels'] = labels
all_metrics['pseudo_idx'] = pseudo_idx
all_metrics['not_pseudo_idx'] = not_pseudo_idx
all_metrics['query_center'] =query_center
all_metrics['keywords_as_labels'] = keywords_as_labels
model_topics = all_results['topics']
model_topics = [t.split(':')[1].strip() for t in model_topics]
all_metrics['topics'] = model_topics
topics_wordlist = [t.split(' ') for t in model_topics]
all_metrics['sum_avg_cos'] = get_cosine_sum_topics(topics_wordlist,embeddings,keywords)
all_metrics['KNN'] = cal_knn(X,labels)         
all_metrics['aupr'] = aupr
all_metrics['map_true_ranking'] = map_true_ranking
all_metrics['minDist_x_q'] = minDist_x_q
all_metrics['runtime'] = all_results['runtime']
all_metrics['colored_topics'] = get_colored_topics(topics_wordlist,keywords,bigram_coocurring_word_list,extended_keywords_list)
print(os.getcwd())

# sorted_unique_labels = sorted(set(labels))
# lim=15
# plot_fig('WTM',X[not_pseudo_idx],labels[not_pseudo_idx],all_results['phi'],lim,sorted_unique_labels,query_center,keywords_as_labels,keywords)
compressed_pickle(all_metrics,model_name+"_metrics_"+data_name+"_"+"numtopic_"+num_topic+"_run_"+str(r)+"_qs_"+qs+"_frac_"+frac)

# from here...
# PTM/searchsnippet/short/topic_50/qs1/0.2/run_1/results