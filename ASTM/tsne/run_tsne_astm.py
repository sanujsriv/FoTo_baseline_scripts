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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score as AP_score
from sklearn.metrics import precision_recall_curve,auc
from sklearn.neighbors import KNeighborsClassifier
# from utils import flatten_list,get_topwords,get_embedding_tensor,cosine_keywords

model_name ="ASTM"
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

def cal_AUPR(num_keyword,num_coordinate,relv_docs_in_vis_idx,x_list,query_center):
  num_docs = len(x_list)
  map_true_ranking = torch.zeros(num_docs) 
  map_true_ranking[relv_docs_in_vis_idx] = 1.0

  doc_query_size = (num_docs, num_keyword, num_coordinate)

  x_q = toT(x_list).view(num_docs,1,num_coordinate).expand(doc_query_size)
  q_x = toT(query_center).view(1, num_keyword,num_coordinate).expand(doc_query_size)
  dist_x_q = (x_q - q_x).pow(2).sum(-1)
  minDist_x_q = torch.min(dist_x_q,-1).values
  # minDist_x_q = dist_x_q.sum(-1) ## SUM DISTANCE
   
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
      plt.savefig(model_name+"_"+figname+".png", bbox_inches='tight')
    # ax.text(query_center[0],query_center[1], 'X' ,c='black',weight='bold',fontsize=20)
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


paper='emnlp2022'
dataset_dir = '/home/grad16/sakumar/'+paper+'/dataset'
ASTM_results_dir = '/home/student_no_backup/sakumar/'+paper+'/ASTM/root'
data_dir = '/home/grad16/sakumar/'+paper+'/dataset/content'
num_topics = ['10','20','30','40','50']



parser = argparse.ArgumentParser(description='ASTM')
parser.add_argument('--data_name', type=str, default='bbc', help='bbc,searchsnippet, wos')
parser.add_argument('--num_runs', type=str, default='1', help='# run')
parser.add_argument('--queryset', type=int, default=1, help='queryset used to run the model')

args = parser.parse_args()

data_name = args.data_name
dtype = 'short'
r = str(args.num_runs)
qs = str(args.queryset)

all_topics_theta = []
all_topics_wordlist = []
for num_topic in num_topics:

  print(data_name,num_topic,qs,r)
  os.chdir(data_dir+'/data_'+data_name+"/short")
  dtype ='short'
  embeddings = load_obj_pkl5("embeddings_"+data_name+"_short")
  new_docs=load_obj_pkl5("data_preprocessed_"+data_name+"_"+dtype)
  data_preprocessed=load_obj_pkl5("data_preprocessed_"+data_name+"_"+dtype)
  new_labels=load_obj_pkl5("data_preprocessed_labels_"+data_name+"_"+dtype)
  queries_data_dict = decompress_pickle("queries_"+data_name)
  qs = str(qs)
  keywords = queries_data_dict[qs]['query']
  # whole_query = queries_data_dict[qs]['whole_query'] #@NEW
  extend_each_by = queries_data_dict[qs]['extend_each_by']
  extended_keywords_list = queries_data_dict[qs]['extended_keywords_list']

  vectorizer = CountVectorizer(min_df=0,dtype=np.uint8)
  train_vec = vectorizer.fit_transform(new_docs).toarray()
  vocab = vectorizer.vocabulary_

  all_aspects_all_keywords = flatten_list(extended_keywords_list)
  query_as_doc = ' '.join(all_aspects_all_keywords)

  #### DESM
  desm_score = DESM_score_Corpus(all_aspects_all_keywords, train_vec, vocab, embeddings)
  sorted_desm_idx = torch.sort(desm_score,descending=True).indices
  ####

  #### TF-IDF
  tfdifvec = TfidfVectorizer()
  tfdifvec.fit(new_docs)
  tfdif_doc_vectors = torch.from_numpy(tfdifvec.transform(new_docs).toarray())
  tfdif_query_vectors = torch.from_numpy(tfdifvec.transform([query_as_doc]).toarray())

  tfidf_score = cos_sim(tfdif_query_vectors,tfdif_doc_vectors)
  sorted_tfidf_idx = torch.sort(tfidf_score,descending=True).indices
  #### 

  bigram_coocurring_word_list = get_bigram_coocurring_word_list(new_docs,keywords)
  keywords_as_labels = []
  # if data_name =='yahooanswers':
  #   extended_keywords_list_joined = np.asarray([list(extended_keywords_list[x]) + (list(extended_keywords_list[x+1])) for x in range(len(extended_keywords_list)) if x%2==0])
  #   keywords_as_labels = np.asarray(['p{'+keywords[x]+'&'+keywords[x+1]+'}' for x in range(len(keywords)) if x%2==0] )
  #   for e in range(len(extended_keywords_list_joined)):
  #     new_docs = np.append(new_docs,' '.join(extended_keywords_list_joined[e]))
  #     new_labels = np.append(new_labels,keywords_as_labels[e])
  #   extended_keywords_list = extended_keywords_list_joined
  # else:

  #@NEW
  # for keyword in whole_query:
  #   pseudo_label = 'p{'+keyword+'}'
  #   keywords_as_labels.append(pseudo_label)
  #   new_labels = np.append(new_labels,pseudo_label)
  #@NEW

  for ext_kws in extended_keywords_list:
    keyword=ext_kws[0]
    kws = ' '.join(ext_kws)
    pseudo_label = 'p{'+keyword+'}'
    new_docs = np.append(new_docs,kws)
    keywords_as_labels.append(pseudo_label)
    new_labels = np.append(new_labels,pseudo_label)

  keywords_as_labels = np.asarray(keywords_as_labels)
  save_dir_no_bkp = '/SavedOutput/'+data_name+"/short/qs"+qs+"/"
  os.chdir(ASTM_results_dir+save_dir_no_bkp)

  all_topics_runtime = []
  with open(data_name+"_qs_"+qs+"_run_"+str(r)+"-ASTM-"+num_topic+"-tt-0.5-0.5.theta") as f:
            content = f.read()
            content = content.split('\n')[:-1 ]
            theta = np.asarray([np.asarray(x.split(' ')[:-1]).astype('float') for x in content])
            all_topics_theta.append(theta)
  with open(data_name+"_qs_"+qs+"_run_"+str(r)+"-ASTM-"+num_topic+"-tt-0.5-0.5.finalTopWords") as f:
            topicwords = f.read()
            topicwords = topicwords.split('\n')
            topicwords = [x.strip() for x in topicwords][:-1]
            topicwords = list(filter(None, topicwords))
            topics_wordlist = [x.split(' ')[1:] for x in topicwords]
            all_topics_wordlist.append(topics_wordlist)

  os.chdir(ASTM_results_dir+save_dir_no_bkp+'/run_'+str(r))
  # with open('ASTM_output.txt') as f:
  #           content = f.readlines()
  #           content = [x.strip() for x in content]
            
  #           for c in range(len(content)):
  #             if "Runtime:-" in content[c]:
  #               all_topics_runtime.append(float(content[c+1].split('\n')[-1].split('\t')[1])/1000)
  #           runtime = f.read()
  #           runtime = float(runtime.split('\n')[-1].split('\t')[1])/1000
            

for nt in range(len(num_topics)):

  tsne = TSNE(n_jobs=-1)
  labels = new_labels
  X = tsne.fit_transform(all_topics_theta[nt])
  pseudo_idx = np.argwhere(np.isin(labels,keywords_as_labels)).ravel()
  not_pseudo_idx = [i for i in range(len(X)) if i not in pseudo_idx]
  query_center = X[pseudo_idx]

  X_original_seq = X[not_pseudo_idx]
  train_label = labels[not_pseudo_idx]
  sorted_unique_labels = sorted(set(train_label))
  zphi = torch.zeros(int(num_topic),2)
  lim=15
  
  ############
  topk = 100

  ### top k Tf-IDf
  topk_tfidfdocs = X_original_seq[sorted_tfidf_idx[:topk]]
  topk_tfidflabels = train_label[sorted_tfidf_idx[:topk]]

  figname = "TFIDF_"+data_name+"_"+dtype+"_topics_"+str(num_topic)+"_run_"+str(r)+"_qs_"+str(qs)
  plot_fig(model_name,topk_tfidfdocs, topk_tfidflabels, zphi,lim,sorted_unique_labels,query_center,keywords_as_labels,hv_qwords=True,showtopic=True,
  bold_topics=True,remove_legend=False,show_axis=True,save=True,figname=figname)
  ### /top K Tf-IDf ####


  ### top K DESM ####
  topk_DESMdocs = X_original_seq[sorted_desm_idx[:topk]]
  topk_DESMlabels = train_label[sorted_desm_idx[:topk]]

  figname = "DESM_"+data_name+"_"+dtype+"_topics_"+str(num_topic)+"_run_"+str(r)+"_qs_"+str(qs)
  plot_fig(model_name,topk_DESMdocs, topk_DESMlabels, zphi,lim,sorted_unique_labels,query_center,keywords_as_labels,hv_qwords=True,showtopic=True,
  bold_topics=True,remove_legend=False,show_axis=True,save=True,figname=figname)
  ### /top K DESM ####


  if data_name == 'nfcorpus' or data_name=='opinions_twitter':
    relv_docs = queries_data_dict[qs]['ground_truth_docs']
    relv_labels = queries_data_dict[qs]['ground_truth_labels']
    d = {item: idx for idx, item in enumerate(data_preprocessed)} 
    relv_docs_idx_ground_truth = [d.get(item) for item in relv_docs]

    sorted_unique_labels_relv = sorted(set(relv_labels))

    figname = "GROUND_TRUTH_"+data_name+"_"+dtype+"_topics_"+str(num_topic)+"_run_"+str(r)+"_qs_"+str(qs)
    plot_fig(model_name,X_original_seq[relv_docs_idx_ground_truth], relv_labels, zphi,lim,sorted_unique_labels_relv,query_center,keywords_as_labels,hv_qwords=True,showtopic=True,
    bold_topics=True,remove_legend=False,show_axis=True,save=True,figname=figname)
    _,_,aupr_ground_truth = cal_AUPR(len(keywords_as_labels),2,relv_docs_idx_ground_truth,X_original_seq,query_center)

    print('AUCPR (ground_truth):- ',aupr_ground_truth)

  # AUPR
  relv_docs_idx_DESM = sorted_desm_idx[:topk]
  _,_,aupr_DESM = cal_AUPR(len(keywords_as_labels),2,relv_docs_idx_DESM,X_original_seq,query_center)
  print('AUCPR (DESM):- ',aupr_DESM)

  relv_docs_idx_tfidf = sorted_tfidf_idx[:topk]
  _,_,aupr_tfidf = cal_AUPR(len(keywords_as_labels),2,relv_docs_idx_tfidf,X_original_seq,query_center)

  print('AUCPR (tf-idf):- ',aupr_tfidf)

  os.chdir(ASTM_results_dir+save_dir_no_bkp+'/run_'+str(r))
  all_results = {}
  all_results['X'] = X 
  all_results['X_no_pseudo'] = X[not_pseudo_idx]
  all_results['labels'] = labels
  all_results['labels_no_pseudo'] = labels[not_pseudo_idx]
  all_results['pseudo_idx'] = pseudo_idx
  all_results['not_pseudo_idx'] = not_pseudo_idx
  all_results['query_center'] =query_center
  all_results['keywords_as_labels'] = keywords_as_labels
  all_results['topics'] = [' '.join(t) for t in all_topics_wordlist[nt]]
  # all_results['runtime'] = all_topics_runtime[nt]
  for aa in range(len(all_topics_wordlist[nt])):
    for a in range(len(all_topics_wordlist[nt][aa])):
      if all_topics_wordlist[nt][aa][a] =='psxbox':
        all_topics_wordlist[nt][aa][a] = 'xbox'
      if all_topics_wordlist[nt][aa][a] =='anykno':
        all_topics_wordlist[nt][aa][a] = 'kno'
      elif all_topics_wordlist[nt][aa][a] =='pssale':
        all_topics_wordlist[nt][aa][a] = 'sale'
      elif all_topics_wordlist[nt][aa][a] =='splenovo':
        all_topics_wordlist[nt][aa][a] = 'lenovo' 
      elif all_topics_wordlist[nt][aa][a] =='anyvideo':
        all_topics_wordlist[nt][aa][a] = 'video'
      elif all_topics_wordlist[nt][aa][a] =='mpsong':
        all_topics_wordlist[nt][aa][a] = 'song'
      elif all_topics_wordlist[nt][aa][a] =='spplaza':
        all_topics_wordlist[nt][aa][a] = 'plaza'
      elif all_topics_wordlist[nt][aa][a] =='anytwitter':
        all_topics_wordlist[nt][aa][a] = 'twitter'
      elif all_topics_wordlist[nt][aa][a] =='mpdownload':
        all_topics_wordlist[nt][aa][a] = 'download'

      elif all_topics_wordlist[nt][aa][a] =='mpfree':
        all_topics_wordlist[nt][aa][a] = 'free'
      elif all_topics_wordlist[nt][aa][a] =='mpborn':
        all_topics_wordlist[nt][aa][a] = 'born'
      elif all_topics_wordlist[nt][aa][a] =='anyborn':
        all_topics_wordlist[nt][aa][a] = 'born'
      elif all_topics_wordlist[nt][aa][a] =='mpopposite':
        all_topics_wordlist[nt][aa][a] = 'opposite'
      elif all_topics_wordlist[nt][aa][a] =='mpenglish':
        all_topics_wordlist[nt][aa][a] = 'english'
      elif all_topics_wordlist[nt][aa][a] =='sometopic':
        all_topics_wordlist[nt][aa][a] = 'topic'
      elif all_topics_wordlist[nt][aa][a] =='anybypass':
        all_topics_wordlist[nt][aa][a] = 'bypass'
      elif all_topics_wordlist[nt][aa][a] == 'anyplay':
        all_topics_wordlist[nt][aa][a] = 'play'
      elif all_topics_wordlist[nt][aa][a] == 'spwindow':
        all_topics_wordlist[nt][aa][a] = 'window'
      
        
  all_results['aupr_DESM'] = aupr_DESM
  all_results['aupr_tfidf'] = aupr_tfidf
  if data_name == 'nfcorpus' or data_name=='opinions_twitter':
    all_results['aupr_ground_truth'] = aupr_ground_truth
  all_results['sum_avg_cos'] = get_cosine_sum_topics(all_topics_wordlist[nt],embeddings,keywords)
  all_results['colored_topics'] = get_colored_topics(all_topics_wordlist[nt],keywords,bigram_coocurring_word_list,extended_keywords_list)
  all_results['KNN'] = cal_knn(X,labels)

  all_results['topic#'] = num_topics[nt]
  print('topic:',num_topics[nt],os.getcwd())
  compressed_pickle(all_results,"ASTM_metrics_"+data_name+"_"+"numtopic_"+num_topics[nt]+"_run_"+str(r)+"_qs_"+qs)