from collections import defaultdict
import torch.nn as nn 
import numpy as np
import torch

cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-8)
torch.pi = torch.acos(torch.zeros(1)).item() * 2 
# model 5.3
sig_exp_dist = 50.0


def flatten_list(user_list): return [item for sublist in user_list for item in sublist]
def get_embedding_tensor(word_list,embeddings): return torch.tensor([embeddings[w] for w in word_list]) 


def cosine_angular_sim(keyword_torch,words_tensor): return 1 - (torch.acos(cos_sim(keyword_torch,words_tensor))) / torch.pi
def cosine_norm_01(keyword_torch,words_tensor): return (cos_sim(keyword_torch,words_tensor) + 1.0)/2.0
def cosine_sqrt(keyword_torch,words_tensor): return 1 - ((1 - cos_sim(keyword_torch,words_tensor) )/2)**0.5
def euclidean_dist(keyword_torch,words_tensor) : return ((words_tensor-keyword_torch).pow(2).sum(-1))**0.5

def edist(x,x1):
  dist = torch.pow((x-x1),2).sum(-1)
  dist_sim = torch.exp(-dist /0.1)
  return dist_sim

  
def inv_new_fn(x,x1):
  beta = 3
  dist = torch.pow((x-x1),2).sum(-1)
  dist_sim = 1.0 / (1.0 + (dist/(1-dist))**(-beta))
  return dist_sim

def exp_dist_sim_x_x1(x,x1,sig=sig_exp_dist):
  # sig = 0.1
  dist = torch.pow((x-x1),2).sum(-1)
  # dist2 = torch.norm((x-x1),dim=-1)**2
  exp_dist_sim = torch.exp(- dist / sig)
  return exp_dist_sim

def get_dist_val_score(f,keyword_torch,words_tensor):
  score = f(keyword_torch,words_tensor)
  return score
  
def cosine_keywords(keywords,words_tensor,word_list,embeddings):
  all_keywords_score = []
  all_cosine_sim = []
  all_dist = []
  cosine_score_vocab = {}
  keyword_total_score = torch.zeros(words_tensor.shape[0])
  for k in keywords:
    keyword_torch = torch.from_numpy(embeddings[k])
    keyword_torch = keyword_torch.unsqueeze(0).expand(words_tensor.shape[0],words_tensor.shape[1])
    
    cosine_sim_score = cos_sim(keyword_torch,words_tensor)
    score = get_dist_val_score(cosine_sqrt,keyword_torch,words_tensor)
    # cosine_score = cosine_sqrt(keyword_torch,words_tensor)    
    # cosine_score[idx_keys_relvWord[k]] = 1.0
    all_keywords_score.append(score)
    all_cosine_sim.append(cosine_sim_score)

    # dist_sim_val,dist_val = exp_dist_sim_x_x1(words_tensor,keyword_torch,sig_exp_dist)
    # dist_sim_val,dist_val = edist(keyword_torch,words_tensor,sig_exp_dist)

    # dist_sim_val[idx_keys_relvWord[k]] = 1.0
    # all_keywords_score.append(dist_sim_val)
    # all_dist.append(dist_val)
    
    keyword_total_score += cosine_sim_score
  keywords_max_score,keyword_max_score_idx = torch.max(torch.stack(all_keywords_score),dim=0)
  cosine_score_vocab = dict(zip(word_list,keywords_max_score.numpy()))

  return all_keywords_score,keyword_total_score/len(keywords),keywords_max_score,cosine_score_vocab,all_cosine_sim


def get_ranking_parameters(train_vec,preprossed_data_non_zeros,keywords,all_keywords_score,word_list,vocab):
  doc_contains_anykey = torch.zeros(train_vec.shape[0])
  for k in keywords:
    for d in range(train_vec.shape[0]):
        doc_contains_anykey[d] += train_vec[d][vocab[k]]

  doc_contains_key = torch.zeros(len(preprossed_data_non_zeros),len(keywords)+1)
  for i in range(len(preprossed_data_non_zeros)):
    for k in range(len(keywords)):
      if keywords[k] in preprossed_data_non_zeros[i]:
        doc_contains_key[i][k] = 1.0 

  sum_doc_contains_key = doc_contains_key.sum(-1)     
  for i in range(len(preprossed_data_non_zeros)):
    if sum_doc_contains_key[i] == 0:
      doc_contains_key[i][-1] = 1.0

  most_similar_word_dict = {}
  for k in range(len(keywords)):
    v, i = torch.sort(all_keywords_score[k],descending=True)
    most_similar = np.array(word_list)[np.array(i)][:100]
    most_similar_word_dict[keywords[k]] = most_similar

  extended_keywords = keywords.copy()
  avg_doc_len = round(np.mean([len(d.split(" ")) for d in preprossed_data_non_zeros]))
  extend_each_by = avg_doc_len
  for k in keywords:
    extended_keywords.extend(list(most_similar_word_dict[k][1:extend_each_by+1]))


  ###keywords as docs
  most_similar_word_dict = {}
  extended_keywords_list = []
  for k in range(len(keywords)):
    v, i = torch.sort(all_keywords_score[k],descending=True)
    most_similar = np.array(word_list)[np.array(i)][:extend_each_by]
    extended_keywords_list.append(most_similar)
    print(most_similar)
    most_similar_word_dict[keywords[k]] = most_similar

  keywords_as_docs = np.zeros(shape=(len(keywords),len(word_list)))
  top_k = extend_each_by
  for i in range(len(keywords)):
    top_sim = most_similar_word_dict[keywords[i]][:top_k]
    for w in top_sim:
      keywords_as_docs[i][word_list.index(w)] += 1


  doc_contains_anykey_ext = torch.zeros(train_vec.shape[0])
  for k in extended_keywords:
    for d in range(train_vec.shape[0]):
        doc_contains_anykey_ext[d] += train_vec[d][vocab[k]]

  doc_contains_anykey_ext = torch.zeros(train_vec.shape[0],len(keywords))
  for i in range(len(keywords)):
    extended_keywords = list(most_similar_word_dict[keywords[i]][0:extend_each_by])
    for k in extended_keywords:
      for d in range(train_vec.shape[0]):
        doc_contains_anykey_ext[d][i] += train_vec[d][vocab[k]]


  ## count of extended keywords metric (kxN)
  count_for_d = []
  c = 0
  for d in range(train_vec.shape[0]):
    count = [0]*len(keywords)
    contain = False
    for i in range(len(keywords)):
      for j in extended_keywords_list[i]:
        word_count_ext_k = train_vec[d][word_list.index(j)]
        count[i] += word_count_ext_k
        if(word_count_ext_k>0): contain = True
    if(contain): c+=1
    count_for_d.append(count)
  score_q_for_all_doc = torch.tensor(count_for_d)
  ranking_q_for_all_doc = score_q_for_all_doc

  return keywords_as_docs,doc_contains_anykey_ext,ranking_q_for_all_doc,extended_keywords_list