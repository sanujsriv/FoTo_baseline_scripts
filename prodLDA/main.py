import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.cuda
from pprint import pprint, pformat
import pickle
import argparse
import os
import math
import matplotlib.pyplot as plt
from pytorch_model import ProdLDA
# from pytorch_visualize import *
from MulticoreTSNE import MulticoreTSNE as TSNE
tsne = TSNE(n_jobs=-1)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from time import time
cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-8)

import csv
from data import get_data_label_vocab
from metrics import get_docs_idx_in_vis,cal_knn,get_colored_topwords,colored_print_Topics
from metrics import get_cosine_sum_topics,plot_fig,plot_relv_irrelv_docs,cal_AUPR,get_bigram_coocurring_word_list
from utils import get_embedding_tensor,get_topwords,print_Topics,generate_co_occurrence_matrix,flatten_list,cosine_keywords,DESM_score_Corpus

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


parser = argparse.ArgumentParser()

parser.add_argument('--run', type=int, default=1, help='run')
parser.add_argument('-qs','--queryset', type=int, default=1, help='the queryset to pass')

parser.add_argument('-d', '--dataset',        type=str,   default='bbc')
parser.add_argument('-dt', '--dtype',        type=str,   default='short')
parser.add_argument('-t', '--num_topic',        type=int,   default=10)
parser.add_argument('-f', '--en1-units',        type=int,   default=100)
parser.add_argument('-s', '--en2-units',        type=int,   default=100)
parser.add_argument('-b', '--batch_size',       type=int,   default=10000)
parser.add_argument('-o', '--optimizer',        type=str,   default='Adam')
parser.add_argument('-r', '--learning-rate',    type=float, default=0.001) #0.002
parser.add_argument('-m', '--momentum',         type=float, default=0.99)
parser.add_argument('-e', '--num-epoch',        type=int,   default=1000)
parser.add_argument('-q', '--init-mult',        type=float, default=1.0)    # t variance in prior normalmultiplier in initialization of decoder weight
parser.add_argument('-v', '--variance',         type=float, default=0.995)  # defaul
parser.add_argument('--start',                  action='store_true')        # start training at invocation
parser.add_argument('--nogpu',                  action='store_true')        # do not use GPU acceleration

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
data_name = args.dataset
dtype = args.dtype
num_topic =args.num_topic

paper = "emnlp2022"
model_name = 'prodLDA'
prod_lda_dir = '/home/grad16/sakumar/'+paper+'/'+model_name
data_dir = '/home/grad16/sakumar/'+paper+'/dataset'
save_dir_no_bkp = '/home/student_no_backup/sakumar/'+paper+'/'+model_name
# save_dir_no_bkp = prod_lda_dir

batch_size = args.batch_size

if 'bbc' in data_name: 
  batch_size = 250
elif 'searchsnippet' in data_name: 
  batch_size = 250
elif 'yahooanswers' in data_name: 
  batch_size = 1000
elif 'nfcorpus' in data_name: 
  batch_size = 250
elif 'opinions_twitter' in data_name: 
  batch_size = 250
else: 
  batch_size = 250


queryset = args.queryset

print('\nbatch_size: ',batch_size)
d_dir = "/content/"+"data_"+data_name+"/"+dtype
os.chdir(data_dir+d_dir)

data_preprocessed=load_obj_pkl5("data_preprocessed_"+data_name+"_"+dtype)
data_preprocessed_labels=load_obj_pkl5("data_preprocessed_labels_"+data_name+"_"+dtype)
embeddings = load_obj("embeddings_"+data_name+"_"+dtype)
queries_data_dict = decompress_pickle("queries_"+data_name)
qs = str(queryset)
keywords = queries_data_dict[qs]['query']
whole_query = queries_data_dict[qs]['whole_query']
extended_keywords_list = queries_data_dict[qs]['extended_keywords_list']
os.chdir(prod_lda_dir)


train_vec,train_label,_,preprossed_data_non_zeros,vocab = get_data_label_vocab(data_preprocessed,data_preprocessed_labels)
train_label  =np.asanyarray(train_label)
all_aspects_all_keywords = flatten_list(extended_keywords_list)
query_as_doc = ' '.join(all_aspects_all_keywords)

#### DESM
desm_score = DESM_score_Corpus(all_aspects_all_keywords, train_vec, vocab, embeddings)
sorted_desm_idx = torch.sort(desm_score,descending=True).indices
####

#### TF-IDF
tfdifvec = TfidfVectorizer()
tfdifvec.fit(preprossed_data_non_zeros)
tfdif_doc_vectors = torch.from_numpy(tfdifvec.transform(preprossed_data_non_zeros).toarray())
tfdif_query_vectors = torch.from_numpy(tfdifvec.transform([query_as_doc]).toarray())

tfidf_score = cos_sim(tfdif_query_vectors,tfdif_doc_vectors)
sorted_tfidf_idx = torch.sort(tfidf_score,descending=True).indices

data = list(data_preprocessed)
labels = data_preprocessed_labels[:]
print('Before adding pseudo : ',len(data),len(labels))
keywords_as_labels = []


for i in range(len(whole_query)):
  kws = extended_keywords_list[i]
  print(kws)
  kws = ' '.join(extended_keywords_list[i])
  data.append(kws)
  pseudo_label = 'p{'+whole_query[i]+'}'
  keywords_as_labels.append(pseudo_label)
  labels = np.append(labels,pseudo_label)

data = np.asarray(data)
print('After adding pseudo : ',len(data),len(labels))
keywords_as_labels = np.asarray(keywords_as_labels)
print(keywords_as_labels)
num_keyword = len(keywords_as_labels)
# default to use GPU, but have to check if GPU exists

def to_onehot(data, min_length):
    return np.bincount(data, minlength=min_length)

def make_data():
    global data_tr, vocab,id_vocab, vocab_size

    data_tr = np.asarray(data)
    vectorizer = CountVectorizer(min_df=0,dtype=np.uint8)
    data_tr = vectorizer.fit_transform(data_tr).toarray()
    vocab = vectorizer.vocabulary_
    id_vocab = dict(map(reversed, vocab.items()))
    vocab_size = len(vocab)
    print(data_tr.shape) 
    # tensor_tr = torch.from_numpy(np.array(data_tr)).float()
    
    # if not args.nogpu:
        # tensor_tr = tensor_tr.cuda()

def make_model():
    global model
    net_arch = args # en1_units, en2_units, num_topic, num_input
    net_arch.num_input = data_tr.shape[1]
    model = ProdLDA(net_arch)
    # if not args.nogpu:
    model = model.to(device)

def make_optimizer():
    global optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, betas=(args.momentum, 0.999))
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum)
    else:
        assert False, 'Unknown optimizer {}'.format(args.optimizer)

def train():
    global all_indices
    all_indices = torch.randperm(data_tr.shape[0]).split(batch_size)
    for epoch in range(args.num_epoch):
        loss_epoch = 0.0
        model.train()                   # switch to training mode
        for batch_indices in all_indices:

            # if not args.nogpu: batch_indices = batch_indices.to(device)
            input_ = torch.tensor(data_tr[batch_indices]).float().to(device)
            # input_ = tensor_tr[batch_indices].to(device)
            recon, loss = model(input_, compute_loss=True)
            # optimize
            optimizer.zero_grad()       # clear previous gradients
            loss.backward()             # backprop
            optimizer.step()            # update parameters
            # report
            loss_epoch += loss.item()    # add loss to loss_epoch
        if epoch % 5 == 0:
            print('Epoch {}, loss={}'.format(epoch, loss_epoch / len(all_indices)))
    

def identify_topic_in_line(line):
    topics = []
    for topic, keywords in associations.items():
        for word in keywords:
            if word in line:
                topics.append(topic)
                break
    return topics

def print_top_words(beta, feature_names, n_top_words=10):
    print('---------------Printing the Topics------------------')
    for i in range(len(beta)):
        line = " ".join([feature_names[j] 
                            for j in beta[i].argsort()[:-n_top_words - 1:-1]])
        # topics = identify_topic_in_line(line)
        # print('|'.join(topics))
        print('     {}'.format(line))
    print('---------------End of Topics------------------')

if __name__=='__main__':
    make_data()
    tstart = time()
    make_model()
    make_optimizer()
    train()
    emb = model.decoder.weight.data.cpu().numpy().T
    print_top_words(emb, list(zip(*sorted(vocab.items(), key=lambda x:x[1])))[0])
    tstop = time()

    model.eval()
    theta_list=[]
    doc_ids = []
    labels_list = []
    with torch.no_grad():
      for batch_indices in all_indices:
          # input_ = tensor_tr[batch_indices].to(device)
          input_ = torch.tensor(data_tr[batch_indices]).float().to(device)
          recon,theta  = model(input_, compute_loss=False)
          theta = model.theta.data.detach().cpu().numpy()
          theta_list = np.append(theta_list,theta)
          labels_list.extend(labels[batch_indices])
          doc_ids = np.append(doc_ids,batch_indices.numpy().astype(int))
    labels_list_ar = np.asarray(labels_list)
    doc_ids_w_pseudo = doc_ids
    m_theta = theta_list.reshape(data_tr.shape[0],args.num_topic)
    beta= emb
    bigram_coocurring_word_list = get_bigram_coocurring_word_list(data_preprocessed,keywords)

    no_of_topwords = 10
    colored_topics = get_colored_topwords(no_of_topwords,beta,id_vocab,keywords,bigram_coocurring_word_list,extended_keywords_list)
    print('Colored topics: <g> extended keyword , <r> keyword, <b> bigram \n\n')
    colored_print_Topics(no_of_topwords,beta,id_vocab,keywords,bigram_coocurring_word_list,extended_keywords_list)    
    
    print('topics: \n\n')
    print_Topics(beta,id_vocab,no_of_topwords)  

    print('Running TSNE....\n\n')
    
    pseudo_idx = np.argwhere(np.isin(labels_list_ar,keywords_as_labels)).ravel()
    not_pseudo_idx = [i for i in range(data_tr.shape[0]) if i not in pseudo_idx]

    X_w_pseudo = tsne.fit_transform(m_theta)
    X_w_pseudo = np.asarray(X_w_pseudo)

    query_center = X_w_pseudo[pseudo_idx]

    X = X_w_pseudo[not_pseudo_idx]
    labels_list = labels_list_ar[not_pseudo_idx]

    x_list = X

    pseudo_idx_in_data = [i for i in range(len(data_preprocessed),len(data_preprocessed)+len(keywords_as_labels)+1)]
    doc_ids = np.asarray([d for d in doc_ids_w_pseudo if d not in pseudo_idx_in_data])
  
    print('X (no pseudo) shape: ',X.shape)
    sorted_unique_labels = sorted(set(labels))
    zphi = torch.zeros(num_topic,X.shape[1]).numpy()
    deep_dir = data_name+"/"+dtype+"/topics_"+str(args.num_topic)+"/qs_"+str(queryset)+"/run_"+str(args.run)

    os.chdir(prod_lda_dir)
    save_dir = save_dir_no_bkp+"/SavedOutput/"+deep_dir
    os.makedirs(save_dir,exist_ok=True)
    os.chdir(save_dir)
    
    figname = data_name+"_"+dtype+"_topics_"+str(args.num_topic)+"_run_"+str(args.run)+"_qs_"+str(queryset)
    # plot_fig(x_list, labels_list,zphi,lim =10,contour='no')
    lim=20
    plot_fig(model_name,x_list, labels_list, zphi,lim,sorted_unique_labels,query_center,keywords_as_labels,hv_qwords=True,showtopic=True,
    bold_topics=True,remove_legend=False,show_axis=True,save=True,figname=figname)

    #*********** Quantitative ***********#
    # if show_knn:
    knn = cal_knn(x_list,labels_list)
    print('KNN:- ',knn)

    ############
    doc_id_in_seq = np.asanyarray([np.where(doc_ids==i)[0][0] for i in range(len(doc_ids))])
    X_original_seq = x_list[doc_id_in_seq]
    topk = 100

    ### top k Tf-IDf
    topk_tfidfdocs = X_original_seq[sorted_tfidf_idx[:topk]]
    topk_tfidflabels = train_label[sorted_tfidf_idx[:topk]]

    figname = "TFIDF_"+data_name+"_"+dtype+"_topics_"+str(args.num_topic)+"_run_"+str(args.run)+"_qs_"+str(queryset)
    plot_fig(model_name,topk_tfidfdocs, topk_tfidflabels, zphi,lim,sorted_unique_labels,query_center,keywords_as_labels,hv_qwords=True,showtopic=True,
    bold_topics=True,remove_legend=False,show_axis=True,save=True,figname=figname)
    ### /top K Tf-IDf ####


    ### top K DESM ####
    topk_DESMdocs = X_original_seq[sorted_desm_idx[:topk]]
    topk_DESMlabels = train_label[sorted_desm_idx[:topk]]

    figname = "DESM_"+data_name+"_"+dtype+"_topics_"+str(args.num_topic)+"_run_"+str(args.run)+"_qs_"+str(queryset)
    plot_fig(model_name,topk_DESMdocs, topk_DESMlabels, zphi,lim,sorted_unique_labels,query_center,keywords_as_labels,hv_qwords=True,showtopic=True,
    bold_topics=True,remove_legend=False,show_axis=True,save=True,figname=figname)
    ### /top K DESM ####

    if data_name == 'nfcorpus' or data_name=='opinions_twitter':
      relv_docs = queries_data_dict[qs]['ground_truth_docs']
      relv_labels = queries_data_dict[qs]['ground_truth_labels']
      d = {item: idx for idx, item in enumerate(data_preprocessed)} 
      relv_docs_idx_ground_truth = [d.get(item) for item in relv_docs]

      sorted_unique_labels_relv = sorted(set(relv_labels))

      figname = "GROUND_TRUTH_"+data_name+"_"+dtype+"_topics_"+str(args.num_topic)+"_run_"+str(args.run)+"_qs_"+str(queryset)
      plot_fig(model_name,X_original_seq[relv_docs_idx_ground_truth], relv_labels, zphi,lim,sorted_unique_labels_relv,query_center,keywords_as_labels,hv_qwords=True,showtopic=True,
      bold_topics=True,remove_legend=False,show_axis=True,save=True,figname=figname)
      _,_,aupr_ground_truth = cal_AUPR(len(keywords_as_labels),2,relv_docs_idx_ground_truth,X_original_seq,query_center)

      print('AUCPR (ground_truth):- ',aupr_ground_truth)
    
    relv_docs_idx_DESM = sorted_desm_idx[:topk]
    _,_,aupr_DESM = cal_AUPR(num_keyword,2,relv_docs_idx_DESM,X_original_seq,query_center)
    print('AUCPR (DESM):- ',aupr_DESM)

    relv_docs_idx_tfidf = sorted_tfidf_idx[:topk]
    _,_,aupr_tfidf = cal_AUPR(num_keyword,2,relv_docs_idx_tfidf,X_original_seq,query_center)
    
    print('AUCPR (tf-idf):- ',aupr_tfidf)

    ## sum avg cosine ##
    all_topics = get_topwords(beta,id_vocab,topwords=20)
    sum_avg_cos = get_cosine_sum_topics(all_topics,beta,no_of_topwords,id_vocab,embeddings,keywords)
 
    with open("results_"+data_name+"_"+str(args.num_topic)+".txt","w") as f:

        f.write('---'*30+'\n\n')
        f.write('runtime: - '+str(tstop-tstart)+'s\n\n')
        f.write('---------------Printing the Topics------------------\n')
        beta  = emb
        feature_names = list(zip(*sorted(vocab.items(), key=lambda x:x[1])))[0]
        n_top_words = 10
        for i in range(len(beta)):
          line = " ".join([feature_names[j] 
                              for j in beta[i].argsort()[:-n_top_words - 1:-1]])
          f.write(str(i+1)+" : "+line+"\n")
        f.write('---------------End of Topics---------------------\n')
        f.write('theta_shape: - '+str(m_theta.shape)+'\n\n')
        f.write('beta_shape: - '+str(beta.shape)+'\n\n')
        f.write('---'*30+'\n\n')
    
    model_signature=model_name+'_'+data_name+'_'+dtype+'_'+str(args.num_topic)+'_'+str(args.run)
    torch.save(model.state_dict(), model_signature+'.pt')

    all_results = {}
    all_results['theta'] = m_theta
    all_results['beta'] = emb
    all_results['doc_ids'] = doc_ids
    all_results['doc_ids_w_pseudo'] = doc_ids_w_pseudo
    all_results['keywords'] = keywords
    all_results['args'] = args
    all_results['X_w_pseudo'] = X_w_pseudo
    all_results['X'] = X
    all_results['labels_list_w_pseudo'] = labels_list_ar
    all_results['labels_list'] = labels_list 
    all_results['query_center'] = query_center
    all_results['phi'] = zphi
    all_results['all_indices'] = all_indices
    all_results['runtime'] = tstop-tstart
    all_results['topics'] =  all_topics
    all_results['colored_topics'] = colored_topics
    all_results['KNN'] = knn
    all_results['aupr_DESM'] = aupr_DESM
    all_results['aupr_tfidf'] = aupr_tfidf
    if data_name == 'nfcorpus' or data_name=='opinions_twitter':
      all_results['aupr_ground_truth'] = aupr_ground_truth
    compressed_pickle(all_results,model_signature+'_all_results')
    os.chdir(prod_lda_dir)
