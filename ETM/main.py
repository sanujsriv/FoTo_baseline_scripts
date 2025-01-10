#/usr/bin/python
from __future__ import print_function

import argparse
import torch
import pickle 
import numpy as np 
import os 
import math 
import random 
import sys
import matplotlib.pyplot as plt 
import data
import scipy.io
from time import time
from torch import nn, optim
from torch.nn import functional as F
from MulticoreTSNE import MulticoreTSNE as TSNE
tsne = TSNE(n_jobs=-1)
from etm import ETM
from data import get_data_label_vocab
from utils import nearest_neighbors, get_topic_coherence, get_topic_diversity

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from metrics import get_docs_idx_in_vis,cal_knn,get_colored_topwords,colored_print_Topics
from metrics import get_cosine_sum_topics,plot_fig,plot_relv_irrelv_docs,cal_AUPR,get_bigram_coocurring_word_list
from utils import list_of_tensors_to_tensor,get_embedding_tensor,get_topwords,generate_co_occurrence_matrix,flatten_list,cosine_keywords,DESM_score_Corpus

import pickle5
import bz2
import pickle
import _pickle as cPickle

import torch.nn as nn
cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-8)


dtype = 'short'
paper = "emnlp2022"
model_name = 'ETM'
ETM_dir = '/home/grad16/sakumar/'+paper+'/'+model_name
data_dir = '/home/grad16/sakumar/'+paper+'/dataset'
save_dir_no_bkp = '/home/student_no_backup/sakumar/'+paper+'/'+model_name

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

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
parser = argparse.ArgumentParser(description='The Embedded Topic Model')

parser.add_argument('-qs','--queryset', type=int, default=1, help='the queryset to pass')
parser.add_argument('--run', type=int, default=1, help='run')


### data and file related arguments
parser.add_argument('--dataset', type=str, default='20ng', help='name of corpus')
parser.add_argument('--data_path', type=str, default='data/20ng', help='directory containing data')
parser.add_argument('--emb_path', type=str, default='data/20ng_embeddings.txt', help='directory containing word embeddings')
parser.add_argument('--save_path', type=str, default='./results', help='path to save results')
parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for training')

### model-related arguments
parser.add_argument('--num_topic', type=int, default=50, help='number of topics')
parser.add_argument('--rho_size', type=int, default=300, help='dimension of rho')
parser.add_argument('--emb_size', type=int, default=300, help='dimension of embeddings')
parser.add_argument('--t_hidden_size', type=int, default=800, help='dimension of hidden space of q(theta)')
parser.add_argument('--theta_act', type=str, default='relu', help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')
parser.add_argument('--train_embeddings', type=int, default=0, help='whether to fix rho or train it')

### optimization-related arguments
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this...')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train...150 for 20ng 100 for others')
parser.add_argument('--mode', type=str, default='train', help='train or eval model')
parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
parser.add_argument('--seed', type=int, default=2022, help='random seed (default: 1)')
parser.add_argument('--enc_drop', type=float, default=0.0, help='dropout rate on encoder')
parser.add_argument('--clip', type=float, default=0.0, help='gradient clipping')
parser.add_argument('--nonmono', type=int, default=10, help='number of bad hits allowed')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')
parser.add_argument('--anneal_lr', type=int, default=0, help='whether to anneal the learning rate or not')
parser.add_argument('--bow_norm', type=int, default=1, help='normalize the bows or not')

### evaluation, visualization, and logging-related arguments
parser.add_argument('--num_words', type=int, default=20, help='number of words for topic viz')
parser.add_argument('--log_interval', type=int, default=2, help='when to log training')
parser.add_argument('--visualize_every', type=int, default=10, help='when to visualize results')
parser.add_argument('--eval_batch_size', type=int, default=1000, help='input batch size for evaluation')
parser.add_argument('--load_from', type=str, default='', help='the name of the ckpt to eval from')
parser.add_argument('--tc', type=int, default=0, help='whether to compute topic coherence or not')
parser.add_argument('--td', type=int, default=0, help='whether to compute topic diversity or not')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device : ',device)
data_name = args.dataset
eval_batch_size = args.eval_batch_size
batch_size = args.batch_size


if 'bbc' in data_name: 
  batch_size = 250
  eval_batch_size = 250
elif 'searchsnippet' in data_name: 
  batch_size = 250
  eval_batch_size = 250
elif 'yahooanswers' in data_name: 
  batch_size = 1000
  eval_batch_size = 1000
elif 'nfcorpus' in data_name: 
  batch_size = 250
  eval_batch_size = 250
elif 'opinions_twitter' in data_name: 
  batch_size = 250
  eval_batch_size = 250
else: 
  batch_size = 250
  eval_batch_size = 250

queryset = args.queryset

print('\n')
print(str(args.dataset)," : eval-bs,bs",eval_batch_size,' , ',batch_size)
print('\n')

# seed = int(time())
# np.random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)

## get data
# 1. vocabulary
vocab, train, valid, test = data.get_data(os.path.join(args.data_path))
vocab_size = len(vocab)
args.vocab_size = vocab_size


d_dir = "/content/"+"data_"+data_name+"/"+dtype
os.chdir(data_dir+d_dir)

data_preprocessed=load_obj_pkl5("data_preprocessed_"+data_name+"_"+dtype)
data_preprocessed_labels=load_obj_pkl5("data_preprocessed_labels_"+data_name+"_"+dtype)
embeddings = load_obj_pkl5("embeddings_"+data_name+"_"+dtype)
queries_data_dict = decompress_pickle("queries_"+data_name)
qs = str(queryset)
keywords = queries_data_dict[qs]['query']
extended_keywords_list = queries_data_dict[qs]['extended_keywords_list']

train_vec,train_label,_,preprossed_data_non_zeros,vocab_data = get_data_label_vocab(data_preprocessed,data_preprocessed_labels)
train_label  =np.asanyarray(train_label)
all_aspects_all_keywords = flatten_list(extended_keywords_list)
query_as_doc = ' '.join(all_aspects_all_keywords)

#### DESM
desm_score = DESM_score_Corpus(all_aspects_all_keywords, train_vec, vocab_data, embeddings)
sorted_desm_idx = torch.sort(desm_score,descending=True).indices
####

#### TF-IDF
tfdifvec = TfidfVectorizer()
tfdifvec.fit(preprossed_data_non_zeros)
tfdif_doc_vectors = torch.from_numpy(tfdifvec.transform(preprossed_data_non_zeros).toarray())
tfdif_query_vectors = torch.from_numpy(tfdifvec.transform([query_as_doc]).toarray())

tfidf_score = cos_sim(tfdif_query_vectors,tfdif_doc_vectors)
sorted_tfidf_idx = torch.sort(tfidf_score,descending=True).indices





os.chdir(args.data_path)
etm_data_dict = decompress_pickle('etm_data_dict')
new_docs = etm_data_dict['new_docs']
new_labels = etm_data_dict['new_labels']
keywords_as_labels = etm_data_dict['keywords_as_labels']
inj_embeddings = etm_data_dict['inj_embeddings']
idx_permute = etm_data_dict['idx_permute']


print("len new_docs (w pseudo): ",len(new_docs))
print(new_docs[-5:])

# 1. training data
train_tokens = train['tokens']
train_counts = train['counts']
args.num_docs_train = len(train_tokens)

# 2. dev set
valid_tokens = valid['tokens']
valid_counts = valid['counts']
args.num_docs_valid = len(valid_tokens)

# 3. test data
test_tokens = test['tokens']
test_counts = test['counts']
args.num_docs_test = len(test_tokens)

test_1_tokens = test['tokens_1']
test_1_counts = test['counts_1']
args.num_docs_test_1 = len(test_1_tokens)

test_2_tokens = test['tokens_2']
test_2_counts = test['counts_2']
args.num_docs_test_2 = len(test_2_tokens)

all_data_tokens = []
all_data_counts = []


all_data_tokens.extend(train_tokens)
all_data_tokens.extend(test_tokens)
all_data_tokens.extend(valid_tokens)

all_data_counts.extend(train_counts)
all_data_counts.extend(test_counts)
all_data_counts.extend(valid_counts)

embeddings = None
if not args.train_embeddings:
    emb_path = args.emb_path
    # vect_path = os.path.join(args.data_path.split('/')[0], 'embeddings.pkl')   
    vectors = {}
    with open(emb_path, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            if word in vocab:
                vect = np.array(line[1:]).astype(np.float)
                vectors[word] = vect
    embeddings = np.zeros((vocab_size, args.emb_size))
    words_found = 0
    for i, word in enumerate(vocab):
        try: 
            embeddings[i] = vectors[word]
            words_found += 1
        except KeyError:
            embeddings[i] = np.random.normal(scale=0.6, size=(args.emb_size, ))
    embeddings = torch.from_numpy(embeddings).to(device)
    args.embeddings_dim = embeddings.size()

print('=*'*100)
print('Training an Embedded Topic Model on {} with the following settings: {}'.format(args.dataset.upper(), args))
print('=*'*100)

## define checkpoint
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if args.mode == 'eval':
    ckpt = args.load_from
else:
    ckpt = os.path.join(args.save_path, 
        'etm_{}_K_{}_Htheta_{}_Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_trainEmbeddings_{}'.format(
        args.dataset, args.num_topic, args.t_hidden_size, args.optimizer, args.clip, args.theta_act, 
            args.lr, batch_size, args.rho_size, args.train_embeddings))

## define model and optimizer
model = ETM(args.num_topic, vocab_size, args.t_hidden_size, args.rho_size, args.emb_size, 
                args.theta_act, embeddings, args.train_embeddings, args.enc_drop).to(device)

print('model: {}'.format(model))

if args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'asgd':
    optimizer = optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
else:
    print('Defaulting to vanilla SGD')
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

def train(epoch):
    model.train()
    acc_loss = 0
    acc_kl_theta_loss = 0
    cnt = 0
    indices = torch.randperm(args.num_docs_train)
    indices = torch.split(indices, batch_size)
    for idx, ind in enumerate(indices):
        optimizer.zero_grad()
        model.zero_grad()
        data_batch = data.get_batch(train_tokens, train_counts, ind, args.vocab_size, device)
        sums = data_batch.sum(1).unsqueeze(1)
        if args.bow_norm:
            normalized_data_batch = data_batch / sums
        else:
            normalized_data_batch = data_batch
        recon_loss, kld_theta = model(data_batch, normalized_data_batch)
        total_loss = recon_loss + kld_theta
        total_loss.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        acc_loss += torch.sum(recon_loss).item()
        acc_kl_theta_loss += torch.sum(kld_theta).item()
        cnt += 1

        if idx % args.log_interval == 0 and idx > 0:
            cur_loss = round(acc_loss / cnt, 2) 
            cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
            cur_real_loss = round(cur_loss + cur_kl_theta, 2)

            print('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                epoch, idx, len(indices), optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
    
    cur_loss = round(acc_loss / cnt, 2) 
    cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
    cur_real_loss = round(cur_loss + cur_kl_theta, 2)
    print('*'*100)
    print('Epoch----->{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
            epoch, optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
    print('*'*100)

def visualize(m, show_emb=True):
    if not os.path.exists('./results'):
        os.makedirs('./results')

    m.eval()
    ## visualize topics using monte carlo
    with torch.no_grad():
        print('#'*100)
        print('Visualize topics...')
        topics_words = []
        gammas = m.get_beta()
        for k in range(args.num_topic):
            gamma = gammas[k]
            top_words = list(gamma.cpu().numpy().argsort()[-args.num_words+1:][::-1])
            topic_words = [vocab[a] for a in top_words]
            topics_words.append(' '.join(topic_words))
            print('Topic {}: {}'.format(k, topic_words))

        if show_emb:
            ## visualize word embeddings by using V to get nearest neighbors
            print('#'*100)
            print('Visualize word embeddings by using output embedding matrix')
            try:
                embeddings = m.rho.weight  # Vocab_size x E
            except:
                embeddings = m.rho         # Vocab_size x E
            neighbors = []
            # for word in queries:
            #     print('word: {} .. neighbors: {}'.format(
            #         word, nearest_neighbors(word, embeddings, vocab)))
            print('#'*100)

def evaluate(m, source, tc=False, td=False):
    """Compute perplexity on document completion.
    """
    m.eval()
    with torch.no_grad():
        theta_list = []
        if source == 'val':
            indices = torch.split(torch.tensor(range(args.num_docs_valid)), eval_batch_size)
            tokens = valid_tokens
            counts = valid_counts
        else: 
            indices = torch.split(torch.tensor(range(args.num_docs_test)), eval_batch_size)
            tokens = test_tokens
            counts = test_counts

        ## get \beta here
        beta = m.get_beta()

        ### do dc and tc here
        acc_loss = 0
        cnt = 0
        
        indices_1 = torch.split(torch.tensor(range(args.num_docs_test_1)), eval_batch_size)
        for idx, ind in enumerate(indices_1):
            ## get theta from first half of docs
            data_batch_1 = data.get_batch(test_1_tokens, test_1_counts, ind, args.vocab_size, device)
            sums_1 = data_batch_1.sum(1).unsqueeze(1)
            if args.bow_norm:
                normalized_data_batch_1 = data_batch_1 / sums_1
            else:
                normalized_data_batch_1 = data_batch_1

            theta, _ = m.get_theta(normalized_data_batch_1)
            ## get prediction loss using second half
            data_batch_2 = data.get_batch(test_2_tokens, test_2_counts, ind, args.vocab_size, device)
            sums_2 = data_batch_2.sum(1).unsqueeze(1)
            res = torch.mm(theta, beta)
            preds = torch.log(res)
            recon_loss = -(preds * data_batch_2).sum(1)
            
            loss = recon_loss / sums_2.squeeze()
            loss = loss.mean().item()
            acc_loss += loss
            cnt += 1

        cur_loss = acc_loss / cnt
        ppl_dc = round(math.exp(cur_loss), 1)
        print('*'*100)
        print('{} Doc Completion PPL: {}'.format(source.upper(), ppl_dc))
        print('*'*100)
        if tc or td:
            beta = beta.data.cpu().numpy()
            if tc:
                print('Computing topic coherence...')
                get_topic_coherence(beta, train_tokens, vocab)
            if td:
                print('Computing topic diversity...')
                get_topic_diversity(beta, 25)

        all_indices = torch.split(torch.tensor(range(len(all_data_counts))),eval_batch_size)
        all_theta =[]
        # doc_ids_w_pseudo = []
        # for batch_ndx in all_indices:
        #   doc_ids_w_pseudo = np.append(doc_ids_w_pseudo,batch_ndx.numpy().astype(int))

        for idx, ind in enumerate(all_indices):
          
          data_batch_1 = data.get_batch(all_data_tokens,all_data_counts,ind, args.vocab_size, device)
          sums_1 = data_batch_1.sum(1).unsqueeze(1)
          if args.bow_norm:
              normalized_data_batch_1 = data_batch_1 / sums_1
          else:
              normalized_data_batch_1 = data_batch_1
          theta, _ = m.get_theta(normalized_data_batch_1)
          theta_cpu = theta.data.detach().cpu().numpy()
          all_theta.extend(theta_cpu)
        return ppl_dc,all_theta,m.get_beta()

if args.mode == 'train':
    ## train model on data 
    best_epoch = 0
    best_val_ppl = 1e9
    all_val_ppls = []
    print('\n')
    print('Visualizing model quality before training...')
    visualize(model)
    print('\n') 
    tstart = time()
    for epoch in range(1, args.epochs):
        train(epoch)
        val_ppl,all_theta,beta = evaluate(model, 'val')
        if val_ppl < best_val_ppl:
            # with open(ckpt, 'wb') as f:
            #     torch.save(model, f)
            best_epoch = epoch
            best_val_ppl = val_ppl
        else:
            ## check whether to anneal lr
            lr = optimizer.param_groups[0]['lr']
            if args.anneal_lr and (len(all_val_ppls) > args.nonmono and val_ppl > min(all_val_ppls[:-args.nonmono]) and lr > 1e-5):
                optimizer.param_groups[0]['lr'] /= args.lr_factor
        if epoch % args.visualize_every == 0:
            visualize(model)
        all_val_ppls.append(val_ppl)
    tstop = time()
    
    # with open(ckpt, 'rb') as f:
    #     model = torch.load(f)
    # model = model.to(device)
    val_ppl,all_theta,beta = evaluate(model, 'val')

    print('---'*30+'\n\n')
    print('runtime: -',tstop-tstart,'s\n\n')
    print('---'*30+'\n\n')
   
    deep_dir = '/SavedOutput/'+data_name+"/short/topics_"+str(args.num_topic)+"/qs_"+str(queryset)+"/run_"+str(args.run)
    saved_dir = save_dir_no_bkp+deep_dir
    os.makedirs(saved_dir,exist_ok=True)
    os.chdir(saved_dir)
    torch.save(model.state_dict(),args.dataset+"_"+str(args.num_topic)+".pt")

    
    model_signature=model_name+'_'+data_name+'_'+dtype+'_'+str(args.num_topic)+'_'+str(args.run)       
    with open("results_"+args.dataset+"_"+str(args.num_topic)+".txt","w") as f:
      beta_np = beta.data.cpu().numpy()
      theta_all_np = np.asarray(all_theta)
     
      f.write('---'*30+'\n\n')
      f.write('runtime: - '+str(tstop-tstart)+'s\n\n')
      f.write('theta_shape: - '+str(theta_all_np.shape)+'\n\n')
      f.write('beta_shape: - '+str(beta_np.shape)+'\n\n')
      f.write('---'*30+'\n\n')

    bigram_coocurring_word_list = get_bigram_coocurring_word_list(data_preprocessed,keywords)
    
    print('\nRunning TSNE....\n\n')

    labels_list_ar = new_labels[idx_permute]

    pseudo_idx = np.argwhere(np.isin(labels_list_ar,keywords_as_labels)).ravel()
    not_pseudo_idx = [i for i in range(theta_all_np.shape[0]) if i not in pseudo_idx]
    print(theta_all_np.shape,pseudo_idx)

    X_w_pseudo = tsne.fit_transform(theta_all_np)
    X_w_pseudo = np.asarray(X_w_pseudo)

    query_center = X_w_pseudo[pseudo_idx]

    X = X_w_pseudo[not_pseudo_idx]
    labels_list = labels_list_ar[not_pseudo_idx]

    # pseudo_idx_in_data = [i for i in range(len(data_preprocessed),len(data_preprocessed)+len(keywords_as_labels)+1)]
    # doc_ids = np.asarray([d for d in doc_ids_w_pseudo if d not in pseudo_idx_in_data])
    # print(len(doc_ids),max(doc_ids))
   
    print('\n\nX (no pseudo) shape: ',X.shape)  

    sorted_unique_labels = sorted(set(labels_list))
    zphi = torch.zeros(args.num_topic,X.shape[1]).numpy()

    figname = data_name+"_"+dtype+"_topics_"+str(args.num_topic)+"_run_"+str(args.run)+"_qs_"+str(queryset)
    lim=20
    plot_fig(model_name,X, labels_list, zphi,lim,sorted_unique_labels,query_center,keywords_as_labels,hv_qwords=True,showtopic=True,
    bold_topics=True,remove_legend=False,show_axis=True,save=True,figname=figname)

    #*********** Quantitative ***********#
    # if show_knn:
    knn = cal_knn(X,labels_list)
    print('KNN:- ',knn)

    ############
    X_original_seq = X
    train_label = labels_list
    num_keyword = len(keywords_as_labels)
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

    zphi = torch.zeros(theta_all_np.shape[1],X.shape[1]).numpy()

    gammas = beta
    all_topics_words = []
    topics_words_str=[]
    for k in range(args.num_topic):
        gamma = gammas[k]
        top_words = list(gamma.cpu().numpy().argsort()[-args.num_words+1:][::-1])
        topic_words = [vocab[a] for a in top_words]
        all_topics_words.append(topic_words)
        topic_words_str = ' '.join(topic_words)
        topics_words_str.append(topic_words_str)
        print('Topic :'+str(k)+" >> "+topic_words_str+"\n")
    # f.write("\n\n\n\n")

    print(all_topics_words)


    sum_avg_cos = []
    for topics in all_topics_words:
      topics_wordtensors = get_embedding_tensor(topics,inj_embeddings)
      topics_wordsScores = cosine_keywords(keywords,topics_wordtensors,inj_embeddings)
      np_topics_wordsScores = list_of_tensors_to_tensor(topics_wordsScores)
      sum_avg_cos.append((np_topics_wordsScores.mean(-1)).sum(-1).item())
    sum_avg_cos = np.array(sum_avg_cos)
    print('\n\n sum_avg_cos: ',sum_avg_cos)

    all_results = {} 
    all_results['theta'] = theta_all_np
    all_results['beta'] = beta_np
    # all_results['seed'] = seed

    all_results['keywords'] = keywords
    all_results['args'] = args
    all_results['keywords_as_labels'] = keywords_as_labels
    all_results['X_w_pseudo'] = X_w_pseudo
    all_results['X'] = X
    all_results['labels_list_w_pseudo'] = labels_list_ar
    all_results['labels_list'] = labels_list 

    all_results['query_center'] = query_center
    all_results['phi'] = zphi
    all_results['doc_ids'] = idx_permute
    all_results['runtime'] = tstop-tstart

    all_results['KNN'] = knn
    all_results['aupr_DESM'] = aupr_DESM
    all_results['aupr_tfidf'] = aupr_tfidf
    if data_name == 'nfcorpus' or data_name=='opinions_twitter':
      all_results['aupr_ground_truth'] = aupr_ground_truth
    all_results['sum_avg_cos'] = sum_avg_cos
    all_results['topics'] =  all_topics_words
    all_results['bigram_coocurring_word_list'] =  bigram_coocurring_word_list
    all_results['extended_keywords_list'] =  extended_keywords_list
    compressed_pickle(all_results,model_signature+'_all_results')
    os.chdir(ETM_dir)
        
else:   
    with open(ckpt, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        results_theta_ind={'theta':[],"indices":[]}
        theta_list_notrain = []
        ind_list_notrain = []

        ## get document completion perplexities
        test_ppl,all_theta,beta = evaluate(model, 'test', tc=args.tc, td=args.td)

        ## get most used topics
        indices = torch.tensor(range(args.num_docs_train))
        indices = torch.split(indices, batch_size)
        thetaAvg = torch.zeros(1, args.num_topic).to(device)
        thetaWeightedAvg = torch.zeros(1, args.num_topic).to(device)
        cnt = 0
        for idx, ind in enumerate(indices):
            data_batch = data.get_batch(train_tokens, train_counts, ind, args.vocab_size, device)
            sums = data_batch.sum(1).unsqueeze(1)
            cnt += sums.sum(0).squeeze().cpu().numpy()
            if args.bow_norm:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch
            theta, _ = model.get_theta(normalized_data_batch)
            theta_list_notrain.append(theta)
            ind_list_notrain.append(ind)
            
            thetaAvg += theta.sum(0).unsqueeze(0) / args.num_docs_train
            weighed_theta = sums * theta
            thetaWeightedAvg += weighed_theta.sum(0).unsqueeze(0)
            if idx % 100 == 0 and idx > 0:
                print('batch: {}/{}'.format(idx, len(indices)))
        thetaWeightedAvg = thetaWeightedAvg.squeeze().cpu().numpy() / cnt
        print('\nThe 10 most used topics are {}'.format(thetaWeightedAvg.argsort()[::-1][:10]))

        ## show topics
        beta = model.get_beta()

        results_theta_ind['theta'].append(theta_list_notrain)
        results_theta_ind['indices'].append(ind_list_notrain)
        
        # save_obj(results_theta_ind,'results_theta_ind_ETM')
        topic_indices = list(np.random.choice(args.num_topic, 10)) # 10 random topics
        print('\n')
        for k in range(args.num_topic):#topic_indices:
            gamma = beta[k]
            top_words = list(gamma.cpu().numpy().argsort()[-args.num_words+1:][::-1])
            topic_words = [vocab[a] for a in top_words]
            print('Topic {}: {}'.format(k, topic_words))

        if args.train_embeddings:
            ## show etm embeddings 
            try:
                rho_etm = model.rho.weight.cpu()
            except:
                rho_etm = model.rho.cpu()
            queries = ['andrew', 'woman', 'computer', 'sports', 'religion', 'man', 'love', 
                            'intelligence', 'money', 'politics', 'health', 'people', 'family']
            print('\n')
            print('ETM embeddings...')

            # for word in queries:
            #     print('word: {} .. etm neighbors: {}'.format(word, nearest_neighbors(word, rho_etm, vocab)))
            # print('\n')
