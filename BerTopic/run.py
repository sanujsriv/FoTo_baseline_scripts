import os
import bz2
import pickle
from bertopic import BERTopic
import pickle5
import _pickle as cPickle
import time
import argparse

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_obj_pkl5(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle5.load(f)



parser = argparse.ArgumentParser(description='running time...')
parser.add_argument('--data_name', type=str, default='newscategory')
args = parser.parse_args()

d_data = args.data_name
os.chdir('/home/grad16/sakumar/emnlp2022/dataset/content/data_'+d_data+'/short')
docs = load_obj_pkl5('data_preprocessed_'+d_data+'_short')

start = time.time()
topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True)
topics, probs = topic_model.fit_transform(docs)

freq = topic_model.get_topic_info()
print(freq.head(20))
print(time.time()-start)


# %cd /home/grad16/sakumar/emnlp2022/dataset/content/data_bbc/short
# docs = load_obj_pkl5('data_preprocessed_bbc_short')
