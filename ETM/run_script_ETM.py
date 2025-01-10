import os
from time import time
import argparse

parser = argparse.ArgumentParser(description='PLSV')
parser.add_argument('--data_name', type=str, default='bbc', help='bbc,searchsnippet, wos')
parser.add_argument('--dtype', type=str, default='short', help='full,short,small')
parser.add_argument('--num_topic', type=str, default='10', help='number of topic')
parser.add_argument('--num_runs', type=str, default='1', help='# run')
parser.add_argument('--queryset', type=int, default=1, help='queryset used to run the model')

args = parser.parse_args()

data_name = args.data_name
dtype = 'short'
num_topic =args.num_topic
num_runs = [args.num_runs]
drop = 0.2
queryset = args.queryset
epochs = '1000'

paper = 'emnlp2022'
model_name = 'ETM'
ETM_dir = '/home/grad16/sakumar/'+paper+'/'+model_name
data_path_ETM_home = '/home/grad16/sakumar/'+paper+'/dataset/content/ETM_data_'+data_name
save_dir_no_bkp = '/home/student_no_backup/sakumar/'+paper+'/'+model_name
# save_dir_no_bkp = ETM_dir

os.chdir(ETM_dir)
## script
for r in num_runs:
  data_path_ETM = data_path_ETM_home+"/short/qs"+str(queryset)
  emb_path_ETM = data_path_ETM+'/embeddings_'+data_name+".txt"
  dataset_name_ETM = data_name
  
  save_dir = save_dir_no_bkp+"/SavedOutput/"+data_name+"/"+dtype+"/topics_"+str(num_topic)+"/qs_"+str(queryset)+"/run_"+str(r)
  os.makedirs(save_dir,exist_ok=True)
  os.system("nohup python3 main.py \
  --mode train \
  --dataset "+dataset_name_ETM+" --data_path "+data_path_ETM+" \
  --emb_path "+emb_path_ETM+" --run "+str(r)+" --queryset "+str(queryset)+" \
  --num_topic "+str(num_topic)+" --train_embeddings 0 --batch_size 1000 --epochs "+epochs+" > " \
  +save_dir+"/"+"output_"+data_name+"_"+str(num_topic)+".txt")