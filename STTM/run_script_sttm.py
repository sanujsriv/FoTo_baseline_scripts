import os
from time import time
import argparse

parser = argparse.ArgumentParser(description='STTM')
parser.add_argument('--m_name', type=str, default='bbc', help='bbc,searchsnippet, wos')
parser.add_argument('--data_name', type=str, default='bbc', help='bbc,searchsnippet, wos')
parser.add_argument('--num_runs', type=str, default='1', help='# run')
parser.add_argument('--num_topic', type=str, default='10', help='number of topic')
parser.add_argument('--queryset', type=int, default=1, help='queryset used to run the model')

args = parser.parse_args()

data_name = args.data_name
dtype = 'short'
run = args.num_runs
qs = args.queryset
m_name = args.m_name
num_topic = args.num_topic 

paper = 'emnlp2022'
model_name = 'STTM'
main_dir = '/home/grad16/sakumar/'+paper+'/'
STTM_dir = main_dir+model_name
data_dir = main_dir+"/dataset/content/STTM_data"

home_dir = '/home/student_no_backup/sakumar/'+paper+'/'+model_name
save_dir_no_bkp = '/SavedOutput/'+m_name+"/"+data_name+"/short/topic_"+str(num_topic)+"/qs"+str(qs)+"/run_"+str(run)
os.makedirs(home_dir+save_dir_no_bkp,exist_ok=True)
os.chdir(home_dir+save_dir_no_bkp)

print('model - ',m_name,'data - ',data_name,'topics - ',num_topic,'qs - ', qs ,' run - ',run)

os.system("nohup lscpu > sys_config.txt")
os.system("nohup java -Xmx1G -jar "+STTM_dir+"/jar/STTM.jar \
-model "+str(m_name)+" \
-corpus "+data_dir+"/STTM_data_"+data_name+"/short"+"/qs"+str(qs)+"/STTM_docs_"+data_name+".txt \
-vectors "+data_dir+"/STTM_data_"+data_name+"/short"+"/qs"+str(qs)+"/embeddings_"+data_name+".txt \
-ntopics "+num_topic+" \
-name "+str(m_name)+"_"+data_name+"_numtopic_"+num_topic+"_run_"+str(run)+"_qs_"+str(qs)+" \
> STTM_output.txt")