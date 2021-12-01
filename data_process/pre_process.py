from unicodedata import east_asian_width
import pandas as pd
import os
import sys
import xml.etree.ElementTree as ET
import re
from collections import defaultdict
import logging
from tqdm import tqdm 
import random
from sklearn.model_selection import train_test_split
RANDOM_SEED = 42
import numpy as np
import torch
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

logger = logging.getLogger(__name__)

DIR_PATH = os.getcwd()
data_path = os.path.join(DIR_PATH, 'data')
source_file = "source.xml"
target_file = "target.xml"
answer_file = "answer2.xml"
DEFAULT_SOURCE_FILE = "source.csv"
DEFTAULT_TARGET_FILE = "target.csv"
DEFAULT_LINK_FILE ="links.csv"


TOKEN = "token"
OID = "oid"
SOID = "s_oid"
TOID = "t_oid"

ID = "id"
SID = "sid"
TID = "tid"
PRED = "pred"
LABEL = "label"

TEXT = "text"
STEXT = "s_text"
TTEXT = "t_text"

INPUT_ID = "input_ids"
TK_TYPE = "token_type_ids"
ATTEN_MASK = "attention_mask"


class DataReader:

    """Creating dictionaries for sarts, tarts, and links"""
    def __init__(self, data_path, source_file = "source.xml", target_file =  "target.xml", answer_file = "answer2.xml"):
        self.ids = []
        self.text =[]
     
        self.data_path = data_path
        self.sarts = os.path.join(self.data_path, source_file)
        print(self.sarts)

        self.tarts = os.path.join(self.data_path, target_file)
        self.links = os.path.join(self.data_path, answer_file)
           
    def read_artifacts(self, file_type):
        arts = {}
        self.fname = self.sarts if file_type == "source" else self.tarts
        tree = ET.parse(self.fname)
        root = tree.getroot()
        for art in root.iter('artifact'):
            id = art.find('art_id').text
            text = art.find('art_title').text
            arts[id] = text
        return arts
        # df = pd.DataFrame({
        #     'id': self.ids,
        #     'text': self.text})
        # df.to_csv(os.path.join(data_path,file_type + '.csv'), index = False)

    def read_link_artifact(self):
        s_ids = []
        t_ids = []
        links = set()
        tree = ET.parse(self.links)
        root = tree.getroot().find('links')
        for lnk in root.iter('link'):
            s_id = lnk.find('source_artifact_id').text
            t_id = lnk.find('target_artifact_id').text

            s_ids.append(s_id)
            t_ids.append(t_id)

            links.add((s_id, t_id))
        return links

  

    def get_examples(self):
        sarts = self.read_artifacts("source")
        tarts = self.read_artifacts("target")
        links = self.read_link_artifact()
        TraceLinks(sarts,tarts, links).gen_training_data()


class TraceLinks:

    def __init__(self, s_arts, t_arts, links):

        """ Index the raw examples with numeric ids (sid and tid) and The origin id is named as s_oid,t_oid.
        :param raw_examples: A list of dictionary with keys: s_oid, s_text, t_oid, t_text"""

        self.s_index, self.t_index = dict(), dict()  # artifact index (id->text)
        self.s_text= []
        self.t_text = []
        self.labels= []
        self.rs_index, self.rt_index = (
            dict(),
            dict(),
        )  # reversed artifact index (oid->id)
        self.s2t, self.t2s = defaultdict(set), defaultdict(set)  # true links
        self.sid_cnt, self.tid_cnt, self.lk_cnt = 0, 0, 0

        for i, s_oid in enumerate(s_arts):
            self.s_index[i] = {SOID: s_oid, TOKEN: s_arts[s_oid]}
            self.rs_index[s_oid] = i
        # print (self.rs_index)
        # print (len(self.rs_index))

        for i, t_oid in enumerate(t_arts):
            self.t_index[i] = {TOID: t_oid, TOKEN: t_arts[t_oid]}
            self.rt_index[t_oid] = i
        # print (len(self.rt_index))


        for lk in links:
            s_oid, t_oid = lk
            sid, tid = self.rs_index[s_oid], self.rt_index[t_oid]
            if tid not in self.s2t[sid]:
                self.lk_cnt += 1

            self.s2t[sid].add(tid)
            self.t2s[tid].add(sid)
        
        
        # print (len(self.s2t))
    def __len__(self):
        return self.lk_cnt

    @staticmethod
    def exclude_and_sample(sample_pool, exclude, num):
        for id in exclude:
            sample_pool.remove(id)
        selected = random.choices(list(sample_pool), k=num)
        return selected

    # def build_feature_entry(self, sid, tid, label, tokenizer, model_arch, max_seq_length  = 256):
    #     s_tks = self.s_index[sid][TOKEN]
    #     t_tks = self.t_index[tid][TOKEN]

    #     pair_feature = tokenizer(
    #         text=s_tks,
    #         text_pair=t_tks,
    #         return_attention_mask=True,
    #         return_token_type_ids=True,
    #         add_special_tokens=True,
    #         padding="max_length",
    #         max_length=max_seq_length,
    #         truncation="longest_first",
    #     )

    #     entry = {SID: sid, TID: tid, "label" : label}

    #     if model_arch.endswith('single'):
    #         entry["input_ids"] = pair_feature[INPUT_ID]
    #         entry["attention_mask"] = pair_feature[ATTEN_MASK]
    #         entry["token_type_ids"] = pair_feature[TK_TYPE]
        
    #     print(entry)
    #     return entry

    # def generate_final_dataset(self, sid, tid, label):
    #     s_tks = self.s_index[sid][TOKEN]
    #     t_tks = self.t_index[tid][TOKEN]

    #     pd.DataFrame({
    #         'S_text' : s_tks,
    #         'T_tks'  : t_tks,
    #         'labels' : label
    #     })
    #     self.s_text.append(s_tks)
    #     self.t_text.append(t_tks)
    #     self.labels.append(label)


    def gen_training_data(self):
        dataset = []
        for sid in tqdm(self.s2t, desc = "gen training dataset"):
            pos_tids = self.s2t[sid]
            for pos_tid in pos_tids:
                dataset.append([self.s_index[sid][TOKEN], self.t_index[pos_tid][TOKEN], 1])
                # self.generate_final_dataset(sid, pos_tid, 1)
        
            neg_tids = TraceLinks.exclude_and_sample(
                set(self.t_index.keys()), pos_tids, num= len(pos_tids)
            )

            for n_tid in neg_tids:
                dataset.append([self.s_index[sid][TOKEN], self.t_index[n_tid][TOKEN], 0])
                # self.generate_final_dataset(sid, n_tid, 0)
            
        df = pd.DataFrame(dataset, columns=["S_text", "T_text", 'labels'])
        df.to_csv('final_data.csv', index=False)
        # return df 


      
if __name__ == "__main__":
    """Generating the csv files from xml file for further processing of raw data!"""
    DataReader(data_path).get_examples()
    df1= pd.read_csv('final_data.csv')
    df_train, df_test = train_test_split(df1, test_size = 0.2, random_state = RANDOM_SEED)
    df_val, df_test = train_test_split(df_test, test_size = 0.5, random_state = RANDOM_SEED)

    df_train.to_csv(os.path.join(data_path,'train.csv'), index = False)
    df_test.to_csv(os.path.join(data_path,'test.csv'), index = False)
    df_val.to_csv(os.path.join(data_path,'val.csv'), index = False)
    # DataReader(data_path).read_artifacts("source")
    # DataReader(data_path).read_artifacts("target")
    # DataReader(data_path).read_link_artifact()
    # DataReader(data_path).read_artifacts("target")







