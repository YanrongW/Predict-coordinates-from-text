import copy
import json
import os
import re
import sys, csv
import logging
import torch
import numpy as np
import random
import time
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import BertTokenizer
from utils.example_and_feature import InputExample, convert_example_to_feature
import math
import pandas as pd
import argparse
from tqdm import tqdm
from utils.POI import Poi
logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(self, args=None, route_data=None):
        self.args = args
        # 加载分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.args.cache_dir)
        # self.decoder_tokenizer = BertTokenizer.from_pretrained(self.args.cache_dir)
        self.route_data = route_data

    def _convert_example(self, poi):
        example = InputExample(guid=poi.id,
                               text_a=poi.name,
                               text_b=poi.adr,
                               label=[float(poi.lng), float(poi.lat)])
        feature = convert_example_to_feature(example, self.args.max_seq_length, self.tokenizer, print_info=True)
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        input_mask = torch.tensor(feature.input_mask, dtype=torch.long)
        segment_ids = torch.tensor(feature.segment_ids, dtype=torch.long)
        position_ids = torch.tensor(feature.position_ids, dtype=torch.long)
        labels = torch.tensor(feature.label, dtype=torch.float)
        return input_ids, input_mask, segment_ids, position_ids, labels


class RelationDataset(BaseDataset):
    def __init__(self, args, all_data=None):
        super(RelationDataset, self).__init__(args=args)
        self.data = all_data
        self.lengths = len(self.data)

    def _select_data(self, index):
        origin_data = json.loads(self.data[index])
        poi = Poi()
        poi.id = origin_data['id']
        poi.name = origin_data['name']
        poi.adr = origin_data['address']
        geo = origin_data['geo'].split(',')
        poi.lng = float(geo[0])
        poi.lat = float(geo[1])
        # poi.s2_coder = [x for x in origin_data['s2_code']] # s2编码
        return poi

    def __getitem__(self, index):
        poi = self._select_data(index=index)
        instance = self._convert_example(poi)
        return instance

    def __len__(self):
        return self.lengths


class RelationDataLoader():
    def __init__(self, args, dataset):
        self.args = args
        self.input_data = dataset
        self.num_labels = 1
        self.examples = self.input_data

    def get_dataloader(self, is_test=False):
        if is_test:
            self.sampler = SequentialSampler(self.input_data)
        else:
            self.sampler = RandomSampler(self.input_data)
        self.dataloader = DataLoader(self.input_data, sampler=self.sampler, batch_size=self.args.train_batch_size)
        return self.dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir",
                        default="/nfs/volume-93-2/poi_data/point/data_sample_trainval.Beijing",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--data_dir2",
                        default="/nfs/volume-93-2/poi_data/point/poi_word_compent.Beijing",
                        type=str,
                        help="The input data dir.")
    parser.add_argument("--prepare_data",
                        default="/nfs/volume-93-2/poi_data/point/prepare_all.beijing",
                        type=str,
                        help="The input data dir.")
    parser.add_argument("--cache_dir",
                        default="./../pretrain",
                        type=str,
                        help="Where do you want to store the pre-trained own_models downloaded from s3")
    parser.add_argument("--no_poi_name",
                        action='store_false',
                        help="Whether use the name in poi or not.")
    parser.add_argument("--prepare_all_pairs_inadvance",
                        action='store_true',
                        help="Whether to prepare all pairs in advance.")
    parser.add_argument("--dataset_val_ratio",
                        type=float,
                        default=0.3,
                        help="The ratio of the whole dataset for validation")
    parser.add_argument("--choose_positive_negative",
                        type=str,
                        default="all",
                        help="all, positive, negative")
    parser.add_argument("--negative_sample_lessthan",
                        action='store_true',
                        help="Only sample negative pair less than 2km.")
    parser.add_argument("--prepare_what",
                        default="positive",
                        type=str,
                        help="positive, negative, all")
    parser.add_argument("--write_pairs_file",
                        default="",
                        type=str,
                        help="prepare pairs file addr")

    args = parser.parse_args()