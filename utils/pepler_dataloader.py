'''
    code ref: https://github.com/lileipisces/PEPLER/blob/master/utils.py
'''
import os
import re
import math
import torch
import random
import pickle
from torch.utils.data import Dataset,DataLoader

class EntityDictionary:
    def __init__(self):
        self.idx2entity = []
        self.entity2idx = {}

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def __len__(self):
        return len(self.idx2entity)


class DataLoader_Rs:
    def __init__(self, data_path, index_dir, tokenizer, seq_len):
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary()
        self.max_rating = float('-inf')
        self.min_rating = float('inf')
        self.initialize(data_path)
        self.feature_set = set()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.train, self.valid, self.test, self.user2feature, self.item2feature = self.load_data(data_path, index_dir)

    def initialize(self, data_path):
        assert os.path.exists(data_path)
        reviews = pickle.load(open(data_path, 'rb'))
        for review in reviews:
            self.user_dict.add_entity(review['user'])
            self.item_dict.add_entity(review['item'])
            rating = review['rating']
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating

    def load_data(self, data_path, index_dir):
        data = []
        reviews = pickle.load(open(data_path, 'rb'))
        for review in reviews:
            (fea, adj, tem, sco) = review['template']
            data.append({'user': self.user_dict.entity2idx[review['user']],
                         'item': self.item_dict.entity2idx[review['item']],
                         'rating': review['rating'],
                         'text': tem,
                         'feature': fea})
            self.feature_set.add(fea)

        train_index, valid_index, test_index = self.load_index(index_dir)
        train, valid, test = [], [], []
        user2feature, item2feature = {}, {}
        for idx in train_index:
            review = data[idx]
            train.append(review)
            u = review['user']
            i = review['item']
            f = review['feature']
            if u in user2feature:
                user2feature[u].append(f)
            else:
                user2feature[u] = [f]
            if i in item2feature:
                item2feature[i].append(f)
            else:
                item2feature[i] = [f]
        for idx in valid_index:
            valid.append(data[idx])
        for idx in test_index:
            test.append(data[idx])
        return train, valid, test, user2feature, item2feature

    def load_index(self, index_dir):
        assert os.path.exists(index_dir)
        with open(os.path.join(index_dir, 'train.index'), 'r') as f:
            train_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'validation.index'), 'r') as f:
            valid_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'test.index'), 'r') as f:
            test_index = [int(x) for x in f.readline().split(' ')]
        return train_index, valid_index, test_index

class Batchify:
    def __init__(self, data, batch_size=2, shuffle=False):
        u, i, r, t, self.feature = [], [], [], [], []
        for x in data:
            u.append(x['user'])
            i.append(x['item'])
            r.append(x['rating'])
            self.feature.append(x['feature'])


        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(r, dtype=torch.int64).contiguous()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        user = self.user[index]  # (batch_size,)
        item = self.item[index]
        rating = self.rating[index]

        return user, item, rating

class Dataset_Rs_Pytorch(Dataset):
    '''
        args: corpus_data: list of dict : [{'user': 3,
                    'item': 0,
                    'rating': 5,
                    'text': ' can absolutely recommend this.',
                    'feature': 'this'},
                    {'user': 6154,
                    'item': 1769,
                    'rating': 5,
                    'text': 'atrium view with great artwork and flowers is worth the visit',
                    'feature': 'artwork'},]
    '''
    def __init__(self, corpus_data):
        self.corpus_data = corpus_data
    def __len__(self):
        return len(self.corpus_data)
    def __getitem__(self,index):
        rs_data = self.corpus_data[index]
        user = torch.tensor(rs_data['user'], dtype=torch.int64).contiguous()
        item = torch.tensor(rs_data['item'], dtype=torch.int64).contiguous()
        rating = torch.tensor(rs_data['rating'], dtype=torch.int64).contiguous()
        text = rs_data['text']
        feature = rs_data['feature']
        return user, item, rating , text, feature
    