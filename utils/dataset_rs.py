import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk, Dataset as HFDataset, load_dataset

class RsDataset(Dataset):
    def __init__(self, explain_data, index_data):
        super(RsDataset , self).__init__()
        self.explain_data = explain_data
        self.index_data = index_data
        self.users = []
        self.items = []
        self.explanations = []
        for user,explanations in index_data.items():
            self.users += [user] * len(explanations)
            self.items += list(explanations.keys())
            self.explanations += list(explanations.values())
    def __len__(self):
        return len(self.users)
    def __getitem__(self, index):

        explain_data = self.explain_data[self.explanations[index]]
        out = {
            'user'        : self.users[index],
            'item'        : self.items[index],
            'explanations': explain_data['explanation']
        }
        return out
