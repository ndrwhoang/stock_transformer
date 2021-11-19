import os
import json
import torch
from tqdm import tqdm

from torch.utils.data import Dataset

class ClusteringDataset(Dataset):
    def __init__(self, config, mode):
        self.config = config
        data_path = self.get_data_path(mode)
        samples = self.make_samples(data_path)
        self.inputs = self.convert_sample_to_input(samples)
        self.n_sample = len(self.inputs)
        
    def get_data_path(self, mode):
        return self.config['path']['subset_test']
    
    def make_samples(self, data_path):
        raise NotImplementedError
    
    def convert_sample_to_input(self, samples):
        raise NotImplementedError
    
    def __len__(self):
        return self.n_sample
    
    def __getitem__(self, item):
        return self.inputs[item]