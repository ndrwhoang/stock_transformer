import torch
import torch.nn as nn

class BaseDecoder(nn.Module):
    def __init__(self, config):
        super(BaseDecoder, self).__init__()
        self.config = config
        
    def forward(self, batch):
        raise NotImplementedError