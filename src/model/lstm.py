import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

class BaseLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layer = 1
        self.bs = 1
        
        self.lstm = nn.LSTM(input_size = self.input_dim, 
                            hidden_size = self.hidden_dim, 
                            num_layers = self.n_layer,
                            batch_first = True)
        self.out = nn.Linear(self.hidden_dim, 1)
        
    def forward(self, batch):
        (input_id, _) = batch
        input_ = input_id[None, :, None]
        self.hidden_state = torch.zeros(self.n_layer, self.bs, self.hidden_dim)
        self.cell_state = torch.zeros(self.n_layer, self.bs, self.hidden_dim)
        lstm_out, (self.hidden_state, self.cell_state) = self.lstm(input_, (self.hidden_state, self.cell_state))
        out = self.out(lstm_out)
        out = out.squeeze(0)
        
        return out[-1, :]
    
