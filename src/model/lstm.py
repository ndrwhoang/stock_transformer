import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer

class OFLinear(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.hidden_cell = None
        
        self.linear_1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_2 = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.ReLU()
        
    def forward(self, batch):
        (input_id, target) = batch
        out = self.activation(self.linear_1(input_id))
        out = self.linear_2(out)
        
        return out
        

class BaseLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layer = 4
        self.bs = 1
        
        self.lstm = nn.LSTM(input_size = self.input_dim, 
                            hidden_size = self.hidden_dim, 
                            num_layers = self.n_layer,
                            # batch_first = True
                            )
        self.out_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, 1)
        # self.hidden_state = torch.zeros(self.n_layer, self.bs, self.hidden_dim, device='cuda:0')
        # self.cell_state = torch.zeros(self.n_layer, self.bs, self.hidden_dim, device='cuda:0')
        # self.hidden_cell = (torch.zeros(self.n_layer,1,self.hidden_dim, device='cuda:0'),
        #                     torch.zeros(self.n_layer,1,self.hidden_dim, device='cuda:0'))
        
    def forward(self, batch):
        (input_id, target) = batch
        input_id = input_id.unsqueeze(1)
        lstm_out, _ = self.lstm(input_id)
        # lstm_out, _ = self.lstm(input_id.view(len(input_id) ,1, -1))
        # print(lstm_out.size())
        out = self.out_2(lstm_out)
        out = self.out(out)
        out = out[-1, 0, 0]
        return out
    
