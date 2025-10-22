import torch 
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)
# net_p
class Residual_Network_exploitation(nn.Module):
    def __init__(self, dim, hidden_size=512, k=1, 
                 num_layers=4, use_residual=False,
                  activate="relu",norm="ln",use_dropout=False,drop_rate=0.1):
        super(Residual_Network_exploitation, self).__init__()
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        self.num_layers=num_layers
        self.use_dropout=use_dropout
        self.input_size=dim
        self.layers.append(nn.Linear(dim, hidden_size))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, k))
        if activate=="relu":
            self.activate = nn.ReLU()
        elif activate=="silu":
            self.activate=Swish()
        if norm=="ln":
            self.layer_norm=nn.LayerNorm(hidden_size,eps=1e-6)
        elif norm=="bn":
            self.layer_norm=nn.BatchNorm1d(num_features=hidden_size)
        self.dropout=nn.Dropout(p=drop_rate) 
    def forward(self, x):        
        x = self.activate(self.layers[0](x)) 
        for _ , layer in enumerate(self.layers[1:self.num_layers-1]):
            if self.use_residual:
                identity = x
                x = self.activate(layer(x))      
                x = x+identity 
                x=self.layer_norm(x) 
            else:
                x = self.activate(layer(x))
            
            if self.use_dropout:
                x=self.dropout(x)
        hiddenstates=x
        x = self.layers[-1](x)
        x = x.squeeze()
        return x, hiddenstates
# net_q 
class Residual_Network_exploration(nn.Module):
    def __init__(self, dim=512, hidden_size=512, k=1, 
                 num_layers=4, use_residual=True,
                 activate="relu",norm="ln",use_dropout=False,drop_rate=0.1):
        super(Residual_Network_exploration, self).__init__()
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.use_dropout = use_dropout
        self.input_size = dim
        self.layers.append(nn.Linear(dim, hidden_size))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, k))
        if activate=="relu":
            self.activate = nn.ReLU()
        elif activate=="silu":
            self.activate=Swish()
        if norm=="ln":
            self.layer_norm=nn.LayerNorm(hidden_size,eps=1e-6)
        elif norm=="bn":
            self.layer_norm=nn.BatchNorm1d(num_features=hidden_size)
        self.dropout=nn.Dropout(p=drop_rate) 
    def forward(self, x):
        h=x.clone()
        x=self.activate(self.layers[0](x))
        x=x+h
        x=self.activate(x)
        for _ , layer in enumerate(self.layers[1:self.num_layers-1]):
            if  self.use_residual:
                x = self.activate(layer(x))
                x = x+h
                x=self.layer_norm(x)
            else:
                x = self.activate(layer(x))
            if self.use_dropout:
                x=self.dropout(x)
        x=self.layers[-1](x) #[1]
        return x
    def init_weights_he(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu') 
            nn.init.zeros_(m.bias)

def train_net_p(model_p,train_data,epoch,lr):
    optimizer=torch.optim.Adam(model_p.parameters(),lr,weight_decay=0.0001)
    cost = nn.L1Loss()
    hidden_state_all = []
    for i in range(epoch):
        model_p.train()
        for x,y in train_data:
            x = x.to("cuda")
            y = y.to("cuda")
            x, hidden_state = model_p(x) 
            hidden_state_all.append(hidden_state)
            loss = cost(x,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    hidden_state_all = torch.cat(hidden_state_all, dim=0)
    return  hidden_state_all

def train_net_q(model_q,train_data,epoch,lr):
    optimizer=torch.optim.Adam(model_q.parameters(),lr,weight_decay=0.0001)
    cost = nn.L1Loss()
    x_all=[]
    for i in range(epoch):
        model_q.train()
        for x,y in train_data:
            x = x.to("cuda")
            y = y.to("cuda")
            x = model_q(x) 
            x = x.squeeze()
            x_all.append(x)
            loss = cost(x,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return True

class Dataset_net_p_q(Dataset):
    def __init__(self, data,y):
        super().__init__()
        self.data = data
        self.label = y

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        return x,y