import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(140,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,2)
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    
class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(7, 32, kernel_size=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=7,
                            hidden_size=64,
                            batch_first=True)
        self.fc = nn.Linear(64,2)
        
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        final_hidden = h_n.squeeze(0)
        x = self.fc(final_hidden)
        return x
    
if __name__ == '__main__':
    import torch
    fake_batch = torch.randn(4,20,7)
    
    #MLP test
    mlp = MLP()
    print(mlp)
    print(f'MLP output: {mlp(fake_batch).shape}')
    print(f'MLP params: {sum(p.numel() for p in mlp.parameters()):,}\n')
    
    #CNN test
    cnn = CNN1D()
    print(cnn)
    print(f'CNN output: {cnn(fake_batch).shape}')
    print(f'CNN params: {sum(p.numel() for p in cnn.parameters()):,}\n')
    
    #LSTM test
    lstm = LSTMModel()
    print(lstm)
    print(f'LSTM output: {lstm(fake_batch).shape}')
    print(f'LSTM params: {sum(p.numel() for p in lstm.parameters()):,}\n')