import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, num_classes, device='cpu'):
        super(RNN, self).__init__()
        if device == 'cpu':
            self.device = device
        else:
            self.device = "cuda:0"
        self.hidden_size = 128
        self.num_layers = 3
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.1)
        # self.fc = nn.Linear(128, num_classes)
        self.fc = nn.Sequential(nn.Linear(128, 64),
                                nn.ReLU(),
                                nn.Dropout(0.1),
                                nn.Linear(64, num_classes))

    def forward(self, x):
        # initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device).float()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device).float()
        out, _ = self.lstm(x.float(), (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


if __name__ == '__main__':
    model = RNN(num_classes=2, input_size=30)

