import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, output_size)  # Adjust the input size of the fully connected layer

    def forward(self, input, hidden):
        input = nn.functional.one_hot(input, num_classes=self.fc.out_features).float()
        output, hidden = self.rnn(input, hidden)
        output = self.fc(output.reshape(-1, output.size(2)))
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)  # Adjust hidden size for unidirectional

class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, output_size)  # Adjust the input size of the fully connected layer

    def forward(self, input, hidden):
        input = nn.functional.one_hot(input, num_classes=self.fc.out_features).float()
        output, hidden = self.lstm(input, hidden)
        output = self.fc(output.reshape(-1, output.size(2)))
        return output, hidden

    def init_hidden(self, batch_size):
        # Adjust hidden size for unidirectional
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))
