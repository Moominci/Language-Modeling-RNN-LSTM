import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from tqdm import tqdm
from dataset import ShakespeareDataset
from model2 import CharRNN, CharLSTM
import matplotlib.pyplot as plt

def train(model, trn_loader, device, criterion, optimizer, is_lstm=False):
    model.train()
    trn_loss = 0
    for input_seq, target_seq in tqdm(trn_loader, desc='Training', leave=False):
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        hidden = model.init_hidden(input_seq.size(0))
        if is_lstm:
            hidden = (hidden[0].to(device), hidden[1].to(device))
        else:
            hidden = hidden.to(device)
        
        optimizer.zero_grad()
        output, hidden = model(input_seq, hidden)
        if output.dim() == 3:
            output = output.reshape(-1, output.size(2))  # (batch_size * seq_length, num_classes)
        
        loss = criterion(output, target_seq.view(-1))
        loss.backward()
        optimizer.step()
        
        trn_loss += loss.item()
    return trn_loss / len(trn_loader)

def validate(model, val_loader, device, criterion, is_lstm=False):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for input_seq, target_seq in tqdm(val_loader, desc='Validation', leave=False):
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            hidden = model.init_hidden(input_seq.size(0))
            if is_lstm:
                hidden = (hidden[0].to(device), hidden[1].to(device))
            else:
                hidden = hidden.to(device)
            
            output, hidden = model(input_seq, hidden)
            if output.dim() == 3:
                output = output.reshape(-1, output.size(2))  # (batch_size * seq_length, num_classes)
            
            loss = criterion(output, target_seq.view(-1))
            val_loss += loss.item()
    return val_loss / len(val_loader)



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ShakespeareDataset('shakespeare_train.txt')
    
    dataset_size = len(dataset)
    val_split = int(0.1 * dataset_size)
    trn_split = dataset_size - val_split
    train_data, val_data = random_split(dataset, [trn_split, val_split])

    trn_loader = DataLoader(train_data, batch_size=64, sampler=SubsetRandomSampler(range(trn_split)))
    val_loader = DataLoader(val_data, batch_size=64, sampler=SubsetRandomSampler(range(val_split)))

    input_size = len(dataset.chars)
    hidden_size = 128
    output_size = len(dataset.chars)
    num_layers = 2

    model_rnn = CharRNN(input_size, hidden_size, output_size, num_layers).to(device)
    model_lstm = CharLSTM(input_size, hidden_size, output_size, num_layers).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer_rnn = optim.AdamW(model_rnn.parameters(), lr=0.002)
    optimizer_lstm = optim.AdamW(model_lstm.parameters(), lr=0.002)

    num_epochs = 10
    rnn_trn_losses, rnn_val_losses = [], []
    lstm_trn_losses, lstm_val_losses = [], []

    best_rnn_val_loss = float('inf')
    best_lstm_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        rnn_trn_loss = train(model_rnn, trn_loader, device, criterion, optimizer_rnn)
        rnn_val_loss = validate(model_rnn, val_loader, device, criterion)
        rnn_trn_losses.append(rnn_trn_loss)
        rnn_val_losses.append(rnn_val_loss)

        if rnn_val_loss < best_rnn_val_loss:
            print('Best RNN model Saved')
            best_rnn_val_loss = rnn_val_loss
            torch.save(model_rnn.state_dict(), 'best_rnn_model.pth')

        lstm_trn_loss = train(model_lstm, trn_loader, device, criterion, optimizer_lstm, is_lstm=True)
        lstm_val_loss = validate(model_lstm, val_loader, device, criterion, is_lstm=True)
        lstm_trn_losses.append(lstm_trn_loss)
        lstm_val_losses.append(lstm_val_loss)

        if lstm_val_loss < best_lstm_val_loss:
            print('Best LSTM model Saved')
            best_lstm_val_loss = lstm_val_loss
            torch.save(model_lstm.state_dict(), 'best_lstm_model.pth')

        print(f'RNN Train Loss: {rnn_trn_loss:.4f}, RNN Val Loss: {rnn_val_loss:.4f}')
        print(f'LSTM Train Loss: {lstm_trn_loss:.4f}, LSTM Val Loss: {lstm_val_loss:.4f}')

    plt.figure()
    plt.plot(rnn_trn_losses, label='RNN Train Loss')
    plt.plot(rnn_val_losses, label='RNN Val Loss')
    plt.plot(lstm_trn_losses, label='LSTM Train Loss')
    plt.plot(lstm_val_losses, label='LSTM Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.jpg')

if __name__ == '__main__':
    main()
