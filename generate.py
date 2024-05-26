import torch
from tqdm import tqdm

def generate(model, seed_characters, temperature, char_to_idx, idx_to_char, device, length=100, is_lstm=False):
    model.eval()
    input_seq = torch.tensor([char_to_idx[ch] for ch in seed_characters], dtype=torch.long).unsqueeze(0).to(device)
    hidden = model.init_hidden(1)
    if is_lstm:
        hidden = (hidden[0].to(device), hidden[1].to(device))
    else:
        hidden = hidden.to(device)
    
    samples = seed_characters
    
    with torch.no_grad():
        for _ in tqdm(range(length), desc='Generating'):
            output, hidden = model(input_seq, hidden)
            # print(f"Output shape: {output.shape}")  # Print the shape of the output tensor
            
            if output.dim() == 3:
                output = output[:, -1, :]  # Get the last time step's output if 3D
            elif output.dim() == 2:
                output = output[-1, :]  # Get the last element if 2D
            
            output = output / temperature  # Apply temperature
            probabilities = torch.softmax(output, dim=-1).cpu()  # Apply softmax to the last dimension
            char_idx = torch.multinomial(probabilities, 1).item()  # Sample from the distribution
            char = idx_to_char[char_idx]
            samples += char
            input_seq = torch.tensor([[char_idx]], dtype=torch.long).to(device)
    
    return samples

if __name__ == '__main__':
    from model import CharRNN, CharLSTM
    from dataset import ShakespeareDataset
    import torch

    dataset = ShakespeareDataset('shakespeare_train.txt')
    char_to_idx = dataset.char_to_idx
    idx_to_char = dataset.idx_to_char
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size = len(dataset.chars)
    hidden_size = 128
    output_size = len(dataset.chars)
    num_layers = 2

    seed_chars_list = ["PANDARUS", "BUCKINGHAM", "GLOUCESTER", "KING LEAR", "VIOLA"]
    temperature = 1.0
    length = 100

    print("RNN Generating...")
    # RNN 모델 사용
    model_rnn = CharRNN(input_size, hidden_size, output_size, num_layers).to(device)
    model_rnn.load_state_dict(torch.load('best_rnn_model.pth'))
    model_rnn.eval()

    # 5개의 RNN 샘플 생성
    for i, seed_chars in enumerate(seed_chars_list):
        generated_text_rnn = generate(model_rnn, seed_chars, temperature, char_to_idx, idx_to_char, device, length, is_lstm=False)
        with open(f'generated_text_rnn_sample_{i+1}.txt', 'w') as f:
            # f.write("seed_character : " + seed_chars)
            f.write(generated_text_rnn)

    print("LSTM Generating...")
    # LSTM 모델 사용
    model_lstm = CharLSTM(input_size, hidden_size, output_size, num_layers).to(device)
    model_lstm.load_state_dict(torch.load('best_lstm_model.pth'))
    model_lstm.eval()

    # 5개의 LSTM 샘플 생성
    for i, seed_chars in enumerate(seed_chars_list):
        generated_text_lstm = generate(model_lstm, seed_chars, temperature, char_to_idx, idx_to_char, device, length, is_lstm=True)
        with open(f'generated_text_lstm_sample_{i+1}.txt', 'w') as f:
            # f.write("seed_character : " + seed_chars)            
            f.write(generated_text_lstm)
