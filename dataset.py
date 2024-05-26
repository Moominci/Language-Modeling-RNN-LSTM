import torch
from torch.utils.data import Dataset

class ShakespeareDataset(Dataset):
    """ Shakespeare dataset

    Args:
        input_file: txt file
    """

    def __init__(self, input_file):
        # Read the input file
        with open(input_file, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        # Create a character dictionary
        self.chars = sorted(set(self.text))
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.idx_to_char = {idx: ch for ch, idx in self.char_to_idx.items()}

        # Convert all characters in the text to their respective indices
        self.data = [self.char_to_idx[ch] for ch in self.text]

        # Define the sequence length
        self.seq_length = 30

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # Get input sequence and target
        input_seq = self.data[idx:idx + self.seq_length]
        target_seq = self.data[idx + 1:idx + self.seq_length + 1]
        
        # Convert to torch tensors
        input_seq = torch.tensor(input_seq, dtype=torch.long)
        target_seq = torch.tensor(target_seq, dtype=torch.long)
        
        return input_seq, target_seq

if __name__ == '__main__':
    dataset = ShakespeareDataset('shakespeare_train.txt')
    
    print(f"전체 문자 수: {len(dataset.text)}")
    print(f"고유 문자 수: {len(dataset.chars)}")
    print(f"처음 100개의 문자: {dataset.text[:100]}")
    print(f"처음 100개의 인덱스: {dataset.data[:100]}")
    print(f"문자 딕셔너리: {dataset.char_to_idx}")
    
    input_seq, target_seq = dataset[0]
    print(f"입력 시퀀스: {input_seq}")
    print(f"타겟 시퀀스: {target_seq}")
