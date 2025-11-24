from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, chunks):
        self.chunks =chunks
    def __len__(self):
        return len(self.chunks)
    def __getitem__(self, index):
        series_chunk = self.chunks[index]
        return series_chunk[:-1],series_chunk[-1]
    

class TimeSeriesDataset2(Dataset):
    def __init__(self, chunks, seq_length):
        self.chunks = chunks
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, index):
        chunk = self.chunks[index]
        data, targets = chunk[:self.seq_length,:].float(), chunk[self.seq_length:,1].float()
        return data, targets