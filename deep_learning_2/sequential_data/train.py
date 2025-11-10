import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import os
import random
from torch.utils.data import random_split
import re 
from collections import Counter, OrderedDict
from typing import Dict, List
from tqdm import tqdm


class TextDataset(Dataset):
    def __init__(self, root_dir, split='train', transform = None):
        self.samples = []
        self.transform = transform 

        combined_path = os.path.join(root_dir,split)
        for label_name in ['pos','neg']:
            label_dir = os.path.join(combined_path, label_name)
            label = 1 if label_name == 'pos' else 0
            for fname in os.listdir(label_dir):
                if fname.endswith('.txt'):
                    path = os.path.join(label_dir, fname)
                    self.samples.append((path, label))
        random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        if self.transform:
            text = self.transform(text)
        return text, label    

root_dir = "/Users/blaise/Documents/ML/Machine-Learning-and-Big-Data-Analytics/data/aclImdb"
train_dataset = TextDataset(root_dir=root_dir, split='train')
test_dataset = TextDataset(root_dir=root_dir, split='test')


train_dataset, valid_dataset = random_split(train_dataset, [20000,5000])


def tokenizer(text):
    text = re.sub('<[^>]*>','', text)
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub(r"[^\w']+",' ', text.lower()) + ' ' +' '.join(emoticons).replace('-','')
    tokenized = text.split()
    return tokenized


token_counts = Counter()

for line, label in train_dataset:
    tokens = tokenizer(line)
    token_counts.update(tokens)



class Vocabulary:
    def __init__(self, ordered_dict: OrderedDict, unk_index: int=None):
        # ordered dict maps token -> frequency (frequency is ignored after init)
        self.itos: List[str] = list(ordered_dict.keys()) # index -> token
        self.stoi: Dict[str,int] = {tok:idx for idx,tok in enumerate(self.itos)}
        self.unk_index = unk_index
        self.default_index = unk_index 
    
    def insert_token(self, token: str, index: int) -> None:
        """Insert a token at a specific index (shifts existing entries)"""
        if token in self.stoi:
            # token already exists -> remove old entry
            old_idx = self.stoi.pop(token)
            # shift everything after old_idx down
            for t, i in self.stoi.items():
                if i > old_idx:
                    self.stoi[t] = i-1
            self.itos = [t for t,_ in sorted(self.stoi.items(), key=lambda x: x[1])]

        # insert at the requested index
        self.itos.insert(index, token)
        self.stoi[token] = index
        # shift everything >= index up by 1
        for t,i in self.stoi.items():
            if i >= index and t!=token:
                self.stoi[t] = i+1
    
    def set_default_index(self, idx: int) -> None:
        self.default_index = idx
    
    # convenience methods 
    def __getitem__(self, token:str) -> int:
        return self.stoi.get(token, self.default_index)
    
    def __len__(self) -> int:
        return len(self.itos)
    
    def lookup_indices(self, tokens: List[str]) -> List[int]:
        return [self[t] for t in tokens]
    
    def lookup_tokens(self, indices:List[int]) -> List[str]:
        return [self.itos[i] for i in indices if i < len(self.itos)]


sorted_by_freq_tuples = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
ordered_dict =OrderedDict(sorted_by_freq_tuples)
vocab = Vocabulary(ordered_dict)
vocab.insert_token("<pad>",0)
vocab.insert_token("<unk>",1)
vocab.set_default_index(1)


text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]


def collate_batch(batch):
    label_list, text_list, lengths = [], [], []

    for _text, _label in batch:
        label_list.append(_label)
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))
    label_list = torch.tensor(label_list)
    lengths = torch.tensor(lengths)
    padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    return padded_text_list, label_list, lengths


batch_size = 128
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch, num_workers=4, persistent_workers=True, prefetch_factor=2)
valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, num_workers=4, persistent_workers=True, prefetch_factor=2)
test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

device = torch.device("mps" if torch.mps.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0
        )
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(rnn_hidden_size*2, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, text, lengths):
        out = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True)
        _, (hidden,cell) = self.rnn(out)
        out = torch.cat((hidden[-2,:,:], hidden[-1, :, :]), dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

vocab_size = len(vocab)
embed_dim = 20
rnn_hidden_size = 64
fc_hidden_size = 64
torch.manual_seed(1)

model = RNN(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size).to(device)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(dataloader):
    model.train()
    total_acc, total_loss = 0, 0
    
    loop = tqdm(dataloader, desc='Training', leave=False)
    for text_batch, label_batch, lengths in loop:
        optimizer.zero_grad()
        label_batch = label_batch.float()
        text_batch, label_batch = text_batch.to(device), label_batch.to(device)
        pred = model(text_batch, lengths)[:,0]
        loss = loss_fn(pred, label_batch)
        loss.backward()
        optimizer.step()
        total_acc += (
            (pred >= 0.5).float() == label_batch
        ).float().sum().item()
        total_loss += loss.item()*label_batch.size(0)
        loop.set_postfix({
            "batch_loss": f"{loss.item():.4f}"
        })
        
        
    return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)

def evaluate(dataloader):
    model.eval()
    total_acc, total_loss = 0, 0

    loop = tqdm(dataloader, desc='Validation', leave=False)
    with torch.no_grad():
        for text_batch, label_batch, lengths in loop:
            
            label_batch = label_batch.float()
            text_batch, label_batch = text_batch.to(device), label_batch.to(device)
            pred = model(text_batch, lengths)[:,0]
            loss = loss_fn(pred, label_batch)
            total_acc += (
                (pred >= 0.5).float() == label_batch
            ).float().sum().item()
            total_loss += loss.item()*label_batch.size(0)

            loop.set_postfix({
                "batch_loss": f"{loss.item():.4f}"
            })
    return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)

num_epochs = 10


if __name__ == "__main__":
    for epoch in range(num_epochs):
        print(f'\nStart of epoch: {epoch+1}')
        acc_train, loss_train = train(train_dl)
        acc_valid, loss_valid = evaluate(valid_dl)
        print(f'Epoch: {epoch} | accuracy: {acc_train:.4f} | val_accuracy: {acc_valid:.4f}')
    
    acc_test, _ = evaluate(test_dl)
    print(f'test accuracy: {acc_test:.4f}')
