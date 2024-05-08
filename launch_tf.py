from model import Mamba, ModelArgs
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch
import os
import torch.nn.functional as F
from Dataset import NarrativeQADataset,NarrativeQADataset_Val,load_narrativeqa_dataset
from utils import train_tf,evaluate_tf

transformers.logging.set_verbosity_error() # Prevent Overflowing Token warning

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 5e-4
epochs = 50

train_data_qa = load_narrativeqa_dataset('train')
#print(train_data_qa[0])
val_data_qa = load_narrativeqa_dataset('validation')


tokenizer_tf = GPT2Tokenizer.from_pretrained('gpt2')
model_tf = GPT2LMHeadModel.from_pretrained('gpt2')
train_dataset_tf = NarrativeQADataset(train_data_qa, tokenizer_tf)
val_dataset_tf = NarrativeQADataset_Val(val_data_qa, tokenizer_tf)
train_loader_tf = DataLoader(train_dataset_tf, batch_size=128, shuffle=True)
val_loader_tf = DataLoader(val_dataset_tf, batch_size=128, shuffle=False)

criterion = torch.nn.CrossEntropyLoss()

optimizer_tf = torch.optim.Adam(model_tf.parameters(), lr=0.001)
model_tf.to(device)

for epoch in range(epochs):
    # Training and validation steps would go here
    
    print(f'Epoch {epoch+1}/{epochs}')
    train_loss = train_tf(model_tf, train_loader_tf, optimizer_tf, criterion, device)
    print(f'Training Loss: {train_loss}')
    val_loss, val_bleu, val_meteor,val_rouge,val_MRR = evaluate_tf(model_tf, val_loader_tf, criterion, device, tokenizer_tf)
