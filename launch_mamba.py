from model import Mamba, ModelArgs
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch
import os
import torch.nn.functional as F
from Dataset import NarrativeQADataset,NarrativeQADataset_Val,load_narrativeqa_dataset
from utils import generate,train,evaluate

transformers.logging.set_verbosity_error() # Prevent Overflowing Token warning

# One of:
#     'state-spaces/mamba-2.8b-slimpj'
#     'state-spaces/mamba-2.8b'
#     'state-spaces/mamba-1.4b'
#     'state-spaces/mamba-790m'
#     'state-spaces/mamba-370m'
#     'state-spaces/mamba-130m'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["HF_DATASETS_CACHE"] = "/scratch/zz4330/data/cache"
learning_rate = 5e-4
epochs = 50


train_data_qa = load_narrativeqa_dataset('train')
#print(train_data_qa[0])
val_data_qa = load_narrativeqa_dataset('validation')

tokenizer_Mamba = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

train_dataset_Mamba = NarrativeQADataset(train_data_qa, tokenizer_Mamba)
val_dataset_Mamba = NarrativeQADataset_Val(val_data_qa, tokenizer_Mamba)
train_loader_Mamba = DataLoader(train_dataset_Mamba, batch_size=128, shuffle=True)
val_loader_Mamba = DataLoader(val_dataset_Mamba, batch_size=128, shuffle=False)

pretrained_model_name = 'state-spaces/mamba-130m'

model_Mamba = Mamba.from_pretrained(pretrained_model_name)
model_Mamba.to(device)
optimizer_Mamba = torch.optim.Adam(model_Mamba.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

os.makedirs('/scratch/zz4330/Mamba/saved/', exist_ok=True)
for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    train_loss = train(model, train_loader, optimizer, criterion, device)
    torch.save(model.state_dict(), '/scratch/zz4330/Mamba/saved/mamba_model.pth')
    print(f'Training Loss: {train_loss}')
    val_loss, val_bleu, val_meteor,val_rouge,val_MRR = evaluate(model, val_loader, criterion, device, tokenizer)
 
