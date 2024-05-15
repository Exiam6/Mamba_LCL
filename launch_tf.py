from model import Mamba, ModelArgs
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch
import os
import torch.nn.functional as F
from Dataset import NarrativeQADataset,NarrativeQADataset_Val,load_narrativeqa_dataset
from utils import train_tf,evaluate_tf
import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def main(rank, world_size):
    setup(rank, world_size)
    transformers.logging.set_verbosity_error()  # Correctly used to reduce logging verbosity
    device = torch.device(f'cuda:{rank}')  # Ensures device is correctly set per rank
    learning_rate = 5e-4
    epochs = 50

    train_data_qa = load_narrativeqa_dataset('train')
    val_data_qa = load_narrativeqa_dataset('validation')

    tokenizer_tf = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model_tf = GPT2LMHeadModel.from_pretrained('gpt2-medium').to(device)
    model_tf = torch.nn.parallel.DistributedDataParallel(model_tf, device_ids=[rank])

    # Corrected variable names for tokenizers and samplers
    train_dataset_tf = NarrativeQADataset(train_data_qa, tokenizer_tf)
    val_dataset_tf = NarrativeQADataset(val_data_qa, tokenizer_tf)

    # Corrected sampler variables to match the correct datasets
    train_sampler = DistributedSampler(train_dataset_tf, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset_tf, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader_tf = DataLoader(train_dataset_tf, batch_size=32, shuffle=False, sampler=train_sampler)
    val_loader_tf = DataLoader(val_dataset_tf, batch_size=32, shuffle=False, sampler=val_sampler)

    optimizer_tf = torch.optim.Adam(model_tf.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()  # Define criterion, assuming CrossEntropyLoss for this model

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)  # Ensure proper shuffling
        print(f'Rank {rank}, Epoch {epoch+1}/{epochs}')

        train_loss = train_tf(model_tf, train_loader_tf, optimizer_tf, criterion, device)
        print(f'Training Loss: {train_loss}')
        val_loss, val_bleu, val_meteor, val_rouge = evaluate_tf(model_tf, val_loader_tf, criterion, device, tokenizer_tf)
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # Assumes all GPUs on a single node
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)
