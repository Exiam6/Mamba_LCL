from model import Mamba, ModelArgs
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch
import os
import transformers
import torch.nn.functional as F
from Dataset import NarrativeQADataset,NarrativeQADataset_Val,load_narrativeqa_dataset
from utils import generate,train,evaluate,setup,cleanup
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

transformers.logging.set_verbosity_error() # Prevent Overflowing Token warning


#     'state-spaces/mamba-1.4b' to 'GPT-2 XL' or "GPT3-Babbage"
#     'state-spaces/mamba-790m'  to 'GPT-2'
#     'state-spaces/mamba-370m' to 'GPT-2 Medium'
#     'state-spaces/mamba-130m' to 'GPT-2 Small'

#os.environ["HF_DATASETS_CACHE"] = "/scratch/zz4330/data/cache"

def main(rank, world_size):
    setup(rank, world_size)

    learning_rate = 1e-5
    epochs = 10
    pretrained_model_name = 'state-spaces/mamba-370m'
    device = torch.device('cuda', rank)

    train_data_qa = load_narrativeqa_dataset('train')
    val_data_qa = load_narrativeqa_dataset('validation')

    tokenizer_Mamba = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    train_dataset_Mamba = NarrativeQADataset(train_data_qa, tokenizer_Mamba)
    val_dataset_Mamba = NarrativeQADataset(val_data_qa, tokenizer_Mamba)

    train_sampler = DistributedSampler(train_dataset_Mamba, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset_Mamba, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader_Mamba = DataLoader(train_dataset_Mamba, batch_size=32, shuffle=False, sampler=train_sampler)
    val_loader_Mamba = DataLoader(val_dataset_Mamba, batch_size=32, shuffle=False, sampler=val_sampler)

    model_Mamba = Mamba.from_pretrained(pretrained_model_name).to(device)
    model_Mamba = torch.nn.parallel.DistributedDataParallel(model_Mamba, device_ids=[rank])

    optimizer_Mamba = torch.optim.Adam(model_Mamba.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs('/scratch/zz4330/Mamba/saved/', exist_ok=True)

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)  # Ensure shuffling different each epoch
        print(f'Rank {rank}, Epoch {epoch+1}/{epochs}')
        train_loss = train(model_Mamba, train_loader_Mamba, optimizer_Mamba, criterion, device)
        torch.save(model_Mamba.module.state_dict(), f'/scratch/zz4330/Mamba/saved/mamba_model_{rank}.pth')
        print(f'Training Loss: {train_loss}')
        val_loss, val_bleu, val_meteor, val_rouge = evaluate(model_Mamba, val_loader_Mamba, criterion, device, tokenizer_Mamba)

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)
