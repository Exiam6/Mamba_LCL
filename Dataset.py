from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch


def load_narrativeqa_dataset(split):
    dataset = load_dataset('narrativeqa', split=split)
    return dataset

class NarrativeQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['document']['summary']['text']
        question = item['question']['text']
        answer = item['answers'][0]['text'] 

        # [CLS] question [SEP] context [SEP]
        encoding = self.tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        answer_encoding = self.tokenizer.encode_plus(
            answer,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),  
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'answer_ids': answer_encoding['input_ids'].squeeze(0)
        }


class NarrativeQADataset_Val(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        context = item['document']['text']

        question = item['question']['text']
        
        answer = item['answers'][0]['text'] 

        # [CLS] question [SEP] context [SEP]
        encoding = self.tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        answer_encoding = self.tokenizer.encode_plus(
            answer,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),  
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'answer_ids': answer_encoding['input_ids'].squeeze(0)
        }
