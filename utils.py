import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import numpy as np
import os
from rouge import Rouge
import torch.nn.functional as F
from tqdm import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    count=0
    for batch in tqdm(data_loader):
        if count>=50:
            break
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        answer_ids = batch['answer_ids'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)


        loss = criterion(outputs.view(-1, outputs.size(-1)), answer_ids.view(-1))
        loss.backward()
        optimizer.step()
        count+=1
        total_loss += loss.item()

    return total_loss / len(data_loader)

def evaluate(model, data_loader, criterion, device, tokenizer):
    model.eval()
    total_loss = 0
    bleu_scores = []
    meteor_scores = []
    rouge_scores = []
    mrr_scores = []
    count=0
    rouge = Rouge()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            if count>=10:
                break
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            answer_ids = batch['answer_ids'].to(device)

            outputs = model(input_ids)
            #print(outputs.shape,answer_ids.shape) torch.Size([8, 512, 50280]) torch.Size([8, 512])
            loss = criterion(outputs.view(-1, outputs.size(-1)), answer_ids.view(-1))
            total_loss += loss.item()
            count+=1
            outputs = torch.argmax(outputs, dim=-1)
            for output, answer_id in zip(outputs, answer_ids):
                ref = tokenizer.decode(answer_id, skip_special_tokens=True)
                pred = tokenizer.decode(output, skip_special_tokens=True)

                ref_words = ref.split()
                pred_words = pred.split()
                #print(ref_words.size(),pred_words.size())
                bleu = sentence_bleu([ref_words], pred_words, smoothing_function=SmoothingFunction().method1)
                met = meteor_score([ref_words], pred_words)
                bleu_scores.append(bleu)
                meteor_scores.append(met)

                try:
                    rouge_score = rouge.get_scores(pred, ref)[0]
                    rouge_f1 = rouge_score['rouge-l']['f']
                    rouge_scores.append(rouge_f1)
                except ValueError:
                    rouge_scores.append(0)

                # MRR for multiple correct answers, adjust accordingly)
                rank = 1 if pred == ref else 0
                mrr_score = 1.0 / (rank + 1) if rank else 0
                mrr_scores.append(mrr_score)

    avg_loss = total_loss / len(data_loader)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_meteor = sum(meteor_scores) / len(meteor_scores)
    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    avg_mrr = sum(mrr_scores) / len(mrr_scores)
    print(f"Avg Loss: {avg_loss}, Avg BLEU: {avg_bleu}, Avg METEOR: {avg_meteor}, Avg ROUGE: {avg_rouge}, Avg MRR: {avg_mrr}")
    return avg_loss, avg_bleu, avg_meteor, avg_rouge, avg_mrr


def generate(model,tokenizer,prompt: str,n_tokens: int = 50,
            sample: bool = True,
            top_k: int = 40,device='cuda'):
    model.eval()
    model.to(device)
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    for _ in range(n_tokens):
        with torch.no_grad():
            indices_to_input = input_ids
            next_token_logits = model(indices_to_input)[:, -1]

        probs = F.softmax(next_token_logits, dim=-1)
        (batch, vocab_size) = probs.shape

        if top_k is not None:
            (values, indices) = torch.topk(probs, k=top_k)
            probs[probs < values[:, -1, None]] = 0
            probs = probs / probs.sum(axis=1, keepdims=True)

        if sample:
            next_indices = torch.multinomial(probs, num_samples=1)
        else:
            next_indices = torch.argmax(probs, dim=-1)[:, None]

        input_ids = torch.cat([input_ids, next_indices], dim=1)

    output_completions = [tokenizer.decode(output.tolist()) for output in input_ids][0]

    return output_completions

def train_tf(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    count=0
    for batch in tqdm(data_loader):
        if count>=50:
            break
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        answer_ids = batch['answer_ids'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = criterion(logits.view(-1, logits.size(-1)), answer_ids.view(-1))
        loss.backward()
        optimizer.step()
        count+=1
        total_loss += loss.item()

    return total_loss / count

def evaluate_tf(model, data_loader, criterion, device, tokenizer):
    model.eval()
    total_loss = 0
    bleu_scores = []
    meteor_scores = []
    rouge_scores = []
    mrr_scores = []
    count=0
    rouge = Rouge()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            if count>=10:
                break
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            answer_ids = batch['answer_ids'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            #print(outputs.shape,answer_ids.shape) torch.Size([8, 512, 50280]) torch.Size([8, 512])
            loss = criterion(logits.view(-1, logits.size(-1)), answer_ids.view(-1))
            total_loss += loss.item()
            if input_ids.max() >= tokenizer.vocab_size:
                print(f"Batch contains input_ids exceeding vocab size {input_ids.max()}")
            count+=1
            outputs = torch.argmax(logits, dim=-1)
            for output, answer_id in zip(outputs, answer_ids):
                ref = tokenizer.decode(answer_id, skip_special_tokens=True)
                pred = tokenizer.decode(output, skip_special_tokens=True)

                ref_words = ref.split()
                pred_words = pred.split()
                #print(ref_words.size(),pred_words.size())
                bleu = sentence_bleu([ref_words], pred_words, smoothing_function=SmoothingFunction().method1)
                met = meteor_score([ref_words], pred_words)
                bleu_scores.append(bleu)
                meteor_scores.append(met)

                try:
                    rouge_score = rouge.get_scores(pred, ref)[0]
                    rouge_f1 = rouge_score['rouge-l']['f']
                    rouge_scores.append(rouge_f1)
                except ValueError:
                    rouge_scores.append(0)

                # MRR for multiple correct answers, adjust accordingly)
                rank = 1 if pred == ref else 0
                mrr_score = 1.0 / (rank + 1) if rank else 0
                mrr_scores.append(mrr_score)

    avg_loss = total_loss / count
    avg_bleu = sum(bleu_scores) / count
    avg_meteor = sum(meteor_scores) / count
    avg_rouge = sum(rouge_scores) / count
    avg_mrr = sum(mrr_scores) / count
    print(f"Avg Loss: {avg_loss}, Avg BLEU: {avg_bleu}, Avg METEOR: {avg_meteor}, Avg ROUGE: {avg_rouge}, Avg MRR: {avg_mrr}")
    return avg_loss, avg_bleu, avg_meteor, avg_rouge, avg_mrr
