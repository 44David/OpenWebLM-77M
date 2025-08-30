#!/usr/bin/env python3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import pickle
import time
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from model import Transformer

class PreTokenizedDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        print("Pre-tokenizing all texts...")
        self.input_ids = []
        self.targets = []
        
        for i, text in enumerate(texts):
            tokens = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            input_ids = tokens['input_ids'].squeeze(0)
            self.input_ids.append(input_ids[:-1])
            self.targets.append(input_ids[1:])
            
            if i % 50000 == 0:
                print(f"Tokenized {i}/{len(texts)} texts")
        
        # Convert to tensors
        self.input_ids = torch.stack(self.input_ids)
        self.targets = torch.stack(self.targets)
        print(f"Pre-tokenization complete. Shape: {self.input_ids.shape}")
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'targets': self.targets[idx]
        }

def save_checkpoint(model, optimizer, scheduler, epoch, batch_idx, loss, training_log, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'training_log': training_log,
        'timestamp': time.time()
    }
    epoch_path = os.path.join(checkpoint_dir, f'WebLM-15M-epoch-{epoch}.pth')
    torch.save(checkpoint, epoch_path)
    print(f"Checkpoint saved: {epoch_path}")
    latest_path = os.path.join(checkpoint_dir, 'WebLM-15M-latest.pth')
    torch.save(checkpoint, latest_path)
    return epoch_path

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_path, map_location=model.device if hasattr(model, 'device') else 'cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    training_log = checkpoint.get('training_log', {'epochs': [], 'batches': []})
    print(f"Resumed from epoch {checkpoint['epoch']}, batch {checkpoint['batch_idx']}")
    print(f"Previous loss: {checkpoint['loss']:.4f}")
    return start_epoch, training_log

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    resume_from_checkpoint = False
    checkpoint_path = "checkpoints/WebLM-15M-latest.pth"
    if os.path.exists(checkpoint_path):
        resume_choice = input(f"Found checkpoint at {checkpoint_path}. Resume training? (y/n): ").lower().strip()
        resume_from_checkpoint = resume_choice in ['y', 'yes']

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True

    use_mixed_precision = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    print(f"Mixed precision: {use_mixed_precision}")

    start_time = time.time()
    training_log = {'epochs': [], 'batches': [], 'total_time_hours': 0, 'estimated_cost': 0, 'final_stats': {}}

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading FineWeb dataset...")
    texts = []
    target_samples = 1_000_000  
    dataset_max_len = 512

    local_path = './fineweb_1M'
    if not os.path.exists(local_path):
        print("Downloading dataset to local storage (one-time setup)...")
        fineweb_dataset = load_dataset('HuggingFaceFW/fineweb', split='train', streaming=True)
        temp_texts = []
        for i, item in enumerate(fineweb_dataset):
            if len(temp_texts) >= target_samples:
                break
            if item.get('text') and len(item['text'].strip()) > 100:
                clean_text = item['text'].strip()
                if len(clean_text) > 2000:
                    chunks = [clean_text[j:j+1500] for j in range(0, len(clean_text), 1200)]
                    temp_texts.extend(chunks[:3])
                else:
                    temp_texts.append(clean_text)
            if i % 50000 == 0:
                print(f"Downloaded {len(temp_texts)} samples...")


        os.makedirs(local_path, exist_ok=True)
        with open(f'{local_path}/texts.pkl', 'wb') as f:
            pickle.dump(temp_texts[:target_samples], f)
        print(f"Saved {len(temp_texts[:target_samples])} samples locally")

    print("Loading from local storage...")
    with open(f'{local_path}/texts.pkl', 'rb') as f:
        texts = pickle.load(f)

    print(f"Loaded {len(texts)} text samples from local storage")

    print(f"Final dataset: {len(texts)} text samples")

    sample_text = ' '.join(texts[:100])
    sample_tokens = len(tokenizer(sample_text, truncation=True, max_length=dataset_max_len)['input_ids'])
    avg_tokens_per_sample = sample_tokens / 100
    estimated_total_tokens = len(texts) * avg_tokens_per_sample
    print(f"Estimated total tokens: {estimated_total_tokens/1e9:.2f}B")

    train_dataset = PreTokenizedDataset(texts, tokenizer, max_length=dataset_max_len)
    batch_size = 128
    gradient_accumulation_steps = 1
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True
    )
    print(f"Using batch size: {batch_size}, grad_accum_steps: {gradient_accumulation_steps}")

    model = Transformer(d_model=128, n_layers=8).to(device)
    try:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing.")
    except Exception:
        pass

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params/1e6:.1f}M parameters")

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1)
    num_epochs = 6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    start_epoch = 0
    if resume_from_checkpoint:
        try:
            start_epoch, training_log = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
            print(f"Resuming training from epoch {start_epoch}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting fresh training...")
            start_epoch = 0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.train()
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        processed_batches = 0
        epoch_start = time.time()
        epoch_log = {'epoch': epoch, 'batches': [], 'avg_loss': 0, 'avg_perplexity': 0, 'time_minutes': 0}

        optimizer.zero_grad(set_to_none=True)
        last_grad_norm = 0.0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device, non_blocking=True, memory_format=torch.contiguous_format)
            targets = batch['targets'].to(device, non_blocking=True, memory_format=torch.contiguous_format)

            with torch.amp.autocast(device_type='cuda', enabled=use_mixed_precision):
                logits = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            raw_loss = loss.detach().item()
            perplexity = math.exp(min(raw_loss, 10))

            loss = loss / gradient_accumulation_steps

            try:
                if use_mixed_precision:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"OOM at epoch {epoch} batch {batch_idx} — clearing cache and skipping batch")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    continue
                else:
                    raise

            total_loss += raw_loss
            processed_batches += 1

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_mixed_precision:
                    scaler.unscale_(optimizer)
                    last_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    last_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if batch_idx % 100 == 0:
                elapsed_hours = (time.time() - start_time) / 3600
                current_lr = scheduler.get_last_lr()[0]
                batch_log = {
                    'batch': batch_idx,
                    'loss': raw_loss,
                    'perplexity': perplexity,
                    'learning_rate': current_lr,
                    'grad_norm': float(last_grad_norm),
                    'elapsed_hours': elapsed_hours,
                    'estimated_cost': elapsed_hours * 0.26
                }
                epoch_log['batches'].append(batch_log)
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {raw_loss:.4f}, Perplexity: {perplexity:.2f}, '
                      f'LR: {current_lr:.6f}, Grad Norm: {last_grad_norm:.4f}, Time: {elapsed_hours:.2f}h, '
                      f'Cost: ${elapsed_hours * 0.26:.2f}')

        scheduler.step()

        avg_loss = total_loss / processed_batches if processed_batches > 0 else float('inf')
        avg_perplexity = math.exp(min(avg_loss, 10))
        epoch_time = (time.time() - epoch_start) / 60.0

        epoch_log['avg_loss'] = avg_loss
        epoch_log['avg_perplexity'] = avg_perplexity
        epoch_log['time_minutes'] = epoch_time
        training_log['epochs'].append(epoch_log)

        print(f'Epoch {epoch} completed in {epoch_time:.1f} min. Average Loss: {avg_loss:.4f}, '
              f'Average Perplexity: {avg_perplexity:.2f}')

        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            save_checkpoint(model, optimizer, scheduler, epoch, len(train_loader)-1, avg_loss, training_log)

        if avg_loss < 1.5:
            print(f"Loss is very low ({avg_loss:.4f}), model likely converged")

    total_time = (time.time() - start_time) / 3600.0
    estimated_cost = total_time * 0.26

    training_log['total_time_hours'] = total_time
    training_log['estimated_cost'] = estimated_cost
    training_log['final_stats'] = {
        'final_loss': avg_loss,
        'final_perplexity': avg_perplexity,
        'dataset_size': len(texts),
        'estimated_tokens': estimated_total_tokens,
        'batch_size': batch_size,
        'mixed_precision': use_mixed_precision,
        'num_epochs': num_epochs,
        'model_parameters': total_params
    }

    with open('training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'tokenizer_vocab_size': len(tokenizer),
        'final_loss': avg_loss,
        'final_perplexity': avg_perplexity,
        'training_time_hours': total_time,
        'mixed_precision': use_mixed_precision,
        'batch_size': batch_size,
        'model_parameters': total_params
    }, 'WebLM-15M.pth')

    print(f"\nTraining completed!")
    print(f"Total time: {total_time:.2f} hours")
    print(f"Estimated cost: ${estimated_cost:.2f}")
    print(f"Final loss: {avg_loss:.4f}")
    print(f"Final perplexity: {avg_perplexity:.2f}")
    print("Model saved as 'WebLM-15M.pth'")
    print("Training log saved as 'training_log.json'")

if __name__ == "__main__":
    main()