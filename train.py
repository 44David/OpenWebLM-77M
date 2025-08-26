import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from model import Transformer
import time
import json
import math
import os

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(
            text, 
            max_length=self.max_length, 
            truncation=True, 
            padding='max_length', 
            return_tensors='pt'
        )
        
        input_ids = tokens['input_ids'].squeeze(0)
        
        return {
            'input_ids': input_ids[:-1],  
            'targets': input_ids[1:]      
        }

def save_checkpoint(model, optimizer, scheduler, epoch, batch_idx, loss, training_log, checkpoint_dir="checkpoints"):
    """Save training checkpoint"""
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
    
    epoch_path = os.path.join(checkpoint_dir, f'WebLM-77M-epoch-{epoch}.pth')
    torch.save(checkpoint, epoch_path)
    print(f"Checkpoint saved: {epoch_path}")
    
    latest_path = os.path.join(checkpoint_dir, 'WebLM-77M-latest.pth')
    torch.save(checkpoint, latest_path)
    
    return epoch_path

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """Load training checkpoint"""
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
    
    # Check for resume option
    resume_from_checkpoint = False
    checkpoint_path = "checkpoints/WebLM-77M-latest.pth"
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
    
    training_log = {
        'epochs': [],
        'batches': [],
        'total_time_hours': 0,
        'estimated_cost': 0,
        'final_stats': {}
    }
    
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading high-quality datasets for better language understanding...")
    
    texts = []
    target_samples = 2000000      
    print("Loading Wikipedia...")
    try:
        wiki_dataset = load_dataset('wikipedia', '20220301.en', split='train', streaming=True)
        for i, item in enumerate(wiki_dataset):
            if len(texts) >= target_samples // 3:  
                break
            if item['text'] and len(item['text'].strip()) > 100:
                clean_text = item['text'].strip()
                if len(clean_text) > 2000:
                    chunks = [clean_text[i:i+1500] for i in range(0, len(clean_text), 1200)]
                    texts.extend(chunks[:3])  
                else:
                    texts.append(clean_text)
            
            if i % 10000 == 0:
                print(f"Loaded {len(texts)} Wikipedia samples...")
                
    except Exception as e:
        print(f"Wikipedia not available: {e}")
    
    print("Loading BookCorpus...")
    try:
        book_dataset = load_dataset('bookcorpus', split='train', streaming=True)
        for i, item in enumerate(book_dataset):
            if len(texts) >= (target_samples * 2) // 3:  
                break
            if item['text'] and len(item['text'].strip()) > 100:
                clean_text = item['text'].strip()
                if len(clean_text) > 2000:
                    chunks = [clean_text[i:i+1500] for i in range(0, len(clean_text), 1200)]
                    texts.extend(chunks[:2])  
                else:
                    texts.append(clean_text)
            
            if i % 5000 == 0:
                print(f"Loaded {len(texts)} total samples (including books)...")
                
    except Exception as e:
        print(f"BookCorpus not available: {e}")
    
    if len(texts) < target_samples:
        print(f"Loading C4 to reach {target_samples} samples (currently have {len(texts)})...")
        try:
            c4_dataset = load_dataset('allenai/c4', 'en', split='train', streaming=True)
            for i, item in enumerate(c4_dataset):
                if len(texts) >= target_samples:
                    break
                if item['text'] and len(item['text'].strip()) > 100:
                    clean_text = item['text'].strip()
                    if len(clean_text) < 3000 and clean_text.count('\n') < 10:
                        texts.append(clean_text)
                
                if i % 25000 == 0:
                    print(f"Loaded {len(texts)} total samples...")
                    
        except Exception as e:
            print(f"C4 not available: {e}")
            try:
                web_dataset = load_dataset('openwebtext', split='train', streaming=True)
                for i, item in enumerate(web_dataset):
                    if len(texts) >= target_samples:
                        break
                    if item['text'] and len(item['text'].strip()) > 100:
                        clean_text = item['text'].strip()[:1500]
                        texts.append(clean_text)
                    if i % 25000 == 0:
                        print(f"Loaded {len(texts)} total samples...")
            except Exception as e2:
                print(f"Fallback dataset also failed: {e2}")
    
    print(f"Final dataset: {len(texts)} text samples")
    
    sample_text = ' '.join(texts[:100])
    sample_tokens = len(tokenizer(sample_text)['input_ids'])
    avg_tokens_per_sample = sample_tokens / 100
    estimated_total_tokens = len(texts) * avg_tokens_per_sample
    print(f"Estimated total tokens: {estimated_total_tokens/1e9:.2f}B")
    
    train_dataset = TextDataset(texts, tokenizer)
    batch_size = 16
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    print(f"Using batch size: {batch_size}")
    
    model = Transformer(d_model=512, n_layers=8).to(device)  
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params/1e6:.1f}M parameters")
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1)
    
    num_epochs = 100  
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
    
    model.train()
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        epoch_start = time.time()
        epoch_log = {
            'epoch': epoch,
            'batches': [],
            'avg_loss': 0,
            'avg_perplexity': 0,
            'time_minutes': 0
        }
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            targets = batch['targets'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    logits = model(input_ids)
                    loss = criterion(
                        logits.view(-1, logits.size(-1)), 
                        targets.view(-1)
                    )
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids)
                loss = criterion(
                    logits.view(-1, logits.size(-1)), 
                    targets.view(-1)
                )
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            total_loss += loss.item()
            perplexity = math.exp(min(loss.item(), 10)) 
            
            if batch_idx % 100 == 0:
                elapsed_hours = (time.time() - start_time) / 3600
                current_lr = scheduler.get_last_lr()[0]
                batch_log = {
                    'batch': batch_idx,
                    'loss': loss.item(),
                    'perplexity': perplexity,
                    'learning_rate': current_lr,
                    'grad_norm': grad_norm.item(),
                    'elapsed_hours': elapsed_hours,
                    'estimated_cost': elapsed_hours * 0.25
                }
                epoch_log['batches'].append(batch_log)
                
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, '
                      f'Perplexity: {perplexity:.2f}, LR: {current_lr:.6f}, '
                      f'Grad Norm: {grad_norm.item():.4f}, '
                      f'Time: {elapsed_hours:.2f}h, Cost: ${elapsed_hours * 0.25:.2f}')
            
            # Save intermediate checkpoint every 2000 batches
            if batch_idx > 0 and batch_idx % 2000 == 0:
                print(f"Saving intermediate checkpoint at epoch {epoch}, batch {batch_idx}")
                save_checkpoint(model, optimizer, scheduler, epoch, batch_idx, loss.item(), training_log)
        
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        avg_perplexity = math.exp(min(avg_loss, 10))
        epoch_time = (time.time() - epoch_start) / 60
        
        epoch_log['avg_loss'] = avg_loss
        epoch_log['avg_perplexity'] = avg_perplexity
        epoch_log['time_minutes'] = epoch_time
        training_log['epochs'].append(epoch_log)
        
        print(f'Epoch {epoch} completed in {epoch_time:.1f} min. '
              f'Average Loss: {avg_loss:.4f}, Average Perplexity: {avg_perplexity:.2f}')
        

        
        # Save epoch checkpoint
        checkpoint_path = save_checkpoint(model, optimizer, scheduler, epoch, len(train_loader)-1, avg_loss, training_log)
        
        # Early stopping if loss is very low
        if avg_loss < 1.5:
            print(f"Loss is very low ({avg_loss:.4f}), model likely converged")
            print("You can stop training or continue for more epochs.")
    
    total_time = (time.time() - start_time) / 3600
    estimated_cost = total_time * 0.25
    
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
    }, 'WebLM-77M.pth')
    
    print(f"\nTraining completed!")
    print(f"Total time: {total_time:.2f} hours")
    print(f"Estimated cost: ${estimated_cost:.2f}")
    print(f"Final loss: {avg_loss:.4f}")
    print(f"Final perplexity: {avg_perplexity:.2f}")
    print("Model saved as 'WebLM-77M.pth'")
    print("Training log saved as 'training_log.json'")
    


if __name__ == "__main__":
    main()