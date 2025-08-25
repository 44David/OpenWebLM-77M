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

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
    
    # Mixed precision training
    use_mixed_precision = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    print(f"Mixed precision: {use_mixed_precision}")
    
    # cost tracking
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
    
  
    print("Loading OpenWebText subset-00 dataset...")
    dataset = load_dataset('finnstrom3693/opewebtext-subset-00', split='train', streaming=True)
    
    texts = []
    for i, item in enumerate(dataset):
        if i >= 300000:
            break
        if item['text'] is not None and len(item['text'].strip()) > 0:
            texts.append(item['text'])
        
        if i % 50000 == 0:
            print(f"Loaded {i} samples...")
    
    print(f"Loaded {len(texts)} text samples from OpenWebText subset-00")
    
    train_dataset = TextDataset(texts, tokenizer)  
    batch_size = 32
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    print(f"Using batch size: {batch_size}")
    
    model = Transformer().to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)  # Lower LR
    
    # learning rate scheduler
    num_epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    model.train()
    for epoch in range(num_epochs):
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
            perplexity = math.exp(loss.item())
            
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
                    'estimated_cost': elapsed_hours * 0.526
                }
                epoch_log['batches'].append(batch_log)
                
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, '
                      f'Perplexity: {perplexity:.2f}, LR: {current_lr:.6f}, '
                      f'Grad Norm: {grad_norm.item():.4f}, '
                      f'Time: {elapsed_hours:.2f}h, Est. cost: ${elapsed_hours * 0.526:.2f}')
        
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        avg_perplexity = math.exp(avg_loss)
        epoch_time = (time.time() - epoch_start) / 60  # minutes
        
        epoch_log['avg_loss'] = avg_loss
        epoch_log['avg_perplexity'] = avg_perplexity
        epoch_log['time_minutes'] = epoch_time
        training_log['epochs'].append(epoch_log)
        
        print(f'Epoch {epoch} completed in {epoch_time:.1f} min. '
              f'Average Loss: {avg_loss:.4f}, Average Perplexity: {avg_perplexity:.2f}')
    
    total_time = (time.time() - start_time) / 3600
    estimated_cost = total_time * 0.526  # g4dn.xlarge price ( change for runpod )
    
    training_log['total_time_hours'] = total_time
    training_log['estimated_cost'] = estimated_cost
    training_log['final_stats'] = {
        'final_loss': avg_loss,
        'final_perplexity': avg_perplexity,
        'dataset_size': len(texts),
        'batch_size': batch_size,
        'mixed_precision': use_mixed_precision,
        'num_epochs': num_epochs
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
        'batch_size': batch_size
    }, 'OpenWebLM-v1-77M.pth')
    
    print(f"\nTraining completed!")
    print(f"Total time: {total_time:.2f} hours")
    print(f"Estimated cost: ${estimated_cost:.2f}")
    print(f"Final loss: {avg_loss:.4f}")
    print(f"Final perplexity: {avg_perplexity:.2f}")
    print("Model saved as 'OpenWebLM-v1-77M.pth'")
    print("Training log saved as 'training_log.json'")

if __name__ == "__main__":
    main()