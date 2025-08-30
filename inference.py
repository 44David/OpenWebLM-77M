import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from model import Transformer
import os

class ModelChat:
    def __init__(self, model_path='WebLM-15M.pth', use_mixed_precision=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            
        print(f"Loading model on {self.device}")
        if self.use_mixed_precision:
            print("Using mixed precision inference")
        
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = Transformer(d_model=192, n_layers=8).to(self.device)
        if self.use_mixed_precision:
            self.model = self.model.half()
            
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
        
        if 'epoch' in checkpoint:
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        if 'training_time_hours' in checkpoint:
            print(f"Model trained for {checkpoint['training_time_hours']:.2f} hours")
        if 'final_loss' in checkpoint:
            print(f"Final training loss: {checkpoint['final_loss']:.4f}")
        if 'loss' in checkpoint:
            print(f"Checkpoint loss: {checkpoint['loss']:.4f}")
        if 'timestamp' in checkpoint:
            import datetime
            checkpoint_time = datetime.datetime.fromtimestamp(checkpoint['timestamp'])
            print(f"Checkpoint saved at: {checkpoint_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model loaded successfully!")
    
    @torch.inference_mode()
    def generate_text(self, prompt, max_length=100, temperature=0.8, top_k=50):
        """Generate text from a prompt with optimized inference"""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        original_length = input_ids.size(1)
        
        max_total_length = min(original_length + max_length, 1024)
        generated_ids = torch.zeros(1, max_total_length, dtype=input_ids.dtype, device=self.device)
        generated_ids[0, :original_length] = input_ids[0]
        
        current_length = original_length
        
        autocast_context = torch.cuda.amp.autocast() if self.use_mixed_precision else torch.no_grad()
        
        with autocast_context:
            for step in range(max_length):
                if current_length >= max_total_length:
                    break
                    
                current_input = generated_ids[:, :current_length]
                logits = self.model(current_input)
                
                last_logits = logits[0, -1, :] / temperature
                
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(last_logits, top_k)
                    last_logits = torch.full_like(last_logits, float('-inf'))
                    last_logits[top_k_indices] = top_k_logits
                
                probs = F.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                generated_ids[0, current_length] = next_token
                current_length += 1
        
        full_text = self.tokenizer.decode(generated_ids[0, :current_length], skip_special_tokens=True)
        prompt_text = self.tokenizer.decode(generated_ids[0, :original_length], skip_special_tokens=True)
        generated_text = full_text[len(prompt_text):]
        
        return generated_text

    def interactive_chat(self):
        print("\n" + "="*50)
        print("WebLM-15M Interactive Chat")
        print("Commands:")
        print("  'quit' or 'exit' - Exit chat")
        print("  'temp X' - Set temperature (e.g., 'temp 0.7')")
        print("  'length X' - Set max generation length")
        print("="*50 + "\n")
        
        temperature = 0.8
        max_length = 100
        
        while True:
            try:
                prompt = input("You: ").strip()
                
                if prompt.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                
                if prompt.startswith('temp '):
                    try:
                        temperature = float(prompt.split()[1])
                        print(f"Temperature set to {temperature}")
                        continue
                    except:
                        print("Invalid temperature. Use: temp 0.7")
                        continue
                
                if prompt.startswith('length '):
                    try:
                        max_length = int(prompt.split()[1])
                        print(f"Max length set to {max_length}")
                        continue
                    except:
                        print("Invalid length. Use: length 50")
                        continue
                
                if not prompt:
                    continue
                
                print("Model: ", end="", flush=True)
                response = self.generate_text(
                    prompt, 
                    max_length=max_length, 
                    temperature=temperature
                )
                print(response)
                print()  
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

def main():
    import sys
    import glob
    
    model_path = 'WebLM-15M.pth'
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    if model_path == 'WebLM-15M.pth' and not os.path.exists(model_path):
        checkpoint_files = glob.glob('checkpoints/WebLM-15M-epoch-*.pth')
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: int(x.split('-epoch-')[1].split('.')[0]))
            print("Available epoch checkpoints:")
            for i, checkpoint in enumerate(checkpoint_files):
                epoch_num = checkpoint.split('-epoch-')[1].split('.')[0]
                print(f"  {i+1}. {checkpoint} (Epoch {epoch_num})")
            
            try:
                choice = input(f"Select checkpoint (1-{len(checkpoint_files)}) or press Enter for latest: ").strip()
                if choice:
                    model_path = checkpoint_files[int(choice)-1]
                else:
                    model_path = checkpoint_files[-1]  # Latest
                print(f"Using checkpoint: {model_path}")
            except (ValueError, IndexError):
                print("Invalid choice, using latest checkpoint")
                model_path = checkpoint_files[-1]
    
    try:
        chat = ModelChat(model_path)
        chat.interactive_chat()
    except FileNotFoundError:
        print(f"Error: {model_path} not found!")
        print("Make sure you've trained the model first with train.py")
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    main()