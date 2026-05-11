import torch
import torch.nn.functional as F
from model import MiniGPT
from tokenizer import Tokenizer

checkpoint = torch.load("mini_gpt_model.pt", map_location="cpu")

vocab_size = checkpoint["vocab_size"]
max_length = checkpoint["max_length"]

token_to_id = checkpoint["token_to_id"]
id_to_token = checkpoint["id_to_token"]

tokenizer = Tokenizer(vocab=vocab_size)
tokenizer.token_to_id = token_to_id
tokenizer.id_to_token = id_to_token

model = MiniGPT(vocab_size=vocab_size, max_len=max_length, embed_dim=512, num_heads=8, num_layers=4)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def generate(model, tokenizer, prompt, max_new_tokens=50,temperature=0.7, use_argmax=False):
    model.eval()

    input_id = tokenizer.encode(prompt)

    unk_id = tokenizer.token_to_id["[UNK]"]
    eos_id = tokenizer.token_to_id["[EOS]"]

    for _ in range(max_new_tokens):
        input_tensor = torch.tensor([input_id], dtype=torch.long)

        with torch.no_grad():
            logits = model(input_tensor)

        next_token_logits = logits[0, -1]

        next_token_logits[unk_id] = -1e9 

        if use_argmax:
            next_token_id = torch.argmax(next_token_logits).item()
        else:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()

        input_id.append(next_token_id)
        
        if next_token_id == eos_id:
            break
    
    return tokenizer.decode(input_id)

prompt = "The Expo 2025 Osaka is a world exhibition that will be held in"
output = generate(model, tokenizer, prompt, max_new_tokens=40)

print("Prompt: ", prompt)
print("Output: ", output)