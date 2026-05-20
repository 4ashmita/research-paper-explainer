import torch
import torch.nn.functional as F
from model import MiniGPT
from tokenizer import Tokenizer

# Loads the model and important info
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
    # This generates the actual text from the model
    model.eval()

    input_id = tokenizer.encode(prompt) # Converts the input into tokens

    # Gets the special ids for UNK and EOS
    unk_id = tokenizer.token_to_id["[UNK]"] 
    eos_id = tokenizer.token_to_id["[EOS]"]

    for _ in range(max_new_tokens): # Loop to generate one word at a time
        input_tensor = torch.tensor([input_id], dtype=torch.long) # Wraps list in PyTorch tensor

        with torch.no_grad(): # Tells torch not to calculate the gradients
            logits = model(input_tensor) # Model looks at the sequence and produces a score for every single word in its 40k vocabulary

        next_token_logits = logits[0, -1] # Scores for very last word in sequence

        next_token_logits[unk_id] = -1e9 # manually set the score of the [UNK] token to be extremely low so the model never chooses it.

        if use_argmax: # Greedy search -> choose the word with the highest score
            next_token_id = torch.argmax(next_token_logits).item()
        else: # Or use sampling. Temp: if temp is greater than 1 the scores become more equal, but if it is less than 1, the model ecomes more confident and focused 
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()

        input_id.append(next_token_id)
        
        if next_token_id == eos_id:
            break
    
    return tokenizer.decode(input_id)

prompt = "Large language models are increasingly deployed as autonomous"
output = generate(model, tokenizer, prompt, max_new_tokens=40)

print("Prompt: ", prompt)
print("Output: ", output)