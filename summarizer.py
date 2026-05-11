import torch
import torch.nn.functional as F
import unicodedata
from model import MiniGPT
from tokenizer import Tokenizer

def load_inference_objects(checkpoint_path="mini_gpt_summarizer.pt"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
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
    
    return model, tokenizer

def force_clean(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = " ".join(text.split())
    return text

def top_k_top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value
    return logits

def generate_summary(model, tokenizer, abstract, max_new_tokens=100, temperature=0.7, top_p=0.9):
    model.eval()
    clean_abstract = force_clean(abstract)
    prompt = f"Abstract: {clean_abstract} Summary: This research paper introduces"
    input_ids = tokenizer.encode(prompt)
    
    if len(input_ids) > (model.max_length - max_new_tokens):
        input_ids = input_ids[-(model.max_length - max_new_tokens):]

    generated_ids = []
    eos_id = tokenizer.token_to_id.get("[EOS]", -1)
    
    for _ in range(max_new_tokens):
        input_context = input_ids[-model.max_length:]
        input_tensor = torch.tensor([input_context], dtype=torch.long).to(next(model.parameters()).device)

        with torch.no_grad():
            logits = model(input_tensor)

        next_token_logits = logits[0, -1, :]
        for token_id in set(generated_ids):
            next_token_logits[token_id] -= 2.0 

        filtered_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p)
        probs = F.softmax(filtered_logits / temperature, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()

        if next_token_id == eos_id:
            break

        input_ids.append(next_token_id)
        generated_ids.append(next_token_id)
            
    return tokenizer.decode(generated_ids)