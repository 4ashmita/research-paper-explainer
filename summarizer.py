import torch
import torch.nn.functional as F
import unicodedata
from model import MiniGPT
from tokenizer import Tokenizer

CHECKPOINT_MODEL_PATH = "mini_gpt_summarizer.pt"

checkpoint = torch.load(CHECKPOINT_MODEL_PATH, map_location="cpu")

vocab_size = checkpoint["vocab_size"]
max_length = checkpoint["max_length"]

token_to_id = checkpoint["token_to_id"]
id_to_token = checkpoint["id_to_token"]

tokenizer = Tokenizer(vocab=vocab_size)
tokenizer.token_to_id = token_to_id
tokenizer.id_to_token = id_to_token

model = MiniGPT(vocab_size=vocab_size, max_len=max_length, embed_dim=512, num_heads=8, num_layers=4)

model.load_state_dict(checkpoint["model_state_dict"])
#model.eval()

def force_clean(text):
    # Turn "Smart" characters into "Standard" characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    # Replace all whitespace (tabs, newlines, non-breaking spaces) with a single standard space
    text = " ".join(text.split())
    return text

def generate_summary(model, tokenizer, abstract, max_new_tokens=100, temperature=0.7, top_p=0.9):
    model.eval()
    
    # 1. Prepare and Clean the Prompt
    # We add a clear separator so the model knows where the summary starts
    clean_abstract = force_clean(abstract)
    prompt = f"Abstract: {clean_abstract} Summary: This research paper introduces"
    input_ids = tokenizer.encode(prompt)
    
    # 2. Strict Context Window Management
    # Your model was trained on max_len=256. 
    # We must ensure input + new tokens don't exceed this.
    if len(input_ids) > (model.max_length - max_new_tokens):
        input_ids = input_ids[-(model.max_length - max_new_tokens):]

    generated_ids = []
    eos_id = tokenizer.token_to_id.get("[EOS]", -1)
    
    # 3. Generation Loop
    for _ in range(max_new_tokens):
        # Slice current context to the model's capacity
        input_context = input_ids[-model.max_length:]
        input_tensor = torch.tensor([input_context], dtype=torch.long).to(next(model.parameters()).device)

        with torch.no_grad():
            logits = model(input_tensor) # Forward pass

        # Focus only on the logits for the very last token
        next_token_logits = logits[0, -1, :]

        # 4. Repetition Penalty
        # Heavily penalize tokens already in our summary to prevent loops
        for token_id in set(generated_ids):
            next_token_logits[token_id] -= 2.0 # Adjust this if still looping

        # 5. Nucleus (Top-P) Sampling
        # This is better than argmax for "Summary" tasks
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p)
        probs = F.softmax(filtered_logits / temperature, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()

        # Check for End of Sequence
        if next_token_id == eos_id:
            break

        input_ids.append(next_token_id)
        generated_ids.append(next_token_id)
            
    return tokenizer.decode(generated_ids)

def top_k_top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
    """
    Filters a distribution of logits using nucleus (top-p) filtering.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value
    return logits



test_abstract = "Understanding how people argue across ideological divides online is important for studying political polarization, misinformation, and content moderation. Existing datasets capture only part of this problem some preserve text but ignore interaction structure, some model structure without rich semantics, and others represent conversations without stable userlevel ideological identity. We introduce ControBench, a benchmark for controversial discourse analysis that combines heterogeneous social interaction graphs with rich textual semantics. Built from Reddit discussions on three topics, Trump, abortion, and religion, ControBench contains 7,370 users, 1,783 posts, and 26,525 interactions. The graph contains user and post nodes connected by semantically enriched edges in particular, usercommentuser edges encode both a reply and the parent comment that it responds to, preserving local argumentative context. User labels are derived from selfdeclared Reddit flairs, providing a scalable proxy for ideological identity without manual annotation. The resulting datasets exhibit low or negative adjusted homophily Trump 0.77, Abortion 0.06, Religion 0.04, reflecting the crosscutting structure of realworld debate. We evaluate graph neural networks, pretrained language models, and large language models on ControBench and observe distinct performance patterns across topics and model families, especially when ideological boundaries are ambiguous. These results position ControBench as a challenging and realistic benchmark for controversial discourse analysis."

raw_output = generate_summary(model, tokenizer, test_abstract, max_new_tokens=100)


summary_part = raw_output.lower().split("summary")[-1].strip()
    
clean_summary = summary_part.lstrip("is ").lstrip("the ").capitalize()
    
if not clean_summary.endswith("."):
    clean_summary += "."
        
print("\n--- Final Clean Summary ---")
print(clean_summary)
