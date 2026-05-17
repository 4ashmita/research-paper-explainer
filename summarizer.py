import torch
import torch.nn.functional as F
import unicodedata
from model import MiniGPT
from tokenizer import Tokenizer

def load_inference_objects(checkpoint_path="mini_gpt_summarizer.pt"):
    # Reconstructs the model
    checkpoint = torch.load(checkpoint_path, map_location="mps")
    
    vocab_size = checkpoint["vocab_size"]
    max_length = checkpoint["max_length"]
    token_to_id = checkpoint["token_to_id"]
    id_to_token = checkpoint["id_to_token"]

    tokenizer = Tokenizer(vocab=vocab_size) # Rebuilds the tokenizer
    tokenizer.token_to_id = token_to_id
    tokenizer.id_to_token = id_to_token

    # Creates the model
    model = MiniGPT(vocab_size=vocab_size, max_len=max_length, embed_dim=512, num_heads=8, num_layers=4)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, tokenizer

def clean_text(text):
    # Pre processes the prompt
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii') # breaks down complex characters into their base components and then ignores anything that isn't standard ASCII
    text = " ".join(text.split())
    return text

def top_k_top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
    # prevents the model from choosing "nonsense" words while keeping the output creative
    sorted_logits, sorted_indices = torch.sort(logits, descending=True) # Ranks every word in the vocabulary from "most likely" to "least likely" to appear next
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) # Sums up the probabilities from the top down
    sorted_indices_to_remove = cumulative_probs > top_p  # Only look at the top cluster of words that add up to 90% of the total probability
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone() # Mutes the unlikely words by setting their scores to negative infinity
    sorted_indices_to_remove[..., 0] = 0 # Don't delete the first most likely word
    indices_to_remove = sorted_indices[sorted_indices_to_remove] # Map rejected words back to their original token ID
    logits[indices_to_remove] = filter_value # Mute bad word by dropping theor scores to negative infinity. This makes it way less likely that the model will use bad words to create the summary
    return logits # Returns cleaned and modifies list of scores

def generate_summary(model, tokenizer, abstract, max_new_tokens=100, temperature=0.7, top_p=0.9):
    # Main function that generates the summaries

    model.eval() # Puts the model in evaluation mode turning off training features like dropout
    clean_abstract = clean_text(abstract) # Pre process the data
    prompt = f"Abstract: {clean_abstract} Summary: This research paper introduces" # The prompt. The This research paper indroduces forces the model to pattern match the summary behavior learned during training
    input_ids = tokenizer.encode(prompt) # Encodes the prompt
    
    if len(input_ids) > (model.max_length - max_new_tokens): # Ensures the prompt is not too long or else there won't be any words for the actual summary
        input_ids = input_ids[-(model.max_length - max_new_tokens):]

    generated_ids = []
    eos_id = tokenizer.token_to_id.get("[EOS]", -1) # Finds the unique ID for EOS. The -1 is if it can't find the ID
    
    for _ in range(max_new_tokens): # Loops 100 times to generate 1 token per loop
        input_context = input_ids[-model.max_length:] # grabs the most recent tokens up till max_length(256)
        input_tensor = torch.tensor([input_context], dtype=torch.long).to(next(model.parameters()).device) # Converts to torch tensor matriz

        with torch.no_grad(): # Turns off its gradient-math tracking as it is mostly used during training (saves memory)
            logits = model(input_tensor)

        next_token_logits = logits[0, -1, :] # Isolates the final row of scores for the next token as that is what we care about
        for token_id in set(generated_ids): # loops over the IDs already generated and lowers the score. This is to ensure the model does not keep repeating words that were already used
            next_token_logits[token_id] -= 0.5 

        filtered_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p) # This filters the logits or scores
        probs = F.softmax(filtered_logits / temperature, dim=-1) # Divides filtered score by the temperature we set and runs them through softMax which gives real percentages that adds up to 100%
        next_token_id = torch.multinomial(probs, num_samples=1).item() # To ensure it doesn't always choose the 1st choosen ID multinomial acts like a roulette wheel and spins it once.

        if next_token_id == eos_id: # If we reach EOS that means we finished the summary early and it breaks the loop 
            break

        input_ids.append(next_token_id) # Adds it to the input ids so model can look at the id in the next ieration 
        generated_ids.append(next_token_id) # Adds it to generated so the model doesn't choose it again
            
    return tokenizer.decode(generated_ids) # Return the decoded version of the IDs so the words.

def save_summary(abstract,summaries, path):
    # Saves the summaries to the path specified
    with open(path, "w", encoding="utf-8") as f:
        for i, summary in enumerate(summaries):
            f.write("Abstract: " + abstract[i] + "\n" + "Summary: " + summary + "\n\n")