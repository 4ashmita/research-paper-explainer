import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
from tokenizer import Tokenizer
from model import MiniGPT
from dataset import AbstractData

# Paths to the pretrained model, data, and where to save new model
DATA_PATH = "data/summary_pairs.txt"
PRETRAINED_MODEL_PATH = "mini_gpt_model.pt"
SAVE_MODEL_PATH = "mini_gpt_summarizer.pt"

MAX_LENGTH = 512 # Training example will be 512 tokens long
BATCH_SIZE = 4 # Will train on 4 examples at a time
EPOCHS = 10 # Go through whole dataset 10 times
LEARNING_RATE = 5e-4 # Controls how big each model update is
IGNORE_INDEX = -100 # Signal for the loss function to ignore specific index

checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location="mps") # Opens the model trained before

vocab_size = checkpoint["vocab_size"]
token_to_id = checkpoint["token_to_id"]
id_to_token = checkpoint["id_to_token"]

# Reads the tokenizer that was previously trained
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

tokenizer.token_to_id = token_to_id
tokenizer.id_to_token = id_to_token

# These store the special token IDs
PAD_ID = tokenizer.token_to_id["[PAD]"] # Padding token
BOS_ID = tokenizer.token_to_id["[BOS]"] # Beginning of Sequence
EOS_ID = tokenizer.token_to_id["[EOS]"] # End of Sequence

# Reads the summary pairs data
with open(DATA_PATH, "r") as fh:
    example_text = [line.strip() for line in fh if line.strip()]
    
print("loaded summary pairs data: ", len(example_text))

def prepare_sequence(text, tokenizer, max_len):
    # Converts raw text into a format the the model can learn from 
    marker = "Summary:"

    if marker not in text:
        raise ValueError("Each example must have Summary:")
    
    # Splits Abstract from Summary
    parts = text.split(marker) 
    abstract = parts[0].strip()
    summary = parts[1].strip()

    prompt_text = abstract + "\n" + marker
    summary_text = " " + summary

    prompt_ids = [BOS_ID] + tokenizer.encode(prompt_text) # Adds the BOS token
    summary_ids = tokenizer.encode(summary_text) + [EOS_ID] # Adds the EOS token

    max_summary_len = 96 
    max_prompt_len = max_len - max_summary_len
    
    prompt_ids = prompt_ids[-max_prompt_len:] # The prompt ids. Cuts the beginning of abstract if it is too long
    summary_ids = summary_ids[:max_summary_len] # Summary ids. Cuts the end if it is too long 

    full_ids = prompt_ids + summary_ids # The full prompt that will be used in training

    input_id = full_ids[:-1] # The ids the model sees
    target_id = full_ids[1:] # The ids the model has to guess

    prompt_len = len(prompt_ids)
    masked_target_ids = []

    for i, target in enumerate(target_id):
        if i < prompt_len - 1:
            masked_target_ids.append(IGNORE_INDEX) # The model will ignore the abstracts
        else:
            masked_target_ids.append(target)
    
    if len(input_id) > max_len:
        input_id = input_id[:max_len]
        masked_target_ids = masked_target_ids[:max_len]
    else:
        pad_length = max_len - len(input_id)
        input_id += [PAD_ID] *pad_length
        masked_target_ids += [IGNORE_INDEX] * pad_length
    
    return input_id, masked_target_ids

examples = []

# Creates the examples that will be used in training
for text in example_text:
    input_id, target_id = prepare_sequence(text, tokenizer, MAX_LENGTH)
    if all(t == IGNORE_INDEX for t in target_id):
        continue
    examples.append((input_id, target_id))

torch_dataset = AbstractData(examples) # Wraps the examples in a PyTorch dataset class

loader = DataLoader(torch_dataset, batch_size=BATCH_SIZE, shuffle=True) # Creates the DataLoader

# Chooses the device 
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using: ", device)

# Creates the MiniGPT model
model = MiniGPT(vocab_size=vocab_size, max_len=MAX_LENGTH, embed_dim=512, num_heads=8, num_layers=4, dropout=0.2).to(device)
model.load_state_dict(checkpoint["model_state_dict"])

# Creates the loss function. Cross-Entropy is cimmonly used for classification problem and the token prediction is similar to a classification: Which token should come next
criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, label_smoothing=0.1) # The label smoothing prevents the model from becoming too over confident with its predictions

# Creates the optimizer which updates the model weights based on the loss
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Gradually lowers the learning rate over time
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

for epoch in range(EPOCHS): # Loops over the dataset multiple times
    model.train() #T raining mode
    total_loss = 0

    for batch_index, batch in enumerate(loader): # Loops through all batches from the DataLoader where batch_index is the batch number and batch is a dictionary
        input_id = batch["input_ids"].to(device)
        target_id = batch["target_id"].to(device)

        logits = model(input_id) # Runs model forward

        loss = criterion(logits.reshape(-1, vocab_size), target_id.reshape(-1)) # Calculates how wrong the model was

        optimizer.zero_grad() # Clears old gradients
        loss.backward() # Computes gradients
        optimizer.step() # Updates model weights

        total_loss += loss.item() # Adds this btach's loss to total epoch loss

        if batch_index % 20 == 0: # Print progress at every 20th batch
            print(f"Epoch: {epoch + 1}/{EPOCHS}, Batch: {batch_index}/{len(loader)}, Loss: {loss.item()}")
    
    average_loss = total_loss/len(loader) #C omputes average loss for a specific epoch
    print(f"Epoch: {epoch+1}, Average Loss: {average_loss}")
    scheduler.step() # Lowers the learning rate

# Saves the new model
torch.save(
    {"model_state_dict": model.state_dict(),
     "token_to_id": tokenizer.token_to_id,
     "id_to_token":tokenizer.id_to_token,
     "vocab_size": vocab_size,
     "max_length": MAX_LENGTH},
     SAVE_MODEL_PATH
     )

print(f"Model saved to {SAVE_MODEL_PATH}")