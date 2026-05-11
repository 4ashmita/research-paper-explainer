import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
from tokenizer import Tokenizer
from model import MiniGPT
from dataset import AbstractData

DATA_PATH = "data/summary_pairs.txt"
PRETRAINED_MODEL_PATH = "mini_gpt_model.pt"
SAVE_MODEL_PATH = "mini_gpt_summarizer.pt"

MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 5e-4
IGNORE_INDEX = -100

checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location="mps")

vocab_size = checkpoint["vocab_size"]
token_to_id = checkpoint["token_to_id"]
id_to_token = checkpoint["id_to_token"]

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

tokenizer.token_to_id = token_to_id
tokenizer.id_to_token = id_to_token

PAD_ID = tokenizer.token_to_id["[PAD]"]
BOS_ID = tokenizer.token_to_id["[BOS]"]
EOS_ID = tokenizer.token_to_id["[EOS]"]

with open(DATA_PATH, "r") as fh:
    example_text = [line.strip() for line in fh if line.strip()]

    
print("loaded summary pairs data: ", len(example_text))

def prepare_sequence(text, tokenizer, max_len):
    marker = "Summary:"

    if marker not in text:
        raise ValueError("Each example must have Summary:")
    
    parts = text.split(marker)
    abstract = parts[0].strip()
    summary = parts[1].strip()

    prompt_text = abstract + "\n" + marker
    summary_text = " " + summary

    prompt_ids = [BOS_ID] + tokenizer.encode(prompt_text)
    summary_ids = tokenizer.encode(summary_text) + [EOS_ID]

    max_summary_len = 96 
    max_prompt_len = max_len - max_summary_len
    
    prompt_ids = prompt_ids[-max_prompt_len:]
    summary_ids = summary_ids[:max_summary_len]

    full_ids = prompt_ids + summary_ids

    input_id = full_ids[:-1]
    target_id = full_ids[1:]

    prompt_len = len(prompt_ids)
    masked_target_ids = []

    for i, target in enumerate(target_id):
        if i < prompt_len - 1:
            masked_target_ids.append(IGNORE_INDEX)
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

for text in example_text:
    input_id, target_id = prepare_sequence(text, tokenizer, MAX_LENGTH)
    if all(t == IGNORE_INDEX for t in target_id):
        continue
    examples.append((input_id, target_id))

torch_dataset = AbstractData(examples)

loader = DataLoader(torch_dataset, batch_size=BATCH_SIZE, shuffle=True)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using: ", device)

model = MiniGPT(vocab_size=vocab_size, max_len=MAX_LENGTH, embed_dim=512, num_heads=8, num_layers=4, dropout=0.2).to(device)
model.load_state_dict(checkpoint["model_state_dict"])

criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_index, batch in enumerate(loader):
        input_id = batch["input_ids"].to(device)
        target_id = batch["target_id"].to(device)

        logits = model(input_id) #Runs model forward

        loss = criterion(logits.reshape(-1, vocab_size), target_id.reshape(-1)) #Calculates how wrong the model was

        optimizer.zero_grad() #Clears old gradients
        loss.backward() #Computes gradients
        optimizer.step() #Updates model weights

        total_loss += loss.item() #Adds this btach's loss to total epoch loss

        if batch_index % 20 == 0: #Print progress at every 20th batch
            print(f"Epoch: {epoch + 1}/{EPOCHS}, Batch: {batch_index}/{len(loader)}, Loss: {loss.item()}")
    
    average_loss = total_loss/len(loader) #Computes average loss for a specific epoch
    print(f"Epoch: {epoch+1}, Average Loss: {average_loss}")
    scheduler.step()

torch.save(
    {"model_state_dict": model.state_dict(),
     "token_to_id": tokenizer.token_to_id,
     "id_to_token":tokenizer.id_to_token,
     "vocab_size": vocab_size,
     "max_length": MAX_LENGTH},
     SAVE_MODEL_PATH
     )

print(f"Model saved to {SAVE_MODEL_PATH}")