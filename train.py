import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import random

from tokenizer import Tokenizer
from dataset import AbstractData
from model import MiniGPT

DATA = "data/clean_data.txt" #Path to the clean data
MAX_LENGTH = 512 #Training example will be 512 tokens long
BATCH_SIZE = 16 #Will train on 16 examples at a time
EPOCHS = 40 #Go through whole dataset 40 times
LEARNING_RATE = 3e-4 #Controls how big each model update is

MODEL_SAVE_PATH = "mini_gpt_model.pt" #File where trained model will be saved

#Reads the data file
with open(DATA, "r") as fh:
    text = [line.strip() for line in fh if line.strip()]

print("Loaded data: ", len(text))

#Reads the tokenizer object from the previous training
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

VOCAB_SIZE = len(tokenizer.id_to_token) #Maximum number of tokens the tokenizer will keep

#These store the special token IDs
PAD_ID = tokenizer.token_to_id["[PAD]"] #Padding token
BOS_ID = tokenizer.token_to_id["[BOS]"] #Beginning of Sequence
EOS_ID = tokenizer.token_to_id["[EOS]"] #End of Sequence

vocab_size = len(tokenizer.id_to_token)
print("vocabulary size: ", vocab_size)

random.shuffle(text) # shuffles the text so it isn't in order

all_tokens = []
for t in text:
    all_tokens += [BOS_ID] + tokenizer.encode(t) + [EOS_ID]

examples = [] 
for i in range(0, len(all_tokens) - MAX_LENGTH, MAX_LENGTH):
    #take a slice of data + 1 for the target shift
    chunk = all_tokens[i : i + MAX_LENGTH + 1]
    input_id = chunk[:-1]
    target_id = chunk[1:]
    examples.append((input_id, target_id))

print(f"Packed {len(all_tokens)} tokens into {len(examples)} sequences.")

torch_data = AbstractData(examples) #Wraps the examples in a PyTorch dataset class

loader = DataLoader(torch_data, batch_size=BATCH_SIZE, shuffle=True) #Creates the DataLoader
print("Examples for Training: ", len(torch_data))

# Chooses the device 
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using: ", device)

#Creates the MiniGPT model
model = MiniGPT(vocab_size=vocab_size, max_len=MAX_LENGTH, embed_dim=512, num_heads=8, num_layers=4, dropout=0.1).to(device)

#Creates the loss function. Cross-Entropy is cimmonly used for classification problem and the token prediction is similar to a classification: Which token should come next
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID) #ignore_index=PAD_ID tells loss function to ignore padding tokens when calculating loss

#Creates the optimizer which updates the model weights based on the loss
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)

for epoch in range(EPOCHS): #Loops over the dataset multiple times
    model.train() #Training mode
    total_loss = 0

    for batch_index, batch in enumerate(loader): #Loops through all batches from the DataLoader where batch_index is the batch number and batch is a dictionary
        #Gets input and target tensors from the batch and moves them to the same device as the model
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

#Saves a checkpoint file which stores the trained model and important metadata
torch.save(
    {"model_state_dict": model.state_dict(),
     "token_to_id": tokenizer.token_to_id,
     "id_to_token":tokenizer.id_to_token,
     "vocab_size": vocab_size,
     "max_length": MAX_LENGTH},
     MODEL_SAVE_PATH
     )
print(f"Model saved to {MODEL_SAVE_PATH}")