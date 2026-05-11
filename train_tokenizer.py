from tokenizer import Tokenizer
import pickle

# Load your abstracts
# ... (your loading code) ...
DATA = "data/clean_data.txt"
with open(DATA, "r") as fh:
    text = [line.strip() for line in fh if line.strip()]

tokenizer = Tokenizer(vocab=40000)
# Use a slice to keep it fast!
tokenizer.train(text) 

# Save it to a file
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Tokenizer saved! 'Osaka' id is:", tokenizer.encode("Osaka"))