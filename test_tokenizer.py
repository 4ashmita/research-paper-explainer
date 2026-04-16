from tokenizer import Tokenizer

with open("data/clean_data.txt", "r") as fh:
    text = fh.readlines()

tokenizer = Tokenizer(vocab=5000)
tokenizer.train(text)


print("=== VOCAB TEST ===")
print(tokenizer.token_to_id)

print("\n=== TOKENIZE TEST ===")
sample = "This study investigates memory."
tokens = tokenizer.tokenize(sample)
print("Input:", sample)
print("Tokens:", tokens)

print("\n=== ENCODE TEST ===")
encoded = tokenizer.encode(sample)
print("Encoded:", encoded)

print("\n=== DECODE TEST ===")
decoded = tokenizer.decode(encoded)
print("Decoded:", decoded)

print("\n=== UNKNOWN WORD TEST ===")
unknown_sample = "This Biochemical agent study works."
unknown_encoded = tokenizer.encode(unknown_sample)
unknown_decoded = tokenizer.decode(unknown_encoded)
print("Input:", unknown_sample)
print("Encoded:", unknown_encoded)
print("Decoded:", unknown_decoded)