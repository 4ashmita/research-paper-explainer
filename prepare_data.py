from tokenizer import Tokenizer

with open('data/clean_data.txt', 'r', encoding="utf-8") as fh:
    text = [line.strip() for line in fh if line.strip()]

tokenizer = Tokenizer(vocab=5000)
tokenizer.train(text)

PAD_ID = tokenizer.token_to_id['[PAD]']
BOS_ID = tokenizer.token_to_id['[BOS]']
EOS_ID = tokenizer.token_to_id['[EOS]']

MAX_LENGTH = 128

def sequence(text, tokenizer, length):
    token_id = tokenizer.encode(text)
    token_id = [BOS_ID] + token_id + [EOS_ID]
    input_id = token_id[:-1]
    target_id = token_id[1:]

    if len(input_id) > length:
        input_id = token_id[:length]
        target_id = target_id[:length]
    else:
        pad_length = length - len(input_id)
        input_id += [PAD_ID] * pad_length
        target_id += [PAD_ID] * pad_length
    
    return input_id, target_id

dataset = []
for t in text:
    input_ids, target_ids = sequence(t, tokenizer, MAX_LENGTH)
    dataset.append((input_ids, target_ids))

print("Number of examples:", len(dataset))
print("First input:", dataset[0][0])
print("First target:", dataset[0][1])
print("Input length:", len(dataset[0][0]))
print("Target length:", len(dataset[0][1]))