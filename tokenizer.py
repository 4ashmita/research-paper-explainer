import re
from collections import Counter

class tokenizer:
    def __init__(self, vocab=500):
        self.vocab_size = vocab
        self.token_to_id = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2,"[EOS]": 3}
        self.id_to_token = {i: j for j, i in self.token_to_id.items()}
    
    def tokenize(self, text):
        return re.findall(r"\w+|[^\w\s]", text)
    
    def train(self, text):
        count = Counter()
        for t in text:
            token = self.tokenize(t)
            count.update(token)

        common = count.most_common(self.vocab_size-4)
        for i, (token, _) in enumerate(common, start=4):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.token_to_id.get(t, self.token_to_id["[UNK]"])for t in tokens]
    
    def decode(self, token_id):
        tokens = [self.id_to_token.get(i,"[UNK]") for i in token_id]
        return " ".join(tokens)
    
