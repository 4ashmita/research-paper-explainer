import re
from collections import Counter

class Tokenizer:
    """
    This is the translator between human language and the numbers the Transformer model understands
    
    Attributes
    ----------------
    vocab (default 500): int
        the maximumnnumber of unique words allowed

    Methods
    -----------------
    def tokenize(self, text)
        This splits the text
    text
        The text that will be split
    
    def train(self, text)
        Builds the dictionary
    text
        the text that is used to base the dictionary off of
    
    def encode(self, text)
        Translates from word to number
    text
        The text that will be encoded
    
    def decode(self, token_id)
        Translates from number to word
    token_id
        The number that is to be decoded
    """
    def __init__(self, vocab=500):
        self.vocab_size = vocab
        # Define "Special" tokens that handle technical tasks
        # PAD: fills space to make all sequences the same length
        # UNK: Placeholder for words the model never saw during training
        # BOS: Marks the beginning of a sequence
        # EOS: Marks the end of a sequence
        self.token_to_id = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2,"[EOS]": 3}
        self.id_to_token = {i: j for j, i in self.token_to_id.items()} # Creates a reverse deictionary for decoding (number to words)
    
    def tokenize(self, text):
        # Uses regex to find all words or punctuation
        return re.findall(r"\w+|[^\w\s]", text) # Prevents Hello! from being treated as one word and becomes instead "Hello" and "!"
    
    def train(self, text):
        count = Counter() # Blank frequency counter
        for t in text: # Loop through the text
            token = self.tokenize(t) # Break the text into individual words
            count.update(token) # Add word counts to master list

        common = count.most_common(self.vocab_size-4) # Counts the most frequent words, leaving space for the 4 special tokens
        
        # Assign each word a unique id starting from 4
        for i, (token, _) in enumerate(common, start=4):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

    def encode(self, text):
        tokens = self.tokenize(text) # Split the text into words

        # Look up the ID for each word and if it isn't there replace with ID for UNK
        return [self.token_to_id.get(t, self.token_to_id["[UNK]"])for t in tokens]
    
    def decode(self, token_id):
        tokens = [self.id_to_token.get(i,"[UNK]") for i in token_id] # Turns list of tokens into words
        return " ".join(tokens)