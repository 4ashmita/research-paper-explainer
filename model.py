import torch
import torch.nn as nn
import torch.nn.functional as f

class GPTBlock(nn.Module):
    """
    One Transformer block -> a proper GPT model is made by stacking many of these blocks

    Attributes
    ------------
    embed_dim: int
        the size of each token vector
    num_heads: int
        The number of attention heads or attention mechanisms
    dropout: float (default 0.1)
        How many values to randomly turn off (default 10%) while training to avoid overfitting
    
    Methods
    -----------
    def forward(self, x, causal_mask)
        defines what happens when the data passes through on GPT block
    x
        the token representations
    causal_mask
        prevents the model from looking at future tokens -> prevents cheating
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim) #first layer normalization -> stabilizes training by normalizing values in each token vector 
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True) #Creates self-attention layer
        self.ln2 = nn.LayerNorm(embed_dim) # created another LayerNorm which is applied before the feedforward network

        # feedforward network. Creates a small neuron network that process each token independently after the attention layers
        self.feedforward = nn.Sequential(nn.Linear(embed_dim, 4*embed_dim), nn.ReLU(), nn.Linear(4*embed_dim, embed_dim), nn.Dropout(dropout))

    def forward(self, x, causal_mask):
        normalized_x = self.ln1(x) #Applies first LayerNorm to x which doesn't change its shape
        
        #runs self-attention. _ -> stores attention weights but is being ignored 
        attention_output, _ = self.attention(normalized_x, normalized_x, normalized_x, attn_mask=causal_mask)

        x = x + attention_output

        # Applies second LayerNorm before feedforward
        normalized_x = self.ln2(x)
        feedforward_output = self.feedforward(normalized_x)
        
        x = x + feedforward_output
        return x
    
class MiniGPT(nn.Module):
    """
    Defines full GPT model

    Attributes
    ------------
    vocab_size: int
        number of tokens in vocabulary
    max_len: int (default 128)
        maximum sequence length
    embed_dim: int (default 128)
        the size of each token vector
    num_heads: int (default 4)
        The number of attention heads or attention mechanisms in each Transformer block
    num_layers: int (default 2)
        Number of GPT blocks stacjs together
    dropout: float (default 0.1)
        How many values to randomly turn off (default 10%) while training to avoid overfitting
    
    Methods
    -----------
    def forward(self, input_ids)
        defines how input flows through the model
    input_id: list
        List of all the token_IDs
    """
    def __init__(self, vocab_size, max_len=128, embed_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__() #Initializes parent nn.Module

        self.max_length = max_len
        self.token_embedding = nn.Embedding(vocab_size, embed_dim) #Creates a lookup table converting token IDs into vectors
        self.position_embedding = nn.Embedding(max_len, embed_dim) #Creates lookup table for position as Self-attention alone doesn't know order

        #Creates a list of GPT blocks
        self.blocks = nn.ModuleList([GPTBlock(embed_dim,num_heads,dropout) for _ in range(num_layers)])

        self.ln_final = nn.LayerNorm(embed_dim) #Adds one final normalization layer after all the Transformer blocks
        self.output_layer = nn.Linear(embed_dim, vocab_size) #Converts each final token vectors into scores over the vocab

    def forward(self, input_id):
        batch_size, sequence_length = input_id.shape

        positions = torch.arange(sequence_length, device=input_id.device) #Creates position numbers
        positions = positions.unsqueeze(0).expand(batch_size, sequence_length) #Reshapes and repeats the position for every example in the batch

        token_embeds = self.token_embedding(input_id) #Turns token IDs into token vectors
        position_embeds = self.position_embedding(positions) #Turns position IDs into position vectors

        x = token_embeds + position_embeds #Now each token vector contains what token it is and where it appears in the sequence
        
        #Creates the GPT mask so as to not let the model look at future tokens during training
        causal_mask = torch.triu(torch.ones(sequence_length, sequence_length, device=input_id.device), diagonal=1).bool()
        
        #passes x through each GPT block
        for block in self.blocks:
            x = block(x, causal_mask)
        
        x = self.ln_final(x) #Final LayerNorm after all the Transformer blocks

        logits = self.output_layer(x) #Projects final vectors into vocabulary scores
        return logits
