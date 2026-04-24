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
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super.__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.feedforward = nn.Sequential(nn.Liner(embed_dim, 4*embed_dim), nn.ReLU(), nn.Linear(4*embed_dim, embed_dim), nn.Dropout(dropout))

    def forward(self, x, causal_mask):
        normalized_x = self.ln1(x)
        attention_output, _ = self.attention(normalized_x, normalized_x, normalized_x, attn_mask=causal_mask)

        x += attention_output

        normalized_x = self.ln2(x)
        feedforward_output = self.feedforward(normalized_x)
        return x
    
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, max_len=128, embed_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()

        self.max_length = max_len
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

        self.blocks = nn.ModuleList([GPTBlock(embed_dim,num_heads,dropout) for _ in range(num_layers)])

        self.ln_final = nn.LayerNorm(embed_dim)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_id):
        batch_size, sequence_length = input_id.shape

        positions = torch.arange(sequence_length, device=input_id.device)
        positions = positions.unsqueeze(0).expand(batch_size, sequence_length)

        token_embeds = self.token_embedding(input_id)
        position_embeds = self.position_embedding(positions)

        x = token_embeds + position_embeds

        causal_mask = torch.triu(torch.ones(sequence_length, sequence_length, device=input_id.device), diagonal=1).bool()
        
        for block in self.blocks:
            x = block(x, causal_mask)
        
        x = self.ln_final(x)

        logits = self.output_layer(x)
        return logits
