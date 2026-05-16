import torch
from torch.utils.data import Dataset

class AbstractData(Dataset):
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        input_id, target_id = self.examples[index] # Go into the database, find paper number index, and grab its pre-tokenized inputs and targets
        return {"input_ids": torch.tensor(input_id,dtype=torch.long), "target_id": torch.tensor(target_id,dtype=torch.long)} # Convert raw Python lists of numbers into mathematical matrices (Tensors) and wrap them in a labeled dictionary.