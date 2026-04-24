import torch
from torch.utils.data import Dataset

class AbstractData(Dataset):
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        input_id, target_id = self.examples[index]
        return {"input_ids": torch.tensor(input_id,dtype=torch.long), "target_id": torch.tensor(target_id,dtype=torch.long)}