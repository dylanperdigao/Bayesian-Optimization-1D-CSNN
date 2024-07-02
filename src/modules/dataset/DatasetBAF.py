from torch.utils.data import Dataset 
  
class DatasetBAF(Dataset): 
    def __init__(self, x, y):
        self.data = x
        self.targets = y
  
    def __len__(self): 
        return len(self.data) 
  
    def __getitem__(self, index): 
        return self.data[index], self.targets[index]