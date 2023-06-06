import torch
from torchvision.transforms import transforms
from imblearn.over_sampling import SMOTE
from torch.utils.data.sampler import SubsetRandomSampler



class ImageFolderSubset(torch.utils.data.Subset):
    '''A ImageFolder subset class'''
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.targets = dataset.targets # 保留targets属性
        self.classes = dataset.classes # 保留classes属性
        self.indices = indices

    def __getitem__(self, idx): #同时支持索引访问操作
        x, y = self.dataset[self.indices[idx]]      
        return x, y 

    def __len__(self): # 同时支持取长度操作
        return len(self.indices)

# for moving data to device (CPU or GPU)
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# for loading in the device (GPU if available else CPU)
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)
        
    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
def load_dataset(dataset, n_train, n_valid, batch_size, num_workers, device, num_classes):

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [n_train, n_valid])

    # sampled_indices = [260, 1500, 800, 288, 292, 1100, 829, 238, 362]
    
    # 用 SubsetRandomSampler 根据索引列表对train_dataset进行采样
    # train_sampler = SubsetRandomSampler(sampled_indices)
    
    print("length of train_dataset: " + str(len(train_dataset)))
    
    # 定义dataloader
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    valid_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    print("len of train_dl: " + str(len(train_dl)))
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)
    return train_dl, valid_dl

    
    
