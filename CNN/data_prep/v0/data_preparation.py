import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.nn.functional import one_hot
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

import os


class CustomCNNDataset(Dataset):

    '''
    For more info README.md
    '''
    

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.image_paths = []
        self.labels = []

        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                class_images = os.listdir(class_dir)
                self.image_paths.extend([os.path.join(class_dir, img) for img in class_images])
                self.labels.extend([i] * len(class_images))
    
    def __len__(self):

        return len(self.labels)
    
    def __getitem__(self, index):

        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        one_hot_label = one_hot(torch.tensor(label), len(self.classes)).float()


        return image, one_hot_label
    
    def splitter(self, splits=[85,15]):
        index = torch.arange(len(self.image_paths))
        splits_tensor = torch.as_tensor(splits)
  
        multiplier = len(self.image_paths) / splits_tensor.sum()   
        splits_tensor = (multiplier * splits_tensor).long()
        
        # the split_tensor obtained could have a different number of index
        # to avoid it add a possible difference to the first split (train_split)
        diff = len(self.image_paths) - splits_tensor.sum()
        
        splits_tensor[0] += diff
        print(index, splits_tensor)
        
        # get the index for the Subset
        return random_split(index, splits_tensor) 


if __name__ == '_name__':

    # EXAMPLE
    PATH = 'path\to\dataset'

    transform = Compose([
                    Resize((224, 224)),  
                    ToTensor(),
                    Normalize(mean=(.5,), std=(.5,))
                ])
    
    dataset = CustomCNNDataset(root_dir=rf'{PATH}', transform=transform)

    train_idx, val_idx = dataset.splitter(splits=[85, 15])

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
