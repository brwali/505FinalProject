import os
import glob
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms


# Custom dataset that reads images from the preprocessed "words" folder
# and extracts the label from the filename.
class WordsDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # List all .tif images in the directory
        self.files = glob.glob(os.path.join(root, '*.tif'))
        self.labels = []
        for f in self.files:
            base = os.path.basename(f)
            # Split on '-' and take the last element (remove extension) as the label.
            parts = base.split('-')
            label = os.path.splitext(parts[-1])[0]
            # If label is empty or equals ".tif", set it to a default value.
            if not label or label.lower() == '.tif':
                label = "Unknown"
            self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


def createLoaders(batch_size=128):
    # Define transform: convert image to tensor and normalize.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create the dataset from the preprocessed "words" folder
    dataset_root = os.path.normpath(os.path.join('cvl-database-1-1', 'preprocessed_dataset', 'words_padded'))
    dataset = WordsDataset(root=dataset_root, transform=transform)

    # Split indices: 70% training, 15% validation, 15% testing.
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    train_size = int(0.7 * dataset_size)
    valid_size = int(0.15 * dataset_size)
    # test_size = dataset_size - train_size - valid_size

    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size + valid_size]
    test_indices = indices[train_size + valid_size:]

    # Create samplers for each split.
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create DataLoaders.
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    # Determine the list of unique classes (labels) from the dataset.
    classes = sorted(list(set(dataset.labels)))

    # Create a dictionary mapping index to class label.
    label_dict = {i: c for i, c in enumerate(classes)}

    print('Done creating loaders')
    return classes, train_loader, valid_loader, test_loader, label_dict


# Example usage:
classes, train_loader, valid_loader, test_loader, label_dict = createLoaders(batch_size=128)
