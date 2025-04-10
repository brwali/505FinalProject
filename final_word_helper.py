import os
import glob
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torchvision import transforms, models

###############################################################################
#                            Dataset Definition                               #
###############################################################################

class WordsDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.files = sorted(glob.glob(os.path.join(root, '*.tif')))

        # Extract labels from filename: part after the last '-' and before the extension
        self.labels = [
            os.path.splitext(os.path.basename(f).split('-')[-1])[0] or 'Unknown'
            for f in self.files
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

###############################################################################
#                         Data Loaders and Samplers                           #
###############################################################################

def create_loaders(batch_size=32):
    """
    Create train/valid/test split with WeightedRandomSampler to handle 
    class imbalance, plus heavy data augmentation for training.
    Adjust transforms as needed for your dataset characteristics.
    """
    # Data augmentation transforms for training
    train_transform = transforms.Compose([
        # Random scale & crop to 224x224
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        # Slight rotation
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        # Normalization for ImageNet pretrained models
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    # For validation/testing, just resize and crop to the same shape
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    dataset_root = os.path.normpath(os.path.join(
        'cvl-database-1-1', 'preprocessed_dataset', 'words_scaled'
    ))
    
    # Create two Dataset objects: one for training, one for validation, one for test
    full_dataset = WordsDataset(root=dataset_root, transform=None)
    
    # Identify all unique classes and create label dictionaries
    classes = sorted(list(set(full_dataset.labels)))
    label_dict = {i: c for i, c in enumerate(classes)}
    inv_label_dict = {v: k for k, v in label_dict.items()}

    # Split indices: 70% train, 15% val, 15% test
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    train_size = int(0.7 * dataset_size)
    valid_size = int(0.15 * dataset_size)

    train_indices = indices[:train_size]
    valid_indices = indices[train_size: train_size + valid_size]
    test_indices  = indices[train_size + valid_size:]

    # Separate Datasets for transforms
    # We'll apply the appropriate transform in __getitem__ by creating separate objects.
    train_dataset = WordsDataset(root=dataset_root, transform=train_transform)
    valid_dataset = WordsDataset(root=dataset_root, transform=eval_transform)
    test_dataset  = WordsDataset(root=dataset_root, transform=eval_transform)

    # WeightedRandomSampler for train
    label_indices = [inv_label_dict[label] for label in full_dataset.labels]
    
    # Compute class counts
    class_counts = np.bincount(label_indices)
    # Inverse frequency
    class_weights = 1.0 / (class_counts + 1e-6)
    
    # We only want to weight the training subset
    train_sample_weights = [class_weights[idx] for idx in label_indices]
    train_subset_weights = [train_sample_weights[i] for i in train_indices]
    
    train_sampler = WeightedRandomSampler(
        weights=train_subset_weights,
        num_samples=len(train_indices),
        replacement=True
    )
    valid_sampler = SubsetRandomSampler(valid_indices)
    test_sampler  = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=train_sampler, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              sampler=valid_sampler, num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              sampler=test_sampler,  num_workers=4)
    
    print("Data Loaders created!")
    return classes, train_loader, valid_loader, test_loader, label_dict, class_weights

###############################################################################
#                            Transfer Learning Model                          #
###############################################################################

def create_resnet50_model(num_classes=1955, pretrained=True):
    """
    Creates a ResNet-50 model, replacing the final layer with a new linear 
    layer for the specified number of classes.
    """
    model = models.resnet50(pretrained=pretrained)
    # Replace the final FC layer
    # The original final layer is model.fc for ResNet
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

###############################################################################
#                          Training & Evaluation                              #
###############################################################################

def train_model(model, train_loader, valid_loader, label_dict,
                epochs=12, lr=1e-3, device=None, class_weights=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare class weights if provided
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Using a OneCycleLR scheduler (requires steps_per_epoch)
    # For smaller datasets, you might reduce the max_lr
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs
    )

    inv_label_dict = {v: k for k, v in label_dict.items()}

    best_val_acc = 0.0
    best_state_dict = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels_idx = torch.tensor([inv_label_dict[l] for l in labels], dtype=torch.long, device=device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels_idx)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels_idx).sum().item()
            total_samples += imgs.size(0)

        epoch_loss = total_loss / total_samples
        epoch_acc = 100.0 * correct / total_samples
        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

        # Evaluate on validation
        val_acc = evaluate_accuracy(model, valid_loader, label_dict, device=device)
        print(f"Validation Acc: {val_acc:.2f}%")

        # Save the best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict()

    # Load the best model state
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Loaded best model with Validation Acc = {best_val_acc:.2f}%")

    return model

def evaluate_accuracy(model, loader, label_dict, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    inv_label_dict = {v: k for k, v in label_dict.items()}

    correct = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels_idx = torch.tensor([inv_label_dict[l] for l in labels],
                                      dtype=torch.long, device=device)

            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels_idx).sum().item()
            total_samples += imgs.size(0)

    acc = 100.0 * correct / total_samples if total_samples > 0 else 0.0
    return acc

def eval_net_full_metrics(model, loader, label_dict, device=None):
    """
    This is a more detailed evaluation, computing macro Precision, Recall, F1.
    If you just want accuracy, use `evaluate_accuracy`.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)

    inv_label_dict = {v: k for k, v in label_dict.items()}
    num_classes = len(label_dict)

    correct, total = 0, 0
    true_positive = [0] * num_classes
    false_positive = [0] * num_classes
    false_negative = [0] * num_classes

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels_idx = torch.tensor([inv_label_dict[l] for l in labels],
                                      dtype=torch.long, device=device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            for i in range(len(labels_idx)):
                total += 1
                pred_label = preds[i].item()
                true_label = labels_idx[i].item()

                if pred_label == true_label:
                    correct += 1
                    true_positive[true_label] += 1
                else:
                    false_positive[pred_label] += 1
                    false_negative[true_label] += 1

    accuracy = 100.0 * correct / total if total > 0 else 0.0

    # Compute macro-level precision, recall, f1
    precision_list, recall_list, f1_list = [], [], []
    for i in range(num_classes):
        tp = true_positive[i]
        fp = false_positive[i]
        fn = false_negative[i]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    macro_precision = sum(precision_list) / num_classes
    macro_recall = sum(recall_list) / num_classes
    macro_f1 = sum(f1_list) / num_classes

    print(f"Accuracy:         {accuracy:.2f}%")
    print(f"Macro Precision:  {macro_precision:.4f}")
    print(f"Macro Recall:     {macro_recall:.4f}")
    print(f"Macro F1-Score:   {macro_f1:.4f}")
