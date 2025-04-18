import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from sklearn.metrics import classification_report


class WordsDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.files = sorted(glob.glob(os.path.join(root, '*.tif')))
        self.labels = [os.path.splitext(os.path.basename(f).split('-')[-1])[0] or 'Unknown' for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')  # ensure 3 channels
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

    dataset_root = os.path.normpath(os.path.join('cvl-database-1-1', 'preprocessed_dataset', 'words_scaled'))
    dataset = WordsDataset(root=dataset_root, transform=transform)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    train_size = int(0.7 * dataset_size)
    valid_size = int(0.15 * dataset_size)

    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size + valid_size]
    test_indices = indices[train_size + valid_size:]

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(valid_indices))
    test_loader  = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices))

    classes = sorted(list(set(dataset.labels)))
    label_dict = {i: c for i, c in enumerate(classes)}

    print('Done creating loaders')
    return classes, train_loader, valid_loader, test_loader, label_dict


class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512), nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def trainNet(model, train_loader, valid_loader, label_dict, epochs=10, lr=1e-3, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    inv_label = {v: k for k, v in label_dict.items()}

    for epoch in range(epochs):
        model.train()
        loss_sum = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels_idx = torch.tensor([inv_label[l] for l in labels], dtype=torch.long, device=device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels_idx)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print(f"Epoch {epoch+1}/{epochs} Train Loss {loss_sum/len(train_loader):.4f}")
        evalNet(model, valid_loader, label_dict, device)


def evalNet(model, loader, label_dict, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    correct = total = 0
    inv_label = {v: k for k, v in label_dict.items()}
    
    num_classes = len(label_dict)
    true_positive = [0] * num_classes
    false_positive = [0] * num_classes
    false_negative = [0] * num_classes
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels_idx = torch.tensor([inv_label[l] for l in labels if l in inv_label], dtype=torch.long, device=device)
            
            preds = model(imgs).argmax(dim=1)
            
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
    
    # Print accuracy
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Calculate precision, recall, and F1-score
    precision_list = []
    recall_list = []
    f1_list = []
    
    for i in range(num_classes):
        tp = true_positive[i]
        fp = false_positive[i]
        fn = false_negative[i]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    # Macro average metrics
    macro_precision = sum(precision_list) / num_classes
    macro_recall = sum(recall_list) / num_classes
    macro_f1 = sum(f1_list) / num_classes
    
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")


class CursiveGenerator(nn.Module):
    def __init__(self, num_classes, embed_dim=256, img_shape=(3, 775, 120)):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)
        self.fc = nn.Linear(embed_dim, int(np.prod(img_shape)))
        self.img_shape = img_shape

    def forward(self, labels):
        x = self.embed(labels)
        x = self.fc(x)
        return x.view(-1, *self.img_shape)


def trainCursiveNet(model, loader, label_dict, epochs=10, lr=1e-3, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    inv_label = {v: k for k, v in label_dict.items()}

    for epoch in range(epochs):
        model.train()
        loss_sum = 0
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels_idx = torch.tensor([inv_label[l] for l in labels], dtype=torch.long, device=device)
            optimizer.zero_grad()
            loss = criterion(model(labels_idx), imgs)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print(f"Epoch {epoch+1}/{epochs} Gen Loss {loss_sum/len(loader):.4f}")


def evalCursiveNet(model, loader, label_dict, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    inv_label = {v: k for k, v in label_dict.items()}
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels_idx = torch.tensor([inv_label[l] for l in labels], dtype=torch.long, device=device)
            total_loss += criterion(model(labels_idx), imgs).item()
    print(f"Validation Gen Loss {total_loss/len(loader):.4f}")
