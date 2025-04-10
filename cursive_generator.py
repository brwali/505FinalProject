import os
import glob
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

##############################################################################
#                         1) Text + Image Dataset                            #
##############################################################################

class TextImageDataset(Dataset):
    """
    A minimal dataset for (word, cursive_image) pairs:
      - Each image is assumed to correspond to a 'word' label (string).
      - Filenames might have some consistent pattern or a CSV file listing pairs.
    """
    def __init__(self, root, transform=None):
        """
        root: a directory with images named like "word_XXX.tif" or a structure
              that we can map from file to word.
        transform: transforms to apply to the images.
        """
        super().__init__()
        self.root = root
        self.transform = transform
        
        # Example: gather tif files, parse the word from the filename
        self.files = sorted(glob.glob(os.path.join(root, '*.tif')))
        
        # Suppose filenames are "hello_001.tif", "hello_002.tif", ...
        # We'll extract the word by splitting on '_' or some known delimiter.
        self.words = []
        for f in self.files:
            fname = os.path.basename(f)
            # e.g., "hello_001.tif" -> word "hello"
            word_part = fname.split('_')[0]
            self.words.append(word_part)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        word = self.words[idx]

        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return word, img

##############################################################################
#                         2) Text Embedding Module                           #
##############################################################################

class TextEmbedding(nn.Module):
    """
    Converts input word into a fixed-size embedding vector, for conditional generation.
    This is a placeholder example using PyTorch embeddings for a small vocabulary.
    """
    def __init__(self, vocab_size=10000, embed_dim=128):
        super().__init__()
        # Simple embedding layer for demonstration
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # For real usage, you’d build a vocabulary/dictionary for all words
        # or use a more advanced approach (e.g., character-level embedding).

    def forward(self, word_indices):
        """
        word_indices: a [batch_size, sequence_length] LongTensor, if we’re
                      embedding text. For single words, sequence_length=1 is possible.
        """
        # shape: (batch_size, seq_len, embed_dim)
        out = self.embedding(word_indices)
        # For single words, we might just squeeze to (batch_size, embed_dim).
        return out.squeeze(1)

##############################################################################
#            3) Generator + Discriminator (Conditional DCGAN)                #
##############################################################################

class CursiveGenerator(nn.Module):
    """
    Generator that takes a random noise vector (z) + text embedding,
    and outputs a cursive image (e.g., 1 x 64 x 128 or 3 x 64 x 128).
    For simplicity, we’ll produce a 3 x 64 x 128 image.
    """
    def __init__(self, noise_dim=100, embed_dim=128, img_channels=3, feature_maps=64):
        super().__init__()
        self.noise_dim = noise_dim
        self.embed_dim = embed_dim
        
        # Project embedding + noise into a larger hidden
        # You can feed text embedding at multiple layers too.
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + embed_dim, feature_maps * 8 * 4 * 4),  # shape assumption
            nn.ReLU(True),
        )

        # Transposed Convolutions
        self.gen = nn.Sequential(
            # Input: (feature_maps*8) x 4 x 4
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            # State size: (feature_maps*4) x 8 x 8
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            # State size: (feature_maps*2) x 16 x 16
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            # State size: (feature_maps) x 32 x 32
            nn.ConvTranspose2d(feature_maps, img_channels, 4, 2, 1),
            nn.Tanh()
            # Output: (3) x 64 x 64 (or 64 x 128, you can adapt as needed)
        )

    def forward(self, noise, text_embed):
        """
        noise: (batch_size, noise_dim)
        text_embed: (batch_size, embed_dim)
        """
        x = torch.cat([noise, text_embed], dim=1)  # (batch_size, noise_dim + embed_dim)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)  # reshape for transposed conv
        img = self.gen(x)
        return img


class CursiveDiscriminator(nn.Module):
    """
    Discriminator to classify real vs. fake images, conditioned on text embedding.
    We'll feed the image + repeated text embedding through a CNN.
    """
    def __init__(self, img_channels=3, embed_dim=128, feature_maps=64):
        super().__init__()
        
        # Project text embedding to a spatial dimension, then concat with image
        self.embed_fc = nn.Linear(embed_dim, 16 * 16)  # example shape
        self.conv = nn.Sequential(
            # input:  (img_channels+1) x 64 x 64 (if we concat embed as an extra channel or
            # embed repeated on a 2D map).
            nn.Conv2d(img_channels + 1, feature_maps, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 4, 1, 4, 1, 0),
            # Output: single scalar => real/fake
        )

    def forward(self, img, text_embed):
        """
        img: (batch_size, img_channels, 64, 64)
        text_embed: (batch_size, embed_dim)
        """
        # Project embed => shape (batch_size, 16*16)
        e = self.embed_fc(text_embed)
        # reshape => (batch_size, 1, 16, 16)
        e = e.view(e.size(0), 1, 16, 16)
        
        # we need to upsample embed to match the image shape (64x64),
        # or we can do something simpler: replicate the embed map 4 times 
        # to get (64x64). For demonstration, let's just do nearest-neighbor:
        e = nn.functional.interpolate(e, size=(64,64), mode='nearest')

        # Now we have e shape => (batch_size, 1, 64, 64)
        # Concat with image on channel dimension
        x = torch.cat([img, e], dim=1)  # (batch_size, img_channels+1, 64, 64)
        
        out = self.conv(x)
        # out shape => (batch_size, 1, 1, 1)
        return out.view(-1, 1)  # flatten to (batch_size, 1)

##############################################################################
#                              4) Training Loop                              #
##############################################################################

def trainCursiveNet(
    generator, 
    discriminator, 
    text_embedder, 
    dataloader,
    vocab,         # a dictionary mapping words -> indices
    device=None,
    epochs=10,
    noise_dim=100,
    lr=2e-4
):
    """
    Trains the cGAN on (word, image) pairs.
    For real images, train D to output 'real', for G's generated images, train D to output 'fake'.
    G tries to fool D into outputting 'real'.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)
    text_embedder.to(device)

    # Loss + Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optim_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for i, (words, real_imgs) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # Convert word -> vocab index. For simplicity, assume single-word, single-index:
            # In a real system, you'd handle unknown words, multi-character embeddings, etc.
            word_indices = [vocab.get(w, 0) for w in words]  # fallback 0 if not found
            word_indices = torch.LongTensor(word_indices).unsqueeze(1).to(device)
            
            # 1) Prepare real/fake labels
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # 2) Train Discriminator
            #    a) Real
            text_emb = text_embedder(word_indices)   # (batch_size, embed_dim)
            out_real = discriminator(real_imgs, text_emb)
            d_loss_real = criterion(out_real, real_labels)

            #    b) Fake
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_imgs = generator(noise, text_emb)
            out_fake = discriminator(fake_imgs.detach(), text_emb)
            d_loss_fake = criterion(out_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake

            optim_D.zero_grad()
            d_loss.backward()
            optim_D.step()

            # 3) Train Generator (wants D to say 'real' on fake images)
            out_fake_forG = discriminator(fake_imgs, text_emb)
            g_loss = criterion(out_fake_forG, real_labels)

            optim_G.zero_grad()
            g_loss.backward()
            optim_G.step()

            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], "
                      f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

    print("Training complete.")

##############################################################################
#                      5) Generating a cursive word image                    #
##############################################################################

def generate_cursive_image(generator, text_embedder, word, vocab,
                           noise_dim=100, device=None):
    """
    Generates a single cursive image from the generator, given a 'word'.
    Returns a PyTorch tensor image: shape (3, 64, 64).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    generator.eval()
    text_embedder.eval()

    # Convert word -> embedding
    word_idx = vocab.get(word, 0)  # fallback if unknown
    word_idx = torch.LongTensor([word_idx]).unsqueeze(1).to(device)
    text_emb = text_embedder(word_idx)

    # Sample noise
    noise = torch.randn(1, noise_dim, device=device)

    # Generate
    with torch.no_grad():
        fake_img = generator(noise, text_emb)
        # shape => (1, 3, 64, 64)
    return fake_img.squeeze(0)  # shape => (3, 64, 64)