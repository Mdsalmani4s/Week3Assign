# =============================================================================
# STEP 3: Load and Prepare the Dataset (MIT Places Subset)
# COMMIT MESSAGE: "Loaded and preprocessed MIT Places dataset"
# =============================================================================

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import kagglehub
import os
import shutil

# 5 urban categories for our subset
CATEGORIES = ["bridge", "downtown", "highway", "parking_lot", "skyscraper"]
SUBSET_DIR  = "./urban_subset"

def subset_is_ready(path, categories):
    """Return True only if every category folder exists and contains at least one image."""
    if not os.path.isdir(path):
        return False
    for cat in categories:
        cat_path = os.path.join(path, cat)
        if not os.path.isdir(cat_path):
            return False
        imgs = [f for f in os.listdir(cat_path) if f.lower().endswith(".jpg")]
        if len(imgs) == 0:
            return False
    return True

if subset_is_ready(SUBSET_DIR, CATEGORIES):
    # Already extracted on this machine – skip download & copy entirely
    print(f"Subset already exists at {SUBSET_DIR}, skipping download.")
else:
    # Download (or grab from kagglehub cache if already on disk)
    print("Downloading MIT Places dataset via KaggleHub...")
    data_root = kagglehub.dataset_download("mittalshubham/images256")
    print(f"Dataset location: {data_root}")

    # Build subset folder by copying images for each category
    os.makedirs(SUBSET_DIR, exist_ok=True)
    for cat in CATEGORIES:
        src = os.path.join(data_root, cat[0], cat)          # e.g. .../b/bridge
        dst = os.path.join(SUBSET_DIR, cat)
        if os.path.exists(src):
            os.makedirs(dst, exist_ok=True)
            imgs = [f for f in os.listdir(src) if f.lower().endswith(".jpg")][:400]
            for img in imgs:
                shutil.copy2(os.path.join(src, img), os.path.join(dst, img))
            print(f"  {cat}: {len(imgs)} images copied")
        else:
            print(f"  WARNING – category folder not found: {src}")

# Transforms: resize → tensor → normalize (ImageNet stats)
preprocessing = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# Load full dataset
full_dataset = ImageFolder(root=SUBSET_DIR, transform=preprocessing)
print(f"\nDataset loaded – {len(full_dataset)} images, classes: {full_dataset.classes}")

# 70 / 15 / 15 split
n_train = int(0.70 * len(full_dataset))
n_val   = int(0.15 * len(full_dataset))
n_test  = len(full_dataset) - n_train - n_val

train_set, val_set, test_set = random_split(full_dataset, [n_train, n_val, n_test])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False)

print(f"Split – Train: {n_train}, Val: {n_val}, Test: {n_test}")

# Show one sample image
img, lbl = full_dataset[0]
# Undo normalization for display
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
display_img = torch.clamp(img * std + mean, 0, 1)

plt.figure(figsize=(4, 4))
plt.imshow(display_img.permute(1, 2, 0).numpy())
plt.title(f"Sample – {full_dataset.classes[lbl]}")
plt.axis("off")
plt.savefig("sample_image.png")
plt.close()
print("Sample image saved as sample_image.png")



# =============================================================================
# STEP 4: Build a Simple CNN Model
# COMMIT MESSAGE: "Implemented CNN model for urban scene classification"
# =============================================================================

import torch.nn as nn
import torch.optim as optim

class UrbanSceneCNN(nn.Module):
    """
    CNN with two conv blocks (each: Conv → BN → ReLU → MaxPool)
    followed by a Dropout layer and one fully-connected classifier.
    """
    def __init__(self, n_classes):
        super().__init__()
        # Block 1: 3 → 32 channels, 128×128 → 64×64
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Block 2: 32 → 64 channels, 64×64 → 32×32
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.dropout = nn.Dropout(p=0.5)
        # 64 channels × 32 × 32 spatial
        self.classifier = nn.Linear(64 * 32 * 32, n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

n_classes = len(full_dataset.classes)
model = UrbanSceneCNN(n_classes)
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")