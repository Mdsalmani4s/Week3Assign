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

# =============================================================================
# STEP 5: Train the CNN Model
# COMMIT MESSAGE: "Trained CNN model for urban scene classification"
# =============================================================================

criterion  = nn.CrossEntropyLoss()
optimizer  = optim.Adam(model.parameters(), lr=0.001)
NUM_EPOCHS = 5

history = {"train_loss": [], "val_loss": [], "val_acc": []}

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=5):
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            out  = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                out  = model(images)
                val_loss += criterion(out, labels).item()
                _, pred = torch.max(out, 1)
                total   += labels.size(0)
                correct += (pred == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc      = correct / total
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

train_model(model, train_loader, val_loader, optimizer, criterion, epochs=NUM_EPOCHS)

# Generate confusion matrix on validation set
from sklearn.metrics import confusion_matrix
import numpy as np

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        out = model(images)
        _, pred = torch.max(out, 1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
actual_classes = full_dataset.classes  # Use the actual loaded classes

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(actual_classes)))
ax.set_yticks(range(len(actual_classes)))
ax.set_xticklabels(actual_classes, rotation=45, ha="right")
ax.set_yticklabels(actual_classes)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix (Validation Set)")

# Add text annotations
for i in range(len(actual_classes)):
    for j in range(len(actual_classes)):
        text = ax.text(j, i, cm[i, j], ha="center", va="center", 
                      color="white" if cm[i, j] > cm.max()/2 else "black")

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("Confusion matrix saved as confusion_matrix.png")


# =============================================================================
# STEP 6: Evaluate Model Performance
# COMMIT MESSAGE: "Evaluated CNN model performance on test data"
# =============================================================================

def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            out      = model(images)
            _, pred  = torch.max(out, 1)
            total   += labels.size(0)
            correct += (pred == labels).sum().item()
    return correct / total

test_accuracy = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Calculate per-class accuracy on test set
actual_classes = full_dataset.classes  # Use the actual loaded classes
class_correct = {cat: 0 for cat in actual_classes}
class_total = {cat: 0 for cat in actual_classes}

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        out = model(images)
        _, pred = torch.max(out, 1)
        for label, prediction in zip(labels, pred):
            cat_name = actual_classes[label]
            class_total[cat_name] += 1
            if label == prediction:
                class_correct[cat_name] += 1

class_accuracies = {cat: class_correct[cat] / class_total[cat] 
                    for cat in actual_classes if class_total[cat] > 0}

# Plot per-class accuracy
fig, ax = plt.subplots(figsize=(10, 5))
cats = list(class_accuracies.keys())
accs = list(class_accuracies.values())
colors = plt.cm.viridis(np.linspace(0, 1, len(cats)))

bars = ax.bar(cats, accs, color=colors)
ax.set_ylabel("Accuracy")
ax.set_xlabel("Urban Scene Category")
ax.set_title(f"Per-Class Test Accuracy (Overall: {test_accuracy:.2%})")
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha="right")

# Add value labels on bars
for bar, acc in zip(bars, accs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{acc:.2%}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("class_accuracy_comparison.png")
plt.close()
print("Per-class accuracy plot saved as class_accuracy_comparison.png")

# Save trained model weights
torch.save(model.state_dict(), "urban_scene_cnn_model.pth")
print("Model saved as urban_scene_cnn_model.pth")