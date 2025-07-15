import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from detection.deepfake_model import DeepfakeCNN
from torch import nn, optim
import os

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Check if dataset folders exist
if not os.path.exists('dataset/real') or not os.path.exists('dataset/fake'):
    raise Exception("🚨 dataset/real/ and dataset/fake/ folders are missing. Please create them and add images before training.")

# Dataset and DataLoader
train_data = datasets.ImageFolder('dataset/', transform=transform)
print(f"📌 Detected class mapping: {train_data.class_to_idx}")

train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

# Model, loss, optimizer
model = DeepfakeCNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        # ✅ Remap labels: 0 → 1 (fake), 1 → 0 (real)
        labels = 1 - labels
        labels = labels.unsqueeze(1).float()  # Shape [batch_size, 1]
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

# Save model weights only (state_dict)
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/deepfake_model.pth')
print("✅ Model weights saved to 'models/deepfake_model.pth'")
