import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO

# --- Configuration ---
DATA_NAME = 'breastmnist'
NUM_EPOCHS = 10  # Train for 10 epochs, sufficient for a good result
BATCH_SIZE = 128
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = 'breast_mnist_model.pth'

# --- 1. Load Data and Prepare DataLoaders ---
print("Step 1: Loading and preparing data...")
# Get dataset information from medmnist
info = INFO[DATA_NAME]
n_classes = len(info['label'])
n_channels = info['n_channels']
print(f"Dataset: {DATA_NAME}, Classes: {n_classes}, Channels: {n_channels}")

# Define data transformations
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5]) # Normalize for single-channel images
])

# Download and load the datasets
train_dataset = getattr(medmnist, info['python_class'])(split='train', transform=data_transform, download=True)
test_dataset = getattr(medmnist, info['python_class'])(split='test', transform=data_transform, download=True)

# Create DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("Data loaded successfully.")

# --- 2. Define the CNN Model ---
print("\nStep 2: Defining the CNN model...")
# A simple CNN for 28x28 images
class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1) # Flatten the tensor
        out = self.fc(out)
        return out

model = SimpleCNN(in_channels=n_channels, num_classes=n_classes)
print("Model defined.")
# print(model) # Uncomment to see model architecture

# --- 3. Train the Model ---
print("\nStep 3: Starting model training...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    model.train()
    for images, labels in train_loader:
        labels = labels.squeeze().long() # Ensure labels are in the correct format

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')

print("Training finished.")

# --- 4. Evaluate the Model ---
print("\nStep 4: Evaluating model on test data...")
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        labels = labels.squeeze().long()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f} %')

# --- 5. Save the Trained Model ---
print(f"\nStep 5: Saving the trained model to {MODEL_SAVE_PATH}...")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("Model saved successfully. You are ready for Step 2!")