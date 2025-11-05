import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import zipfile
from PIL import Image
import time

'''
Creating a simple CNN for classifying 4 classes from
the ImageNet dataset. We are classifying cats, bikes,
bananas, and elephants.
'''

class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        if padding is None:
            padding = (kernel_size - 1) // 2
        
        # Convolution layer
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
        # Normalizes the output of the convolution layer
        self.batch_norm = nn.BatchNorm2d(out_channels)

        self.activation = nn.ReLU() 

    def forward(self, x):
        return self.activation(self.batch_norm(self.convolution(x)))
    
class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Start with 3 input channels (RGB) -> 64 output channels
            ConvolutionBlock(in_channels=3, 
                            out_channels=64, 
                            kernel_size=7, 
                            stride=2, 
                            padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2), # Downsample by 2

            ConvolutionBlock(in_channels=64, 
                            out_channels=128,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvolutionBlock(in_channels=128,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            ConvolutionBlock(in_channels=64,
                            out_channels=32,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Fully connected layer (flatten to 1d vector)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 4),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def extract_zip(zip_path, extract_folder="h2-data"):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    # Go through inner folder
    root = extract_folder
    contents = os.listdir(root)
    if len(contents) == 1 and os.path.isdir(os.path.join(root, contents[0])):
        root = os.path.join(root, contents[0])

    return root

class ImageDataset(Dataset):
    def __init__(self, root_dir, txt_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        with open(os.path.join(root_dir, txt_file), 'r') as f:
            self.files = [line.strip() for line in f.readlines()]

        # Build class list from folder names
        self.class_dirs = sorted([
            d for d in os.listdir(root_dir) if d.startswith("n")
        ])
        self.class_mappings = {
            "n02124075": 0,
            "n07753592": 1,
            "n02504458": 2,
            "n03792782": 3
        }

        self.samples = []
        for filename in self.files:
            class_name = filename.split("_")[0] # Ex: n02124075
            label = self.class_mappings[class_name]
            full_path = os.path.join(root_dir, class_name, filename)
            self.samples.append((full_path, label))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

def create_dataloaders(root_dir, batch_size, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = ImageDataset(
        root_dir=root_dir,
        txt_file="train.txt",
        transform=transform
    )
    test_dataset = ImageDataset(
        root_dir=root_dir,
        txt_file="test.txt",
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader

def train(model, train_loader, epochs, lr):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {total_loss / len(train_loader):.4f}")

    print("Training Complete")

    return model

def test(model, test_loader):
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    return accuracy


if __name__ == "__main__":
    start_time = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    zip_path = "h2-data.zip"
    dataset_dir = extract_zip(zip_path, extract_folder="h2-data")
    train_loader, test_loader = create_dataloaders(
        root_dir=dataset_dir,
        batch_size=32,
        image_size=128
    )

    model = CustomCNN()

    model = train(model, train_loader, epochs=8, lr=0.01)
    test(model, test_loader)

    end_time = time.time()
    minutes = (end_time - start_time) / 60
    seconds = (end_time - start_time) % 60
    print(f"Total execution time: {minutes:.0f} minutes {seconds:.2f} seconds")
