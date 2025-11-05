import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
import zipfile
from PIL import Image

'''
Reproducing a Mini DenseNet for classifying 4 classes from
the ImageNet dataset. We are classifying cats, bikes,
bananas, and elephants.
'''

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=4 * growth_rate,
            kernel_size=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(
            in_channels=4 * growth_rate,
            out_channels=growth_rate,
            kernel_size=3,
            padding=1,
            bias=False
        )

    def forward(self, x):
        out = self.conv1(self.bn1(x))
        out = self.conv2(self.bn2(out))
        return torch.cat([x, out], dim=1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(
                in_channels + i * growth_rate,
                growth_rate
            ))
        self.block = nn.Sequential(*layers) # Unpack list to Sequential

    def forward(self, x):
        return self.block(x)
    
class TransitionLayer(nn.Module):
    '''
    The transition layer reduces the number of 
    channels and dimensions
    '''
    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels // 2
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size = 1,
                bias = False
            ),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)

class DenseNet(nn.Module):
    def __init__(self, num_classes, growth_rate):
        super().__init__()
        num_channels = 64

        self.initial_conv = nn.Conv2d(
            in_channels=3,
            out_channels=num_channels,
            kernel_size=7,
            stride=2,
            padding=3
        )

        # Dense Block 1
        self.db1 = DenseBlock(
            num_layers=4,
            in_channels=num_channels,
            growth_rate=growth_rate
        )
        num_channels += 4 * growth_rate # Update channel count
        self.trans1 = TransitionLayer(num_channels)
        num_channels = num_channels // 2

        # Dense Block 2
        self.db2 = DenseBlock(
            num_layers=4,
            in_channels=num_channels,
            growth_rate=growth_rate
        )
        num_channels += 4 * growth_rate
        self.trans2 = TransitionLayer(num_channels)
        num_channels = num_channels // 2

        # Classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.trans1(self.db1(x))
        x = self.trans2(self.db2(x))
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
def extract_zip(zip_path, extract_folder="h2-data"):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    # Go through inner folder
    root = extract_folder
    contents = os.listdir(root)
    if len(contents) == 1 and os.path.isdir(os.path.join(root, contents[0])):
        root = os.path.join(root, contents[0])

    print(f"Dataset root: {root}")
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
        print(f"Classes found: {self.class_dirs}")
        self.class_to_index = {
            cls: i for i, cls in enumerate(self.class_dirs)
        }

        self.samples = []
        for filename in self.files:
            class_name = filename.split("_"[0]) # Ex: n02124075
            label = self.class_to_index[class_name]
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

            print(f"Epoch [{epoch}/{epochs}] Loss: {total_loss / len(train_loader):.4f}")

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    zip_path = "h2-data.zip"
    dataset_dir = extract_zip(zip_path, extract_folder="h2-data")

    class_mappings = {
        "n02124075": 0,
        "n07753592": 1,
        "n02504458": 2,
        "n03792782": 3
    }

    train_loader, test_loader = create_dataloaders(
        root_dir=dataset_dir,
        batch_size=32,
        image_size=128
    )

    # model = DenseNet(num_classes=4, growth_rate=32)

    # model = train(model, train_loader, epochs=25, lr=0.001)
    # test(model, test_loader)
