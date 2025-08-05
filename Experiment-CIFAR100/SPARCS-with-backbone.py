import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import time
import shutil
from tqdm import tqdm

# Choose a readable font size for the images
plt.rcParams.update({'font.size': 16})

# Load the pretrained EfficientNetV2-S model
def get_efficientnet_backbone(pretrained=True):
    
    if pretrained:
        # Initialize the model with pretrained weights
        model = torchvision.models.efficientnet_v2_s(weights='DEFAULT')
    else:
        # Initialize the model without pretrained weights
        model = torchvision.models.efficientnet_v2_s(weights=None)

    # Remove the original classifier
    model.classifier = nn.Identity()
    return model

class EfficientNetWithLinearClassifier(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.backbone = get_efficientnet_backbone()
        # EfficientNetV2-S has 1280 output features
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

class EfficientNetWithSpectralClassifier(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.backbone = get_efficientnet_backbone()
        # EfficientNetV2-S has 1280 output features
        self.input_dim = 1281  # +1 for bias neuron
        self.hidden_dim = 526
        self.output_dim = num_classes

        self.varphi1 = nn.Parameter(torch.empty(self.hidden_dim, self.input_dim), requires_grad=True)
        self.varphi2 = nn.Parameter(torch.empty(self.output_dim, self.hidden_dim), requires_grad=True)

        self.l1_diag = nn.Parameter(torch.empty(self.input_dim), requires_grad=False)
        self.l2_diag = nn.Parameter(torch.empty(self.hidden_dim), requires_grad=True)
        self.l3_diag = nn.Parameter(torch.empty(self.output_dim), requires_grad=True)

        self.spectral_activ = nn.ReLU()

        nn.init.xavier_uniform_(self.varphi1)
        nn.init.xavier_uniform_(self.varphi2)

        with torch.no_grad():
            self.l1_diag.fill_(0.)
            self.l2_diag.fill_(0.)
            self.l3_diag.fill_(1.)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1) # acts like Flatten
        # bias concat
        ones = torch.ones(x.size(0), 1, device=x.device)
        x = torch.cat([x, ones], dim=1)
        # spectral weights via broadcasting
        W21 = self.varphi1 * self.l1_diag.unsqueeze(0) - self.l2_diag.unsqueeze(1) * self.varphi1
        W32 = self.varphi2 * self.l2_diag.unsqueeze(0) - self.l3_diag.unsqueeze(1) * self.varphi2
        W31 = (self.varphi2 * self.l2_diag.unsqueeze(0) - self.l3_diag.unsqueeze(1) * self.varphi2) @ self.varphi1
        # apply spectral pass
        middle_activations = self.spectral_activ(torch.matmul(W21, x.t()))
        output = torch.matmul(W31, x.t()) + torch.matmul(W32, middle_activations)
        return output.t()

# Train function
def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0
    correct = 0
    memory_allocated_list = []

    for images, labels in tqdm(loader, desc="Training", unit="batch"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Implement spectral regularization (effective only if the model is spectral)
        if isinstance(model, EfficientNetWithSpectralClassifier):
            loss += 0.0001 * model.l2_diag.abs().sum()  # L1 regularization on l2_diag
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()

        # Measure memory usage
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # Convert to MB
        memory_allocated_list.append(memory_allocated)


    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / len(loader.dataset)
    average_memory_allocated = np.mean(memory_allocated_list)

    return epoch_loss, epoch_acc, average_memory_allocated

# Validation function
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / len(loader.dataset)

    return epoch_loss, epoch_acc

def validate_spectral_pruned(model, loader, criterion):
    '''
    It validates the model in the pruned state by setting the l2_diag to zero.
    Of course without modifying the model, so that the model can be used later for training.
    '''
    with torch.no_grad():
        backup = model.l2_diag.clone()
        model.l2_diag.zero_()
        loss, acc = validate(model, loader, criterion)
        model.l2_diag.copy_(backup)

    return loss, acc

# Testing function
def test(model, loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

    avg_loss = test_loss / len(loader)
    accuracy = 100. * correct / len(loader.dataset)
    print(f"Test: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":

    # Hyperparameters
    num_epochs = 1
    batch_size = 128
    learning_rate = 0.001
    number_of_runs = 3

    # Select the second GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Get path to the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the results directory if it doesn't exist
    results_directory = os.path.join(current_dir, 'Experiment-2-Results')
    
    # Remove the content of the results directory if it exists
    if os.path.exists(results_directory):
        shutil.rmtree(results_directory)
    os.makedirs(results_directory, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # EfficientNetV2 expects images of size 224x224, so we need to resize them
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load train and test
    train_dataset = torchvision.datasets.CIFAR100(root=os.path.join(current_dir, 'data'), train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR100(root=os.path.join(current_dir, 'data'), train=False, download=True, transform=transform_test)

    # Split train into train and validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = EfficientNetWithLinearClassifier().to(device)

    # Calculate and print the number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in the non-spectral model: {total_params}")

    # Calculate and print number or elements in training set
    num_train_elements = batch_size * len(train_loader)
    print(f"Number of elements in the training set: {num_train_elements}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop with validation and loss tracking

    # Measure resource usage and time for training
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(device)
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, average_memory_allocated = train(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
            f"Avg Memory Allocated: {average_memory_allocated:.2f} MB")

    end_time = time.time()
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2    # MB

    print("Training complete.")
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print(f"Peak allocated memory: {peak_allocated:.2f} MB")
    print(f"Peak reserved memory: {peak_reserved:.2f} MB")
    print(f"Average memory allocated during training: {average_memory_allocated:.2f} MB")

    acc = test(model, test_loader, criterion)

    # Spectral model initialization
    spectral_model = EfficientNetWithSpectralClassifier().to(device)

    # Loss and optimizer
    optimizer = optim.Adam(spectral_model.parameters(), lr=learning_rate)

    # Measure resource usage and time for training
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(device)
    spectral_start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, average_memory_allocated = train(spectral_model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(spectral_model, val_loader, criterion)

        print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
            f"Avg Memory Allocated: {average_memory_allocated:.2f} MB")

    print("Training complete.")

    spectral_end_time = time.time()
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2    # MB

    print(f"Training time: {spectral_end_time - spectral_start_time:.2f} seconds")
    print(f"Peak allocated memory: {peak_allocated:.2f} MB")
    print(f"Peak reserved memory: {peak_reserved:.2f} MB")
    print(f"Average memory allocated during training: {average_memory_allocated:.2f} MB")

    # Make eigenvalues graphs
    for name, param in spectral_model.named_parameters():
        if 'diag' in name:
            values = param.detach().cpu().numpy().flatten()
            plt.figure(figsize=(10, 6))
            plt.hist(values, bins=20, alpha=0.5, edgecolor='black', label=name)
            plt.xlabel('Value')
            plt.ylabel('Count')
            plt.title(f'Eigenvalues Distribution: {name.replace("_diag", "")}')
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-3, 3))
            formatter.set_scientific(True)
            plt.gca().xaxis.set_major_formatter(formatter)
            savepath = os.path.join(results_directory, f'{name}_histogram.png')
            plt.savefig(savepath)
            plt.clf()  # Clear the current figure for the next plot

    spectral_acc = test(spectral_model, test_loader, criterion)

    print(f'Difference between spectral and non-spectral accuracies: {spectral_acc-acc:.2f}%')
    print(f'Difference between spectral and non-spectral time: {spectral_end_time - spectral_start_time - (end_time - start_time):.2f} seconds')