import os
from matplotlib.ticker import ScalarFormatter
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import shutil

# Reference for the base model:
# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# Choose a readable font size for the images
plt.rcParams.update({'font.size': 16})

class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5), # (B, 3, 32, 32) → (B, 6, 28, 28)
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 6, 28, 28) → (B, 6, 14, 14)

            nn.Conv2d(6, 16, kernel_size=5), # (B, 6, 14, 14) → (B, 16, 10, 10)
            nn.BatchNorm2d(16),    
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 16, 10, 10) → (B, 16, 5, 5)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # (B, 16, 5, 5) → (B, 400)
            nn.Linear(400, num_classes) 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class SpectralCNN(CustomCNN):
    def __init__(self, num_classes=10):
        super().__init__(num_classes)
        self.input_dim = 16 * 5 * 5 + 1
        self.hidden_dim = 526
        self.output_dim = num_classes

        self.varphi1 = nn.Parameter(torch.empty(self.hidden_dim, self.input_dim))
        self.varphi2 = nn.Parameter(torch.empty(self.output_dim, self.hidden_dim))
        self.l1_diag = nn.Parameter(torch.zeros(self.input_dim), requires_grad=False)
        self.l2_diag = nn.Parameter(torch.zeros(self.hidden_dim), requires_grad=True)
        self.l3_diag = nn.Parameter(torch.ones(self.output_dim), requires_grad=True)
        self.spectral_activ = nn.ReLU()
        nn.init.xavier_uniform_(self.varphi1)
        nn.init.xavier_uniform_(self.varphi2)

    def forward(self, x):
        x = self.features(x)
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
    running_loss = 0.0
    correct = 0
    memory_allocated_list = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Regularize on the spectral parameters
        if isinstance(model, SpectralCNN):
            loss += 0.0001 * model.l2_diag.abs().sum()  # l1 regularization on l2_diag

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
    running_loss = 0.0
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
    num_epochs = 300
    batch_size = 128
    learning_rate = 0.001
    number_of_runs = 3

    # Select the second GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # Get path to the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the results directory if it doesn't exist
    results_directory = os.path.join(current_dir, 'Experiment-1-Results')
    
    # Remove the content of the results directory if it exists
    if os.path.exists(results_directory):
        shutil.rmtree(results_directory)
    os.makedirs(results_directory, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load CIFAR-10 without normalization to get the normalization constants
    print("Calculating normalization constants for CIFAR-10 dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root=os.path.join(current_dir, 'data'), train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    mean = 0.0
    std = 0.0
    nb_samples = 0

    for data in loader:
        images, _ = data
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    
    # Define the normalization transform for train
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Define the normalization transform for test
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Load train and test
    print("Loading CIFAR-10 dataset...")
    train_dataset = torchvision.datasets.CIFAR10(root=os.path.join(current_dir, 'data'), train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root=os.path.join(current_dir, 'data'), train=False, download=True, transform=transform_test)

    # Split train into train and validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = CustomCNN().to(device)

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

    # Save the model
    model_save_path = os.path.join(results_directory, 'cnn_model.pth')
    torch.save(model.state_dict(), model_save_path)

    # Spectral model initialization
    spectral_model = SpectralCNN().to(device)

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

    # Save the spectral model
    spectral_model_save_path = os.path.join(results_directory, 'spectral_cnn_model.pth')
    torch.save(spectral_model.state_dict(), spectral_model_save_path)

    # Make new validation plot but with 3 trainings and the error bar represented as shaded areas
    # Make also the training curves

    print("Starting multiple runs...")

    trainings_non_spectral = []
    trainings_spectral = []
    validations_non_spectral = []
    validations_spectral = []
    validations_spectral_pruned = []

    for i in range(number_of_runs):
        print(f"\nRun {i+1}/{number_of_runs}\n")
        model = CustomCNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_accs = []
        val_accs = []

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc, _ = train(model, train_loader, criterion, optimizer)
            val_loss, val_acc = validate(model, val_loader, criterion)

            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        trainings_non_spectral.append(train_accs)
        validations_non_spectral.append(val_accs)

        spectral_model = SpectralCNN().to(device)
        optimizer = optim.Adam(spectral_model.parameters(), lr=learning_rate)

        train_accs_spectral = []
        val_accs_spectral = []
        val_accs_spectral_pruned = []

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc, _ = train(spectral_model, train_loader, criterion, optimizer)
            val_loss, val_acc = validate(spectral_model, val_loader, criterion)
            val_loss_pruned, val_acc_pruned = validate_spectral_pruned(spectral_model, val_loader, criterion)

            train_accs_spectral.append(train_acc)
            val_accs_spectral.append(val_acc)
            val_accs_spectral_pruned.append(val_acc_pruned)

            print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Acc (Pruned): {val_acc_pruned:.2f}%")

        trainings_spectral.append(train_accs_spectral)
        validations_spectral.append(val_accs_spectral)
        validations_spectral_pruned.append(val_accs_spectral_pruned)

    # Convert lists to numpy arrays for easier manipulation
    trainings_non_spectral = np.array(trainings_non_spectral)
    validations_non_spectral = np.array(validations_non_spectral)
    trainings_spectral = np.array(trainings_spectral)
    validations_spectral = np.array(validations_spectral)
    validations_spectral_pruned = np.array(validations_spectral_pruned)

    # Calculate means and standard deviations
    mean_train_non_spectral = np.mean(trainings_non_spectral, axis=0)
    std_train_non_spectral = np.std(trainings_non_spectral, axis=0)

    mean_val_non_spectral = np.mean(validations_non_spectral, axis=0)
    std_val_non_spectral = np.std(validations_non_spectral, axis=0)

    mean_train_spectral = np.mean(trainings_spectral, axis=0)
    std_train_spectral = np.std(trainings_spectral, axis=0)

    mean_val_spectral = np.mean(validations_spectral, axis=0)
    std_val_spectral = np.std(validations_spectral, axis=0)

    mean_val_spectral_pruned = np.mean(validations_spectral_pruned, axis=0)
    std_val_spectral_pruned = np.std(validations_spectral_pruned, axis=0)

    # Plot training curves with error bars
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), mean_train_non_spectral, label="Train Accuracy (CNN)", color='blue')
    plt.fill_between(range(1, num_epochs + 1), 
                     mean_train_non_spectral - std_train_non_spectral, 
                     mean_train_non_spectral + std_train_non_spectral,
                     color='blue', alpha=0.2)
    plt.plot(range(1, num_epochs + 1), mean_train_spectral, label="Train Accuracy (Spectral CNN)", color='green')
    plt.fill_between(range(1, num_epochs + 1), 
                     mean_train_spectral - std_train_spectral, 
                     mean_train_spectral + std_train_spectral,
                     color='green', alpha=0.2)
    
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Accuracy Comparison")
    plt.legend()
    plt.grid(True)
    savepath = os.path.join(results_directory, 'train_accuracy_comparison_with_error_bars.png')
    plt.savefig(savepath)

    # Plot validation curves with error bars
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), mean_val_non_spectral, label="Validation Accuracy (CNN)", color='blue')
    plt.fill_between(range(1, num_epochs + 1), 
                     mean_val_non_spectral - std_val_non_spectral, 
                     mean_val_non_spectral + std_val_non_spectral,
                     color='blue', alpha=0.2)
    plt.plot(range(1, num_epochs + 1), mean_val_spectral, label="Validation Accuracy (Spectral CNN)", color='green')
    plt.fill_between(range(1, num_epochs + 1), 
                     mean_val_spectral - std_val_spectral, 
                     mean_val_spectral + std_val_spectral,
                     color='green', alpha=0.2)
    plt.plot(range(1, num_epochs + 1), mean_val_spectral_pruned, label="Validation Accuracy (Spectral CNN Pruned)", color='red')
    plt.fill_between(range(1, num_epochs + 1), 
                     mean_val_spectral_pruned - std_val_spectral_pruned, 
                     mean_val_spectral_pruned + std_val_spectral_pruned,
                     color='red', alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy Comparison")
    plt.legend()
    plt.grid(True)
    savepath = os.path.join(results_directory, 'val_accuracy_comparison_with_error_bars.png')
    plt.savefig(savepath)