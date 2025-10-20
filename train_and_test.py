"""
Simple test of SPARCS module on MNIST with 3 layers, including validation for early stopping and plotting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

import logging
import os
import argparse

from datasets import load_dataset
from torch.utils.data import Dataset

from sparcs import sparcs_module


class HFDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Ensure data is float and label is long
        return torch.tensor(item['x'], dtype=torch.float32), torch.tensor(item['y'], dtype=torch.long)


def get_data_loaders(dataset_name, batch_size, validation_split):
    """
    Returns the data loaders for the specified dataset.
    """
    data_root = "./data/"
    if dataset_name == 'mnist_1d':
        # Load from Hugging Face
        hf_dataset = load_dataset("christopher/mnist1d")
        full_train_dataset = HFDataset(hf_dataset['train'])
        test_dataset = HFDataset(hf_dataset['test'])
        input_dim = 40  # As specified by the dataset
    elif dataset_name == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        input_dim = 28 * 28
        full_train_dataset = datasets.FashionMNIST(data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_root, train=False, transform=transform)
    elif dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        input_dim = 28 * 28
        full_train_dataset = datasets.MNIST(data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_root, train=False, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


    # Split training data into training and validation sets
    num_train = int((1 - validation_split) * len(full_train_dataset))
    num_val = len(full_train_dataset) - num_train
    train_dataset, val_dataset = random_split(full_train_dataset, [num_train, num_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, input_dim


def main(args):
    # --- Create results directory ---
    results_dir = os.path.join('results', args.dataset)
    os.makedirs(results_dir, exist_ok=True)

    # --- Setup logging ---
    log_file = os.path.join(results_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    logger.info(f"Results will be saved to: {results_dir}")

    # --- Hyperparameters ---
    batch_size = 64
    epochs = 500
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_split = 0.1
    patience = 50
    gradient_clipping = 1.0

    # --- Data ---
    train_loader, val_loader, test_loader, input_dim = get_data_loaders(args.dataset, batch_size, validation_split)

    # --- Model ---
    hidden_dim = 256
    output_dim = 10
    model = sparcs_module.SPARCS([input_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim, output_dim], activation="relu", bias=True, dropout=0.2).to(device)

    # --- Loss & Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=patience//2)
    
    # --- Training Loop with Early Stopping ---
    train_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            loss += model.reg_term(reg_cost=1e-3)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100. * correct / len(val_loader.dataset)
        val_accuracies.append(val_accuracy)

        logger.info(f"Epoch: {epoch+1}/{epochs} | Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f} | Val Acc: {val_accuracy:.2f}%")

        # Learning rate scheduler step
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            torch.save(best_model_state, os.path.join(results_dir, 'best_model.pth'))
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve == patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Load the best model for testing
    if best_model_state:
        model.load_state_dict(best_model_state)
    else:
        logger.info("No best model state found, using the last model.")

    # --- Plotting ---
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
    plt.title('Validation Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_plots.png'))

    # --- Testing ---
    # Complete test
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    acc = 100. * correct / len(test_loader.dataset)
    logger.info(f"Test Accuracy: {acc:.2f}%")

    # Plot histogram of eigenvalues
    all_eigenvalues = torch.cat([p.data for p in model.lambda_diags]).cpu().numpy()
    plt.figure(figsize=(8, 5))
    plt.hist(np.abs(all_eigenvalues), bins=50, color='blue', alpha=0.7)
    plt.title('Histogram of Eigenvalues')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'eigenvalues_histogram.png'))

    # Test only for top eigenvalues
    top_eigenvalues_numbers = [1000, 500, 200, 100, 50, 20, 10, 5, 1] # Important to have them in decreasing order
    for num in top_eigenvalues_numbers:
        pruned_model = model.select_just_best_projectors(num_best=num)
        pruned_model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = pruned_model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        acc = 100. * correct / len(test_loader.dataset)
        logger.info(f"Test Accuracy with top {num} eigenvalues: {acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and test SPARCS model on different datasets.')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion_mnist', 'mnist_1d'],
                        help='Dataset to use for training and testing.')
    args = parser.parse_args()
    main(args)