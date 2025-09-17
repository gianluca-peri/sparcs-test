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

from sparcs import sparcs_module


def main():
    # --- Hyperparameters ---
    batch_size = 64
    epochs = 128
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_split = 0.1
    patience = 10
    gradient_clipping = 1.0

    # --- Data ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # flatten 28x28 → 784
    ])
    
    # Load full training data
    full_train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    
    # Split training data into training and validation sets
    num_train = int((1 - validation_split) * len(full_train_dataset))
    num_val = len(full_train_dataset) - num_train
    train_dataset, val_dataset = random_split(full_train_dataset, [num_train, num_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_loader = DataLoader(
        datasets.MNIST("./data", train=False, transform=transform),
        batch_size=batch_size, shuffle=False
    )

    # --- Model ---
    input_dim = 28 * 28
    hidden_dim = 256
    output_dim = 10
    model = sparcs_module.SPARCS([input_dim, hidden_dim, hidden_dim, output_dim], activation="relu", bias=True).to(device)

    # --- Loss & Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=patience//2)

    # --- Training Loop with Early Stopping ---
    train_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            loss += model.reg_term(reg_cost=1e-4)  # L1 regularization term

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

        print(f"Epoch: {epoch+1}/{epochs} | Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f} | Val Acc: {val_accuracy:.2f}%")

        # Learning rate scheduler step
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve == patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Load the best model for testing
    model.load_state_dict(torch.load('best_model.pth'))

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
    plt.savefig('training_plots.png')
    plt.show()

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
    print(f"Test Accuracy: {acc:.2f}%")

    # Test only for top eigenvalues
    model.reset_weights()  # Ensure weights are rebuilt
    top_eigenvalues_numbers = [1000, 500, 200, 100, 50, 20, 10, 5, 1] # Important to have them in decreasing order
    for num in top_eigenvalues_numbers:
        model.select_just_best_projectors(num_best=num)
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        acc = 100. * correct / len(test_loader.dataset)
        print(f"Test Accuracy with top {num} eigenvalues: {acc:.2f}%")
        model.reset_weights()  # Reset weights for next iteration


if __name__ == "__main__":
    main()