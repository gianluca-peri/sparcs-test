import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import argparse
from sparcs import sparcs_module
from tqdm import tqdm
from train_and_test import get_data_loaders


def main(args):
    # --- 1. Get args and define results directory ---
    results_dir = os.path.join('results', args.dataset)
    model_path = os.path.join(results_dir, 'best_model.pth')

    # --- 2. Get best model from folder, throw error if not present ---
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run train_and_test.py first.")

    # --- Hyperparameters ---
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data ---
    _, _, test_loader, input_dim = get_data_loaders(args.dataset, batch_size, validation_split=0.0)

    # --- Model ---
    hidden_dim = 256
    output_dim = 10
    # The architecture should match the one used in training
    model = sparcs_module.SPARCS([input_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim, output_dim], activation="relu", bias=True).to(device)
    
    # Load the best model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded best model from {model_path}")

    # --- 3. Test performance on test set without any modifications ---
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    base_accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test Accuracy (full model): {base_accuracy:.2f}%")

    # --- 4. Graph of test accuracy over number of eigenvalues used ---
    
    # Get all eigenvalues
    all_eigenvalues = torch.cat([p.data for p in model.lambda_diags])
    num_eigenvalues = len(all_eigenvalues)

    print(f"\nTotal number of eigenvalues of the model: {num_eigenvalues}")

    accuracies = []


    print("\nTesting accuracy with a subset of top eigenvalues...")
    for k in tqdm(range(1, num_eigenvalues + 1)):
        # Select top k eigenvalues based on magnitude
        pruned_model = model.select_just_best_projectors(num_best=k)
        pruned_model.eval()
        
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = pruned_model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        acc = 100. * correct / len(test_loader.dataset)
        accuracies.append(acc)

    # --- Plotting the results ---
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_eigenvalues + 1), accuracies)
    plt.title(f'Test Accuracy vs. Number of Top Eigenvalues ({args.dataset})')
    plt.xlabel('Number of Top Eigenvalues Used')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    
    # Save the plot
    plot_path = os.path.join(results_dir, 'accuracy_vs_eigenvalues.png')
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the best SPARCS model and analyze eigenvalue impact.')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion_mnist', 'mnist_1d'],
                        help='Dataset to use for testing.')
    args = parser.parse_args()
    main(args)
