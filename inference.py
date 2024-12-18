import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from model_src.net import Net


def input_args():
  parser = argparse.ArgumentParser(description="Inference test data")
  parser.add_argument("--features", type = str, help="path to the test features")
  parser.add_argument("--labels", type = str, help="path to the test labels")
  args = parser.parse_args()
  return args.features, args.labels

test_features, test_labels = input_args()

test_features_tensor = torch.load(test_features) 
test_labels_tensor = torch.load(test_labels)

print("Shape of test_features_tensor:", test_features_tensor.shape)
print("Shape of test_labels_tensor:", test_labels_tensor.shape)

test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

input_dim = 282
num_classes = 24
net = Net(input_dim=input_dim, num_classes=num_classes)

best_model = torch.load('best_model.pth')
net.load_state_dict(best_model)
net.eval()

test_loss = 0.0
correct = 0
total = 0

true_labels_list = []
predictions_list = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)

loss = torch.nn.CrossEntropyLoss()

with torch.no_grad(): 
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = net(features)
        
        loss_val = loss(outputs, labels)
        test_loss += loss_val.item() * len(labels)
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        true_labels_list.extend(labels.cpu().numpy())  # Move to CPU and convert to numpy
        predictions_list.extend(predicted.cpu().numpy())

test_loss /= total
test_accuracy = 100 * correct / total

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.2f}%')

marathon_names_df = pd.read_csv('test_marathon_names.csv')

results_df = pd.DataFrame({
    'Marathon Name': marathon_names_df["Marathon Name"],
    'True Labels': true_labels_list,
    'Predictions': predictions_list
})

results_df.to_csv('test_predictions.csv', index=False)
print("Predictions saved to 'test_predictions.csv'")
