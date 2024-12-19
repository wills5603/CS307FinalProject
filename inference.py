import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, TensorDataset
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import wandb
from sklearn.utils import resample
from model_src.net import Net

wandb.login()
wandb.init(
    project="Final Project",
    config={
      "learning_rate": 0.001,
      "architecture": "MLP",
      "epochs": 100,
    }
)

data = pd.read_csv('marathon_training_data.csv')

time_intervals = [(120 * 60 + 10 * 60 * i, 120 * 60 + 10 * 60 * (i + 1)) for i in range(24)]  # 24 intervals (2:00 to 6:00)

# Assign each marathon time (in seconds) to the corresponding interval
def assign_time_interval(time_in_seconds):
    for i, (start, end) in enumerate(time_intervals):
        if start <= time_in_seconds < end:
            return i  # Return the class index (the bin number)
    return len(time_intervals) - 1  # In case the time is out of the predefined range (shouldn't happen here)

data['IntervalClass'] = data['MarathonTime'].apply(assign_time_interval)

first_entry = data.iloc[[0]]  # Extract the first row
data = data.iloc[1:].reset_index(drop=True) 
data = data.sample(frac=1, random_state=4).reset_index(drop=True)
data = pd.concat([data, first_entry], ignore_index=True)

train_size = int(0.8 * len(data))
val_size = int(0.2 * train_size)
train_data = data[:train_size - val_size]
val_data = train_data[train_size - val_size:train_size]
test_data = data[train_size:]

majority_class = train_data[train_data['IntervalClass'] == 4]
minority_classes = train_data[train_data['IntervalClass'] != 4]

balanced_train_data = pd.DataFrame()  # Initialize an empty DataFrame

# Loop through each class in the training data
for interval_class in train_data['IntervalClass'].unique():
    class_subset = train_data[train_data['IntervalClass'] == interval_class]
    if len(class_subset) < len(majority_class):  # Oversample if the class is a minority
        class_subset = resample(class_subset, 
                                replace=True, 
                                n_samples=len(majority_class), 
                                random_state=42)
    balanced_train_data = pd.concat([balanced_train_data, class_subset])

# Shuffle the balanced dataset
train_data = balanced_train_data.sample(frac=1, random_state=42).reset_index(drop=True)
class_counts = train_data['IntervalClass'].value_counts()
print("Class Distribution in Training Dataset After Oversampling:")
print(class_counts)
print("Class Distribution in Test Dataset:")
print(test_data['IntervalClass'].value_counts())

data = pd.concat([train_data, val_data, test_data], ignore_index=True)

# Prepare the labels based on marathon time (in seconds)
all_features = data.drop(columns=['MarathonTime', 'IntervalClass'])
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)
all_features = pd.get_dummies(all_features, dummy_na=True)

train_features = torch.from_numpy(all_features[:len(train_data) - val_size].values)
val_features = torch.from_numpy(all_features[len(train_data) - val_size:len(train_data)].values)
test_features = torch.from_numpy(all_features[len(train_data):].values)

train_labels = torch.from_numpy(train_data.IntervalClass.values).view(-1, 1) 
val_labels = train_labels[-12:]  # Take the last 12 labels
train_labels = train_labels[:-12]
test_labels = torch.from_numpy(test_data.IntervalClass.values).view(-1, 1) 

print(f"Train Features: {train_features.shape}, Train Labels: {train_labels.shape}")
print(f"Validation Features: {val_features.shape}, Validation Labels: {val_labels.shape}")
print(f"Test Features: {test_features.shape}, Test Labels: {test_labels.shape}")

train_features = train_features.float()
test_features = test_features.float()
val_features = val_features.float()
train_labels = train_labels.long().squeeze()
val_labels = val_labels.long().squeeze()
test_labels = test_labels.long().squeeze()
torch.save(test_features, 'test_features.pt')
torch.save(test_labels, 'test_labels.pt')

train_dataset = TensorDataset(train_features, train_labels)
val_dataset = TensorDataset(val_features, val_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

input_dim = all_features.shape[1]
num_classes = 24
net = Net(input_dim=input_dim, num_classes=num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)

loss = torch.nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.01)

def calculate_loss(net, data_loader, device, loss):
  net.eval()
  total_loss = 0
  count = 0
  with torch.no_grad():
    for features, labels in data_loader:
      features, labels = features.to(device), labels.to(device)
      outputs = net(features)
      loss_val = loss(outputs, labels)
      total_loss += loss_val.item() * len(labels)
      count += len(labels)
  return total_loss / count

def plot_loss(train_loss, val_loss):
  plt.figure(figsize=(8, 6))
  plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train Loss')
  plt.plot(range(1, len(train_loss) + 1), val_loss, label='Validation Loss', linestyle=':')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Train and Validation Loss')
  plt.legend()
  plt.grid(True)
  plt.show()

num_epochs = 100
train_loss, val_loss = [], []
best_val = float('inf')
best_model = None
for epoch in range(num_epochs):
  net.train()
  for features, labels in train_loader:
    features, labels = features.to(device), labels.to(device)
    optimizer.zero_grad()
    predictions = net(features)
    loss_val = loss(predictions, labels)
    loss_val.backward()
    optimizer.step()
  train_ls = calculate_loss(net, train_loader, device, loss)
  val_ls = calculate_loss(net, val_loader, device, loss)
  train_loss.append(train_ls)
  val_loss.append(val_ls)
  
  wandb.log({
    "epoch": epoch + 1,
    "train_loss": train_ls,
    "val_loss": val_ls
  })

  if val_ls < best_val:
      best_val = val_ls
      best_model = net.state_dict()

  if epoch % 10 == 0:
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_ls:.4f}, Validation Loss: {val_ls:.4f}')

torch.save(best_model, 'best_model.pth')
print("Model saved as best_model.pth")
net.load_state_dict(best_model)
net.eval()
# plot_loss(train_loss, val_loss)

wandb.finish()
