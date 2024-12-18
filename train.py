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
majority_class = data[data['IntervalClass'] == 4]
minority_classes = data[data['IntervalClass'] != 4]

# Oversample the minority classes
minority_oversampled = resample(minority_classes, 
                                replace=True, 
                                n_samples=len(majority_class), 
                                random_state=42)

# Combine back into a balanced dataset
balanced_data = pd.concat([majority_class, minority_oversampled])

# Shuffle the balanced dataset to randomize the row order
data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
data = pd.concat([data, first_entry], ignore_index=True)

class_counts = data['IntervalClass'].value_counts().sort_index()

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.bar(class_counts.index, class_counts.values, width=0.8, edgecolor="black")
plt.xlabel("Interval Class")
plt.ylabel("Number of Samples")
plt.title("Histogram of Interval Classes After Resampling")
plt.xticks(class_counts.index)  # Show all class indices on x-axis
plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()

train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]  # Add first entry to test set
marathon_names = test_data['Marathon Name']
marathon_names.to_csv('test_marathon_names.csv', index=False, header=True)

# Prepare the labels based on marathon time (in seconds)
all_features = data.drop(columns=['MarathonTime', 'IntervalClass'])

numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)
all_features = pd.get_dummies(all_features, dummy_na=True)

train_features = torch.from_numpy(all_features[:train_size].values)
test_features = torch.from_numpy(all_features[train_size:].values)
train_labels = torch.from_numpy(train_data.IntervalClass.values).view(-1, 1) 
test_labels = torch.from_numpy(test_data.IntervalClass.values).view(-1, 1) 


train_features = train_features.float()
test_features = test_features.float()
train_labels = train_labels.long().squeeze()
test_labels = test_labels.long().squeeze()
torch.save(test_features, 'test_features.pt')
torch.save(test_labels, 'test_labels.pt')


train_size = int(0.8 * len(train_features))
val_size = len(train_features) - train_size
train_dataset, val_dataset = random_split(TensorDataset(train_features, train_labels), [train_size, val_size])

train_indices = train_dataset.indices
val_indices = val_dataset.indices
train_data_with_names = train_data.iloc[train_indices]["Marathon Name"].tolist()
val_data_with_names = train_data.iloc[val_indices]["Marathon Name"].tolist()

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

train_true_labels = []
train_predictions = []
val_true_labels = []
val_predictions = []

# Step 1: Define a function to evaluate the model and collect predictions
def get_predictions(data_loader, net, device):
  true_labels = []
  predictions = []
  net.eval()  # Set the model to evaluation mode
  with torch.no_grad():
    for features, labels in data_loader:
      features, labels = features.to(device), labels.to(device)
      outputs = net(features)
      _, predicted = torch.max(outputs, 1)
      true_labels.extend(labels.cpu().numpy())
      predictions.extend(predicted.cpu().numpy())
  return true_labels, predictions

# Step 2: Get predictions for the training set
train_true_labels, train_predictions = get_predictions(train_loader, net, device)

# Step 3: Get predictions for the validation set
val_true_labels, val_predictions = get_predictions(val_loader, net, device)

# Step 4: Combine the predictions with true labels into a DataFrame
train_results_df = pd.DataFrame({
    "Marathon Name": train_data_with_names,
    "True Labels (Train)": train_true_labels,
    "Predictions (Train)": train_predictions
})

val_results_df = pd.DataFrame({
    "Marathon Name": val_data_with_names,
    "True Labels (Val)": val_true_labels,
    "Predictions (Val)": val_predictions
})

# Step 5: Save the predictions to a CSV file
train_results_df.to_csv('train_predictions.csv', index=False)
val_results_df.to_csv('val_predictions.csv', index=False)

print("Train and validation predictions saved to 'train_predictions.csv' and 'val_predictions.csv'")
wandb.finish()
