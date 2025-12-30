import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Load the data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Data Preprocessing
# Combine train and test data for consistent preprocessing
test_data['Transported'] = np.nan
combined_data = pd.concat([train_data, test_data], sort=False)

# Convert CryoSleep and VIP to categorical
combined_data['CryoSleep'] = combined_data['CryoSleep'].astype('category').cat.codes
combined_data['VIP'] = combined_data['VIP'].astype('category').cat.codes

# Extract deck, num, and side from Cabin
combined_data[['Deck', 'Num', 'Side']] = combined_data['Cabin'].str.split('/', expand=True)
combined_data['Deck'] = combined_data['Deck'].astype('category').cat.codes
combined_data['Side'] = combined_data['Side'].astype('category').cat.codes

# Fill missing values
combined_data['Age'].fillna(combined_data['Age'].mean(), inplace=True)
combined_data['RoomService'].fillna(combined_data['RoomService'].mean(), inplace=True)
combined_data['FoodCourt'].fillna(combined_data['FoodCourt'].mean(), inplace=True)
combined_data['ShoppingMall'].fillna(combined_data['ShoppingMall'].mean(), inplace=True)
combined_data['Spa'].fillna(combined_data['Spa'].mean(), inplace=True)
combined_data['VRDeck'].fillna(combined_data['VRDeck'].mean(), inplace=True)

def cryo_or_no(row):
    if row['Spa'] + row['RoomService'] + row['FoodCourt'] + row['ShoppingMall'] + row['VRDeck'] == 0.0:
        return True
    else:
        return False
    
def guess_dest(row):
    if row['FoodCourt'] >= 425.0 or row['VRDeck'] >= 300.0:
        return "55 Cancri e"
    elif row['FoodCourt'] <= 200.0 or row['VRDeck'] <= 200.0:
        return "PSO J318.5-22"
    else:
        return "TRAPPIST-1e"

def guess_home(row):
    if row['Spa'] >= 400.0 or row['VRDeck'] >= 400.0:
        return "Europa"
    elif row['RoomService'] >= 250.0 or row['ShoppingMall'] >= 250.0:
        return "Mars"
    else:
        return "Earth"

combined_data['HomePlanet'] = combined_data.apply(
    lambda row: guess_home(row) if pd.isnull(row['HomePlanet']) else row['HomePlanet'],
    axis=1 
) 
combined_data['CryoSleep'] = combined_data.apply(
    lambda row: cryo_or_no(row) if pd.isnull(row['CryoSleep']) else row['CryoSleep'],
    axis=1
)
combined_data['Destination'] = combined_data.apply(
    lambda row: guess_dest(row) if pd.isnull(row['Destination']) else row['Destination'],
    axis=1 
)
combined_data['VIP'].fillna(False, inplace=True)
combined_data['Deck'].fillna(combined_data['Deck'].mode()[0], inplace=True)
combined_data['Side'].fillna(combined_data['Side'].mode()[0], inplace=True)

# Encode categorical features
for col in ['HomePlanet', 'Destination']:
    combined_data[col] = LabelEncoder().fit_transform(combined_data[col])

# Split combined data back into train and test sets
train_data = combined_data[combined_data['Transported'].notna()]
test_data = combined_data[combined_data['Transported'].isna()]

# Keep PassengerId for submission
test_passenger_ids = test_data['PassengerId'].reset_index(drop=True)

# Feature selection: Remove unnecessary columns
train_data = train_data.drop(columns=['PassengerId', 'Name', 'Num', 'Cabin', 'VIP', 'Side'])
test_data = test_data.drop(columns=['PassengerId', 'Name', 'Num', 'Cabin', 'VIP', 'Side', 'Transported'])

# Split features and target variable
X_train = train_data.drop(columns=['Transported'])
y_train = train_data['Transported']

X_test = test_data

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Split training data into training and validation sets directly
train_indices, val_indices = train_test_split(range(X_train_tensor.size(0)), test_size=0.2, random_state=42)
train_dataset = TensorDataset(X_train_tensor[train_indices], y_train_tensor[train_indices])
val_dataset = TensorDataset(X_train_tensor[val_indices], y_train_tensor[val_indices])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return self.softmax(out)

# Hyperparameters
input_size = X_train.shape[1]
hidden_size = 50
output_size = 2
learning_rate = 0.001
num_epochs = 20

# Model, loss function, optimizer
model = NeuralNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            val_outputs = model(X_val_batch)
            val_loss += criterion(val_outputs, y_val_batch).item()
    
    val_loss /= len(val_loader)
    scheduler.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')

# Predict on test data
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predictions = torch.max(outputs, 1)

# Prepare the submission file
predictions = predictions.cpu().int().tolist()
submission = pd.DataFrame({'PassengerId': test_passenger_ids, 'Transported': predictions})
submission['Transported'] = submission['Transported'].map({1: True, 0: False})
submission.to_csv("submission.csv", index=False)