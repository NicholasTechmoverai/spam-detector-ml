import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------ Load Data ------------------
data_path = "./data/SMSSpamCollection.csv"
df = pd.read_csv(
    data_path,
    sep="\t",
    header=None,
    names=["label", "text"]
)

# Map labels to numbers
lbl = {"ham": 0, "spam": 1}
df['label'] = df['label'].map(lbl)

print(f"Dataset shape: {df.shape}")
print(f"Spam percentage: {df['label'].mean()*100:.2f}%")
print(df.head())

# ------------------ Vectorize Text ------------------
vectorizer = TfidfVectorizer(max_features=1000, lowercase=True, stop_words='english')
X_vec = vectorizer.fit_transform(df['text']).toarray()  # dense array
y_vec = df['label'].values

# ------------------ Train/Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y_vec, test_size=0.2, random_state=42, shuffle=True
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test  = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Optional: Create DataLoader for mini-batch training
train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# ------------------ Define Neural Network ------------------
class SpamDetectModel(nn.Module):
    def __init__(self, input_size=1000):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # single output for binary classification
        )

    def forward(self, x):
        return self.network(x)

model = SpamDetectModel(input_size=X_train.shape[1])

# ------------------ Loss and Optimizer ------------------
loss_fn = nn.BCEWithLogitsLoss()  # suitable for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ------------------ Training Loop ------------------
epochs = 20
losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * X_batch.size(0)
    
    epoch_loss /= len(train_loader.dataset)
    losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")

# ------------------ Evaluation ------------------
model.eval()
with torch.no_grad():
    logits = model(X_test)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    accuracy = (preds.eq(y_test).sum() / y_test.shape[0]).item()
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")

# ------------------ Plot Loss ------------------
plt.figure(figsize=(6,4))
plt.plot(losses, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
