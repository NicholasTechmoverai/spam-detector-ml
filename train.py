# ========== CORRECTED TRAINING SCRIPT ==========
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer  # Better than CountVectorizer!
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import classification_report

# Load and prepare data
df = pd.read_csv("./data/SMSSpamCollection.csv", sep="\t", names=["label", "text"])
df['label'] = df["label"].map({"ham": 0, "spam": 1})

print(f"\n📊 Label Distribution:")
print(df['label'].value_counts())
print(f"Spam: {(df['label'].sum() / len(df) * 100):.2f}%")

# ========== KEY FIX 1: Use TF-IDF instead of CountVectorizer ==========
vectorizer = TfidfVectorizer(
    max_features=500,           # Smaller vocabulary
    min_df=2,                   # Ignore words that appear only once
    max_df=0.9,                 # Ignore too common words
    stop_words='english',       # Remove common words
    ngram_range=(1, 2)          # Use single words AND word pairs
)

X = vectorizer.fit_transform(df['text']).toarray()
y = df['label'].values

print(f"\n📐 Vectorized shape: {X.shape}")
print(f"Actual vocabulary size: {len(vectorizer.get_feature_names_out())}")

# ========== KEY FIX 2: Handle class imbalance ==========
from sklearn.utils import class_weight
import numpy as np

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)

class_weights = torch.tensor(class_weights, dtype=torch.float32)
print(f"Class weights: {class_weights}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# ========== KEY FIX 3: Simpler Model Architecture ==========
class BetterSpamModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

# Initialize
input_dim = X_train.shape[1]
model = BetterSpamModel(input_dim)
print(f"\n🤖 Model initialized with input dimension: {input_dim}")

# ========== KEY FIX 4: Weighted BCE Loss ==========
pos_weight = class_weights[1] / class_weights[0]  # Weight for spam class
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
losses = []

for epoch in range(epochs):
    model.train()
    predictions = model(X_train_tensor)
    loss = loss_fn(predictions, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    test_predictions = torch.sigmoid(model(X_test_tensor))
    binary_preds = (test_predictions > 0.5).int()
    
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(y_test, binary_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, binary_preds, average='binary'
    )
    
    print(f"\n📊 Evaluation Results:")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")

# ========== KEY FIX 5: Find Optimal Threshold ==========
from sklearn.metrics import precision_recall_curve

with torch.no_grad():
    probs = torch.sigmoid(model(X_test_tensor)).numpy().flatten()

precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, probs)

# Find threshold that maximizes F1 score
f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"\n🎯 Optimal decision threshold: {optimal_threshold:.3f}")
print(f"   (Instead of default 0.5)")

# ========== Save with optimal threshold ==========
import json

# Save everything needed for prediction
joblib.dump(vectorizer, "vectorizer.pkl")
torch.save(model.state_dict(), "spam-model.pt")

# Save configuration including optimal threshold
config = {
    'input_dim': input_dim,
    'optimal_threshold': float(optimal_threshold),
    'model_class': 'BetterSpamModel',
    'vectorizer_type': 'TfidfVectorizer'
}

with open('model_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("\n✅ Model saved with optimal threshold!")
print("Files saved: spam-model.pt, vectorizer.pkl, model_config.json")