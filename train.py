import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import pandas as pd
from collections import Counter
from utils.dataset import RhetoricalRoleDataset, collate_fn
from model import BiLSTM_CRF
import numpy as np

# ==== Load vocab and label2id ====
with open("vocab.pkl", "rb") as f:
    word2idx = pickle.load(f)
with open("label2id.pkl", "rb") as f:
    label2idx = pickle.load(f)

if "<PAD>" not in label2idx:
    label2idx["<PAD>"] = len(label2idx)
    with open("label2id.pkl", "wb") as f:
        pickle.dump(label2idx, f)

PAD_IDX = label2idx["<PAD>"]

# ==== Load training data ====
df = pd.read_parquet("data/rr_train.parquet")

# ==== Train Dataset and Dataloader ====
dataset = RhetoricalRoleDataset("data/rr_train.parquet", word2idx, label2idx)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# ==== Compute label frequencies ====
label_counts = Counter()
for _, label_batch, _ in train_loader:
    for label_seq in label_batch:
        label_counts.update(label_seq.tolist())

print("\n📊 Training Label Distribution:")
for label_id, count in sorted(label_counts.items()):
    label_name = [k for k, v in label2idx.items() if v == label_id][0]
    print(f"{label_id} ({label_name}): {count}")

# ==== Compute class weights ====
num_classes = len(label2idx)
weights = torch.ones(num_classes)
total = sum(label_counts.values())
for label_id in range(num_classes):
    if label_id == PAD_IDX:
        weights[label_id] = 0.0
    else:
        weights[label_id] = total / (label_counts.get(label_id, 1) * num_classes)

print("\n📐 Computed Class Weights:")
for label_id, weight in enumerate(weights.tolist()):
    label_name = [k for k, v in label2idx.items() if v == label_id][0]
    print(f"{label_id} ({label_name}): {weight:.4f}")

# ==== Model ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTM_CRF(len(word2idx), len(label2idx), 100, 128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=PAD_IDX)

# Patch the model’s loss function
def new_neg_log_likelihood(self, x, tags, lengths):
    emissions = self._get_lstm_features(x, lengths)
    return loss_fn(emissions.view(-1, self.tagset_size), tags.view(-1))

model.neg_log_likelihood = new_neg_log_likelihood.__get__(model, BiLSTM_CRF)

# ==== Training Loop ====
print("🚀 Starting training with class-weighted loss...")
for epoch in range(30):
    model.train()
    total_loss = 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/30")
    for x_batch, y_batch, lengths in progress:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        model.zero_grad()
        loss = model.neg_log_likelihood(x_batch, y_batch, lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())
    print(f"📉 Epoch {epoch+1} Loss: {total_loss:.4f}")

# ==== Save Model ====
torch.save(model.state_dict(), "models/bilstm_crf_model.pt")
print("✅ Model saved to models/bilstm_crf_model.pt")
