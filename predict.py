import torch
import pickle
import pandas as pd
from model import BiLSTM_CRF
from utils.preprocessing import preprocess_text

# Load vocabulary and label mappings
with open("vocab.pkl", "rb") as f:
    word2idx = pickle.load(f)
with open("label2id.pkl", "rb") as f:
    label2id = pickle.load(f)

id2label = {v: k for k, v in label2id.items()}

# Load trained model
vocab_size = len(word2idx)
tagset_size = len(label2id)
embedding_dim = 100
hidden_dim = 128

model = BiLSTM_CRF(vocab_size, tagset_size, embedding_dim, hidden_dim)
model.load_state_dict(torch.load("models/bilstm_crf_model.pt", map_location=torch.device("cpu")))
model.eval()

# Load the CJPE dataset
df = pd.read_parquet("data/cjpe_multi_train_0000.parquet")
print(f"📦 Loaded {len(df)} rows from CJPE dataset.")

# Prepare for fact extraction
results = []

for i, row in df.iterrows():
    if i % 100 == 0:
        print(f"🔄 Processing row {i}/{len(df)}")


    text = row["text"]
    tokens = preprocess_text(text)
    token_ids = [word2idx.get(token, word2idx["<UNK>"]) for token in tokens]
    input_tensor = torch.tensor([token_ids], dtype=torch.long)
    lengths = [len(token_ids)]

    with torch.no_grad():
        emissions = model(input_tensor, lengths)
        predicted = model.decode(emissions)[0]
        predicted_labels = [id2label[idx] for idx in predicted]

    # Extract only "Facts" tokens (label '2')
    fact_tokens = [token for token, label in zip(tokens, predicted_labels) if label == '2']
    fact_text = " ".join(fact_tokens)

    results.append({
        "id": row["id"],
        "facts": fact_text
    })

# Save to CSV
output_path = "extracted_facts.csv"
pd.DataFrame(results).to_csv(output_path, index=False)
print(f"✅ Saved extracted facts to {output_path}")
