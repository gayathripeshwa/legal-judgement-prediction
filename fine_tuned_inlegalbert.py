import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import torch

# === SPEED SETTING ===
torch.set_num_threads(2)

# === Load Full Data ===
path_0 = "D:/Gayathri Gopal Peshwa/judgement/data/cjpe_multi_train_0000.parquet"
path_1 = "D:/Gayathri Gopal Peshwa/judgement/data/cjpe_multi_train_0001.parquet"

df = pd.concat([
    pd.read_parquet(path_0),
    pd.read_parquet(path_1)
]).reset_index(drop=True)

# Drop rows with missing data
df = df.dropna(subset=["text", "label"])
df["label"] = df["label"].astype(int)

# === Sample 3000 Conviction and 3000 Acquittal (or as much as available) ===
convictions = df[df["label"] == 1]
acquittals = df[df["label"] == 0]

sample_size = min(3000, len(convictions), len(acquittals))

convictions_sampled = convictions.sample(n=sample_size, random_state=42)
acquittals_sampled = acquittals.sample(n=sample_size, random_state=42)

balanced_df = pd.concat([convictions_sampled, acquittals_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

# === Stratified Train/Test Split ===
train_df, val_df = train_test_split(
    balanced_df,
    test_size=0.2,
    stratify=balanced_df["label"],
    random_state=42
)

print("\U0001F4CA Training label distribution:")
print(train_df["label"].value_counts())
print("\n\U0001F4CA Validation label distribution:")
print(val_df["label"].value_counts())

# === Convert to Hugging Face Dataset ===
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

# === Load Tokenizer and Model ===
MODEL_NAME = "law-ai/InLegalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# === Tokenization ===
def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding=True, max_length=512)

train_ds = train_ds.map(tokenize_fn, batched=True)
val_ds = val_ds.map(tokenize_fn, batched=True)

# === Compute Metrics ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    print(f"\n\U0001F4C8 Acc: {acc:.2%} | Precision: {precision:.2%} | Recall: {recall:.2%} | F1: {f1:.2%}")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir="inlegalbert_finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=6,
    weight_decay=0.01,
    logging_steps=10,
    disable_tqdm=False,
    report_to="none"
)

# === Compute class weights (if you want to train on full imbalanced dataset) ===
# For balanced sampling, class_weights won't be used, but included here for completeness
labels_for_weights = df["label"].values
class_weights_np = compute_class_weight(class_weight="balanced", classes=np.array([0,1]), y=labels_for_weights)

class_weights = torch.tensor(class_weights_np, dtype=torch.float)

# === Custom Trainer with weighted loss ===
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

# === Initialize Trainer ===
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    class_weights=class_weights
)

# === Train the Model ===
trainer.train()
trainer.save_model("inlegalbert_finetuned")
tokenizer.save_pretrained("inlegalbert_finetuned")

print("\n\U0001F4C5 Fine-tuning complete. Model saved to 'inlegalbert_finetuned'")
