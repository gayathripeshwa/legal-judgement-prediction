import os
import csv
import pandas as pd
from transformers import AutoTokenizer, BertForSequenceClassification, TextClassificationPipeline
from tqdm import tqdm
import torch

# === CONFIG ===
model_path = "inlegalbert_finetuned"
base_input_folder = "v_summaries"
ground_truth_csv = "ground_truth.csv"
max_files = 3000
versions = ["V1", "V2", "V3", "V4"]

# === Updated label mapping: model label → numeric label
label_map = {
    "LABEL_0": 0,  # Acquittal
    "LABEL_1": 1   # Conviction
}

# === Validate model path ===
if not os.path.isdir(model_path):
    raise FileNotFoundError(f"❌ Model path not found: {model_path}")

# === Load Fine-Tuned Model ===
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
pipeline = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    top_k=1,
    truncation=True,
    max_length=512,
    return_all_scores=False,
    device=0 if torch.cuda.is_available() else -1
)

# === Read TXT and Extract Text ===
def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

# === Load ground truth and get list of filenames ===
gt_df = pd.read_csv(ground_truth_csv)
files_to_predict = sorted(gt_df["filename"].tolist())[:max_files]

# === Predict for each version ===
for version in versions:
    input_folder = os.path.join(base_input_folder, version)
    output_csv = f"predictions_{version.lower()}.csv"

    missing = empty = error = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "prediction", "confidence"])

        for file in tqdm(files_to_predict, desc=f"Predicting {version}"):
            file_txt = file.replace(".html", ".txt")
            file_path = os.path.join(input_folder, file_txt)

            if not os.path.exists(file_path):
                writer.writerow([file, "MISSING", 0.0])
                missing += 1
                continue

            text = extract_text_from_txt(file_path)
            if not text.strip():
                writer.writerow([file, "EMPTY", 0.0])
                empty += 1
                continue

            try:
                raw_pred = pipeline(text)

                # Handle list of lists when using top_k=1
                if isinstance(raw_pred, list) and isinstance(raw_pred[0], list):
                    prediction = raw_pred[0][0]
                else:
                    prediction = raw_pred[0]

                numeric_label = label_map.get(prediction["label"], -1)  # Fallback -1 for unexpected
                writer.writerow([file, numeric_label, round(prediction["score"], 4)])

            except Exception as e:
                writer.writerow([file, "ERROR", 0.0])
                error += 1
                print(f"⚠️ Error predicting {file}: {e}")
                continue

    print(f"\n✅ Done: {output_csv}")
    print(f"  • Missing: {missing}, Empty: {empty}, Errors: {error}")
