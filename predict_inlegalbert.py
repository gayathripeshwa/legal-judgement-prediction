import os
import csv
import pandas as pd
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, BertForSequenceClassification, TextClassificationPipeline
from tqdm import tqdm

# === CONFIG ===
model_path = "inlegalbert_finetuned"
input_folder = "D:/Gayathri Gopal Peshwa/judgement/summarization/extractive/LetSum/letsum_outputs_cleaned"
ground_truth_csv = "ground_truth.csv"  # updated ground truth file
output_csv = "predictions.csv"
max_files = 3000  # Ensure only first 1000 files are predicted

# === Load Fine-Tuned Model ===
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
pipeline = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    top_k=1,
    truncation=True,
    max_length=512
)

# === Read HTML and Extract Text ===
def extract_text_from_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
        return soup.get_text(separator=" ", strip=True)

# === Load ground truth and extract filenames ===
gt_df = pd.read_csv(ground_truth_csv)
files_to_predict = gt_df["filename"].tolist()[:max_files]  # Ensures order and max limit

# === Get Already Processed Files ===
processed_files = set()
if os.path.exists(output_csv):
    with open(output_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            processed_files.add(row[0])

# === Process and Predict ===
with open(output_csv, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if not processed_files:
        writer.writerow(["filename", "prediction", "confidence"])

    for file in tqdm(files_to_predict, desc="Processing files"):
        if file in processed_files:
            continue

        file_path = os.path.join(input_folder, file)
        if not os.path.exists(file_path):
            writer.writerow([file, "MISSING", 0.0])
            continue

        text = extract_text_from_html(file_path)
        if not text.strip():
            writer.writerow([file, "EMPTY", 0.0])
            continue

        prediction = pipeline(text)[0]
        pred_label = prediction[0]["label"]
        confidence = prediction[0]["score"]

        writer.writerow([file, pred_label, round(confidence, 4)])
        f.flush()

print("\n✅ All predictions saved to predictions.csv")
