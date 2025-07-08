import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from tqdm import tqdm

# Load the CSV file with extracted facts
df = pd.read_csv("cjpe_facts_extracted.csv")

# Load Legal-BART tokenizer and model (as per the paper)
model_name = "nsi319/legalbart"  # From HuggingFace Hub
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Summarize each 'Extracted_Facts' entry
summaries = []
for text in tqdm(df["Extracted_Facts"]):
    inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    summary_ids = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summaries.append(summary)

# Add summaries to the DataFrame
df["Facts_Summary"] = summaries

# Save to new CSV
df.to_csv("cjpe_facts_summarized.csv", index=False)
print("✅ Summaries saved to cjpe_facts_summarized.csv")
