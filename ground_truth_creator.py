import pandas as pd
import os

# === Paths ===
input_folder = "D:/Gayathri Gopal Peshwa/judgement/summarization/extractive/LetSum/letsum_outputs_cleaned"
cjpe_path = "D:/Gayathri Gopal Peshwa/judgement/data/cjpe_multi_train_0000.parquet"

# === Step 1: List all files actually present in the input folder ===
existing_files = set(os.listdir(input_folder))

# === Step 2: Load the parquet dataset ===
df = pd.read_parquet(cjpe_path).dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# === Step 3: Assign filenames in the same format ===
df = df.reset_index(drop=True)
df["filename"] = [f"doc_{i}.html" for i in range(len(df))]

# === Step 4: Filter only files that exist ===
df_filtered = df[df["filename"].isin(existing_files)]

print(f"✅ Total rows after filtering by existing files: {len(df_filtered)}")
print("Conviction count (1):", (df_filtered["label"] == 1).sum())
print("Acquittal count (0):", (df_filtered["label"] == 0).sum())

# === Step 5: Take the first 1000 rows only ===
df_sampled = df_filtered.head(3000).copy().reset_index(drop=True)

# Step 6: Save the ground truth
output_path = "ground_truth.csv"
df_sampled[["filename", "label"]].to_csv(output_path, index=False)

print(f"✅ Saved {output_path}")
print("📊 Label distribution:")
print(df_sampled["label"].value_counts())
