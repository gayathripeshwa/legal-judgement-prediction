import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# === Load ground truth ===
gt_df = pd.read_csv("ground_truth.csv")

# === Label mapping ===
label_map = {
    "LABEL_0": 0, "LABEL_1": 1,
    "Acquittal": 0, "Conviction": 1,
    0: 0, 1: 1
}


# === Versions to evaluate ===
versions = ["V1", "V2", "V3", "V4"]

for version in versions:
    pred_file = f"predictions_{version.lower()}.csv"

    try:
        pred_df = pd.read_csv(pred_file)
    except FileNotFoundError:
        print(f"❌ File not found: {pred_file}")
        continue

    pred_df["pred_label_num"] = pred_df["prediction"].map(label_map)

    # Drop unmappable predictions
    before = len(pred_df)
    pred_df = pred_df.dropna(subset=["pred_label_num"])
    after = len(pred_df)
    dropped = before - after

    print(f"\n📄 Evaluating {version} ({pred_file})")
    print(f"⚠️ {dropped} predictions in {pred_file} could not be mapped and will be skipped.")

    # Merge with ground truth
    merged_df = pd.merge(gt_df, pred_df, on="filename", how="inner")
    print(f"🔗 Merged rows: {len(merged_df)}")

    y_true = merged_df["label"].astype(int).tolist()
    y_pred = merged_df["pred_label_num"].astype(int).tolist()

    if not y_true or not y_pred:
        print("❌ No valid predictions to evaluate.")
        continue

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n✅ Accuracy for {version}: {accuracy * 100:.2f}%\n")
    print("📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Acquittal", "Conviction"]))
