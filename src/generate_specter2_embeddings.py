"""
generate_specter2_embeddings.py
--------------------------------
สร้าง SPECTER2 embeddings สำหรับบทความทั้งหมดใน dataset/cleaned_dataset.csv
(ต่อจากที่ 08_Scopus_create_dataset.ipynb ทำไว้กับตัวอย่าง 200 บทความ — สคริปต์นี้รันกับ
ข้อมูลที่ clean แล้วทั้งหมด 36,069 บทความ เพื่อใช้เทียบกับ TF-IDF ใน model comparison)

ขั้นตอน:
1. โหลด cleaned_dataset.csv, สร้าง embedding_text (Title/Abstract/Author/Index Keywords)
   ด้วยรูปแบบเดียวกับที่ train_classifier.py ใช้ เพื่อให้เทียบกันได้ตรง ๆ
2. โหลด SPECTER2 base model + proximity adapter บน GPU (ถ้ามี)
3. Batch encode ทั้งหมด (truncate 512 tokens ตามข้อจำกัดของโมเดล)
4. L2-normalize แล้วบันทึกเป็น dataset/specter2_embeddings.npy
   พร้อม labels เป็น dataset/specter2_labels.npy (เรียงตามแถวเดียวกัน)
"""

import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
CLEANED_PATH = os.path.join(DATASET_DIR, "cleaned_dataset.csv")
EMBEDDINGS_PATH = os.path.join(DATASET_DIR, "specter2_embeddings.npy")
LABELS_PATH = os.path.join(DATASET_DIR, "specter2_labels.npy")

BATCH_SIZE = 32
MAX_LENGTH = 512


def build_text_column(df: pd.DataFrame) -> pd.Series:
    def combine(row):
        parts = []
        if row["Title"]:
            parts.append(f"Title: {row['Title']}")
        if row["Abstract"]:
            parts.append(f"Abstract: {row['Abstract']}")
        if row["Author Keywords"]:
            parts.append(f"Author keywords: {row['Author Keywords']}")
        if row["Index Keywords"]:
            parts.append(f"Index keywords: {row['Index Keywords']}")
        return "\n".join(parts)

    return df.fillna("").apply(combine, axis=1)


def create_specter2_embeddings(texts, model, tokenizer, device, batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    all_embeddings = []

    for start_index in tqdm(
        range(0, len(texts), batch_size), desc="Creating SPECTER2 embeddings"
    ):
        batch_texts = texts[start_index : start_index + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        batch_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        all_embeddings.append(batch_embeddings)

    return np.vstack(all_embeddings)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = pd.read_csv(CLEANED_PATH)
    df["embedding_text"] = build_text_column(df)
    texts = df["embedding_text"].fillna("").astype(str).tolist()
    labels = df["subject"].to_numpy()

    print("จำนวนบทความ:", len(texts))

    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter(
        "allenai/specter2", source="hf", load_as="proximity", set_active=True
    )
    model = model.to(device)
    model.eval()
    print("โหลด SPECTER2 เรียบร้อยแล้ว")

    start = time.time()
    embeddings = create_specter2_embeddings(texts, model, tokenizer, device)
    elapsed = time.time() - start
    print(f"Embedding shape: {embeddings.shape}")
    print(f"เวลาที่ใช้: {elapsed:.1f} วินาที ({elapsed/60:.1f} นาที)")

    embeddings_normalized = normalize(embeddings, norm="l2")
    print(f"Normalized embedding shape: {embeddings_normalized.shape}")

    np.save(EMBEDDINGS_PATH, embeddings_normalized)
    np.save(LABELS_PATH, labels)
    print(f"บันทึก embeddings ที่: {EMBEDDINGS_PATH}")
    print(f"บันทึก labels ที่: {LABELS_PATH}")


if __name__ == "__main__":
    main()
