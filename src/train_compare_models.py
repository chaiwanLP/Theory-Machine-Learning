"""
train_compare_models.py
------------------------
เปรียบเทียบ 3 โมเดลสำหรับงาน Scopus subject classification บน train/test split
เดียวกัน (test_size=0.2, stratify, random_state=42 — ตรงกับ train_classifier.py)
เพื่อตอบคำถาม:
  1) โมเดลปัจจุบัน (TF-IDF + LinearSVC) เหมาะสมกับงานนี้แค่ไหน เทียบกับตัวเลือกอื่น
  2) SPECTER2 embeddings (เข้าใจความหมายเชิงบริบท) ช่วยแก้ปัญหา label ที่คาบเกี่ยวกัน
     (เช่น Computer Science Applications, Computer vision vs Computer Vision and
     Pattern Recognition) ได้จริงหรือไม่

โมเดลที่เทียบ:
  1. TF-IDF + LogisticRegression
  2. TF-IDF + LinearSVC   (ของเดิมใน train_classifier.py)
  3. SPECTER2 embeddings + LinearSVC  (ต้องรัน generate_specter2_embeddings.py ก่อน)

ทุกโมเดลใช้ train/test split ที่ index ตรงกัน (แถวเดียวกันในทั้ง TF-IDF และ
SPECTER2 dataframe) เพื่อให้เทียบผลกันได้อย่างยุติธรรม ไม่มี data leakage
เพราะ cleaned_dataset.csv ผ่านการ dedup EID+Title มาแล้วก่อนหน้านี้
"""

import os

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
CLEANED_PATH = os.path.join(DATASET_DIR, "cleaned_dataset.csv")
EMBEDDINGS_PATH = os.path.join(DATASET_DIR, "specter2_embeddings.npy")
LABELS_PATH = os.path.join(DATASET_DIR, "specter2_labels.npy")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "model_comparison_report.md")

RANDOM_STATE = 42
WATCH_SUBJECTS = ["Computer Science Applications", "Computer vision", "Computer Vision and Pattern Recognition"]


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


def evaluate(name, pipe, X_train, X_test, y_train, y_test, labels, cv_X=None, cv_y=None):
    pipe.fit(X_train, y_train)
    train_pred = pipe.predict(X_train)
    test_pred = pipe.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    macro_f1 = f1_score(y_test, test_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, test_pred, average="weighted", zero_division=0)

    cv_X = X_train if cv_X is None else cv_X
    cv_y = y_train if cv_y is None else cv_y
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipe, cv_X, cv_y, cv=cv, n_jobs=-1)

    precision = precision_score(y_test, test_pred, labels=labels, average=None, zero_division=0)
    recall = recall_score(y_test, test_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_test, test_pred, labels=labels, average=None, zero_division=0)
    per_class = pd.DataFrame(
        {"subject": labels, "precision": precision, "recall": recall, "f1": f1}
    )

    result = {
        "name": name,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "gap": train_acc - test_acc,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }

    print(f"\n=== {name} ===")
    print(f"Train acc: {train_acc:.4f}  Test acc: {test_acc:.4f}  Gap: {train_acc - test_acc:.4f}")
    print(f"CV mean: {cv_scores.mean():.4f}  CV std: {cv_scores.std():.4f}")
    print(f"Macro F1: {macro_f1:.4f}  Weighted F1: {weighted_f1:.4f}")
    watch = per_class[per_class["subject"].isin(WATCH_SUBJECTS)]
    print("Watch subjects:\n", watch.to_string(index=False))

    return result


def main() -> None:
    df = pd.read_csv(CLEANED_PATH)
    df["text"] = build_text_column(df)
    labels_sorted = sorted(df["subject"].unique())

    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"ไม่พบ {EMBEDDINGS_PATH} — กรุณารัน src/generate_specter2_embeddings.py ก่อน"
        )

    specter2_embeddings = np.load(EMBEDDINGS_PATH)
    specter2_labels = np.load(LABELS_PATH, allow_pickle=True)

    if len(specter2_embeddings) != len(df) or not (specter2_labels == df["subject"].to_numpy()).all():
        raise ValueError(
            "SPECTER2 embeddings/labels ไม่ตรงแถวกับ cleaned_dataset.csv "
            "(อาจ regenerate cleaned_dataset.csv หลังสร้าง embeddings ไปแล้ว)"
        )

    # ใช้ index เดียวกันสำหรับ TF-IDF text และ SPECTER2 embeddings
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=df["subject"], random_state=RANDOM_STATE
    )

    y = df["subject"].to_numpy()
    y_train, y_test = y[train_idx], y[test_idx]

    text_train, text_test = df["text"].to_numpy()[train_idx], df["text"].to_numpy()[test_idx]
    emb_train, emb_test = specter2_embeddings[train_idx], specter2_embeddings[test_idx]

    print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

    baseline = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    baseline.fit(text_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(text_test))
    print(f"\nMajority-class baseline test accuracy: {baseline_acc:.4f}")

    results = []

    # Model 1: TF-IDF + LogisticRegression
    pipe1 = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
            ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ]
    )
    results.append(evaluate("TF-IDF + LogisticRegression", pipe1, text_train, text_test, y_train, y_test, labels_sorted))

    # Model 2: TF-IDF + LinearSVC (baseline เดิม)
    pipe2 = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
            ("clf", LinearSVC(C=1.0, random_state=RANDOM_STATE)),
        ]
    )
    results.append(evaluate("TF-IDF + LinearSVC (current baseline)", pipe2, text_train, text_test, y_train, y_test, labels_sorted))

    # Model 3: SPECTER2 embeddings + LinearSVC
    pipe3 = LinearSVC(C=1.0, random_state=RANDOM_STATE)
    results.append(evaluate("SPECTER2 + LinearSVC", pipe3, emb_train, emb_test, y_train, y_test, labels_sorted))

    # สร้างรายงานสรุป
    summary_rows = []
    for r in results:
        summary_rows.append(
            {
                "Model": r["name"],
                "Train Acc": f"{r['train_acc']:.4f}",
                "Test Acc": f"{r['test_acc']:.4f}",
                "Gap": f"{r['gap']:.4f}",
                "CV Mean": f"{r['cv_mean']:.4f}",
                "CV Std": f"{r['cv_std']:.4f}",
                "Macro F1": f"{r['macro_f1']:.4f}",
                "Weighted F1": f"{r['weighted_f1']:.4f}",
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Model Comparison Report: Scopus Subject Classification\n\n")
        f.write(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}\n\n")
        f.write(f"Majority-class baseline test accuracy: {baseline_acc:.4f}\n\n")
        f.write("## Summary\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")
        for r in results:
            f.write(f"## {r['name']}: per-class metrics\n\n")
            f.write(r["per_class"].sort_values("f1").to_markdown(index=False))
            f.write("\n\n")

    print(f"\nบันทึกรายงานที่: {REPORT_PATH}")


if __name__ == "__main__":
    main()
