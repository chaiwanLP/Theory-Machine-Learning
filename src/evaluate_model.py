import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
CLEANED_PATH = os.path.join(DATASET_DIR, "cleaned_dataset.csv")
MODEL_PATH = r"D:\Learn\Theory-Machine-Learning\Scopus\subject_classifier.joblib"
RANDOM_STATE = 42
TOP_N_CONFUSION_PAIRS = 10

# ที่เก็บรูป confusion matrix ที่ export ออกมา
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
CONFUSION_MATRIX_PNG = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
CONFUSION_MATRIX_NORMALIZED_PNG = os.path.join(
    OUTPUT_DIR, "confusion_matrix_normalized.png"
)


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


def top_confusion_pairs(cm: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """หาคู่ (true_subject, predicted_subject) ที่โมเดลสับสนกันมากที่สุด (ไม่รวม diagonal)"""
    rows = []
    for true_label in cm.index:
        for pred_label in cm.columns:
            if true_label == pred_label:
                continue
            count = cm.loc[true_label, pred_label]
            if count > 0:
                rows.append((true_label, pred_label, count))
    pairs = pd.DataFrame(rows, columns=["true_subject", "predicted_as", "count"])
    return pairs.sort_values("count", ascending=False).head(top_n).reset_index(drop=True)


def plot_confusion_matrix(
    cm_df: pd.DataFrame,
    normalize: bool,
    save_path: str,
) -> None:
    """วาด confusion matrix เป็น heatmap แล้วเซฟเป็นไฟล์รูปภาพ

    normalize=True จะ normalize ตามแถว (true label) เป็นสัดส่วน 0-1
    เพื่อให้เทียบกันได้ง่ายแม้จำนวนตัวอย่างในแต่ละ subject ไม่เท่ากัน
    """
    data = cm_df.copy()
    fmt = "d"
    title = "Confusion Matrix"

    if normalize:
        data = data.div(data.sum(axis=1), axis=0).fillna(0)
        fmt = ".2f"
        title = "Confusion Matrix"

    n_labels = len(cm_df)
    fig_size = max(8, n_labels * 0.6)

    plt.figure(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="white",
    )
    plt.title(title)
    plt.xlabel("Predicted subject")
    plt.ylabel("True subject")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"บันทึก confusion matrix ไปที่: {save_path}")


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"ไม่พบโมเดลที่ {MODEL_PATH} — กรุณารัน src/train_classifier.py ก่อน"
        )

    pipe = joblib.load(MODEL_PATH)

    df = pd.read_csv(CLEANED_PATH)
    df["text"] = build_text_column(df)

    X = df["text"]
    y = df["subject"]

    # ใช้ split เดียวกับตอน train (test_size, stratify, random_state ตรงกัน)
    # เพื่อให้ test set ที่นี่คือ test set เดียวกับที่ train_classifier.py ใช้ประเมิน
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    train_pred = pipe.predict(X_train)
    test_pred = pipe.predict(X_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print("=== 1. Overfitting check: Train vs Test accuracy ===")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print(f"Gap (train - test): {train_acc - test_acc:.4f}")
    print(
        "หมายเหตุ: gap ที่มาจาก TF-IDF บนโมเดล linear มักดูสูงเพราะ train accuracy "
        "แตะ 100% ได้ง่ายจาก feature space มิติสูง ให้ดู CV std ในหัวข้อถัดไปเป็นตัวชี้วัดหลักว่า overfit จริงหรือไม่"
    )

    print("\n=== 2. ความเสถียรของผลลัพธ์: 5-fold cross-validation (บน train set) ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, n_jobs=-1)
    print(f"CV scores: {cv_scores}")
    print(f"CV mean: {cv_scores.mean():.4f}  CV std: {cv_scores.std():.4f}")
    if cv_scores.std() < 0.01 and abs(cv_scores.mean() - test_acc) < 0.02:
        print("-> CV เสถียร และใกล้เคียง test accuracy: ไม่ใช่ overfitting ที่เป็นปัญหา")
    else:
        print("-> CV ไม่เสถียร หรือห่างจาก test accuracy มาก: ควรตรวจสอบ overfitting เพิ่มเติม")

    print("\n=== 3. Underfitting check: เทียบกับ majority-class baseline ===")
    baseline = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    baseline.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
    print(f"Baseline (predict subject ที่พบมากสุดเสมอ): {baseline_acc:.4f}")
    print(f"โมเดลจริง: {test_acc:.4f}")
    print(f"ดีกว่า baseline: +{test_acc - baseline_acc:.4f}")

    print("\n=== 4. Per-class metrics (test set) เรียงจาก f1 ต่ำสุดก่อน ===")
    labels = sorted(y.unique())
    precision = precision_score(y_test, test_pred, labels=labels, average=None)
    recall = recall_score(y_test, test_pred, labels=labels, average=None)
    f1 = f1_score(y_test, test_pred, labels=labels, average=None)
    support = y_test.value_counts().reindex(labels)

    per_class = pd.DataFrame(
        {
            "subject": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.values,
        }
    ).sort_values("f1")
    print(per_class.to_string(index=False))

    print(f"\n=== 5. Top {TOP_N_CONFUSION_PAIRS} confusion pairs (test set) ===")
    cm = confusion_matrix(y_test, test_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    pairs = top_confusion_pairs(cm_df, TOP_N_CONFUSION_PAIRS)
    print(pairs.to_string(index=False))

    n_distinct_true_subjects_confused = pairs["true_subject"].nunique()
    if n_distinct_true_subjects_confused >= 5:
        print(
            "\n-> ข้อผิดพลาดกระจายไปหลาย subject ไม่กระจุกที่คู่ใดคู่หนึ่ง "
            "สัญญาณว่าปัญหามาจาก label ที่นิยามคลุมเครือ/ทับซ้อนกัน ไม่ใช่จุดบกพร่องของโมเดล"
        )
    else:
        print(
            "\n-> ข้อผิดพลาดกระจุกอยู่ที่ subject จำนวนน้อยคู่ อาจพิจารณารวม/ปรับนิยาม subject เหล่านั้น"
        )

    print("\n=== 6. Confusion matrix (heatmap) ===")
    plot_confusion_matrix(cm_df, normalize=False, save_path=CONFUSION_MATRIX_PNG)
    plot_confusion_matrix(
        cm_df, normalize=True, save_path=CONFUSION_MATRIX_NORMALIZED_PNG
    )


if __name__ == "__main__":
    main()