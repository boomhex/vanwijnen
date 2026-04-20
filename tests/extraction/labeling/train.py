import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Verwacht een DataFrame met minimaal:
# - "text": de regeltekst
# - "label": bijvoorbeeld "POST" of "GEEN_POST"
def make_features_for_logreg(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "post",
    extra_feature_cols: list[str] = ["has_unit", "page", "has_amount"],
    model_name: str = "textgain/allnli-GroNLP-bert-base-dutch-cased",
):
    data = df.copy()

    required_cols = [text_col, label_col] + extra_feature_cols
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Ontbrekende kolommen: {missing}")

    data = data.dropna(subset=[text_col, label_col]).reset_index(drop=True)
    data[text_col] = data[text_col].astype(str).str.strip()

    # Labels naar 0/1
    label_map = {
        False: 0,
        True: 1,
    }
    data["y"] = data[label_col].map(label_map)

    if data["y"].isna().any():
        unknown = data.loc[data["y"].isna(), label_col].unique()
        raise ValueError(f"Onbekende labels gevonden: {unknown}")

    # Extra features numeriek maken
    for col in extra_feature_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if data[extra_feature_cols].isna().any().any():
        bad_rows = data[data[extra_feature_cols].isna().any(axis=1)]
        raise ValueError(
            f"Niet-numerieke of missende waarden in extra features.\n"
            f"Probleemrijen:\n{bad_rows[[text_col] + extra_feature_cols]}"
        )

    # Embeddings
    model = SentenceTransformer(model_name)
    X_embed = model.encode(
        data[text_col].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Extra features
    X_extra = data[extra_feature_cols].to_numpy(dtype=np.float32)

    # Combineer embeddings + metadata/features
    X = np.hstack([X_embed, X_extra])
    y = data["y"].to_numpy(dtype=np.int64)

    return X, y, data
# Voorbeeldgebruik
if __name__ == "__main__":
    df = pd.read_csv("post_dataset.csv")

    X, y, data = make_features_for_logreg(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=3000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["GEEN_POST", "POST"]))
