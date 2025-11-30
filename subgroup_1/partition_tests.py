import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.model_selection import train_test_split

# ============================================================
#  SELF-CONTAINED PARTITION TEST FUNCTION
# ============================================================

def run_partition_tests_for_model(model_or_path):
    """
    Fully self-contained:
      - Loads translations.csv
      - Cleans data
      - Builds X_test, y_test
      - Defines partitions
      - Loads model (sklearn or ONNX)
      - Runs predictions
      - Prints results

    You only call:
        run_partition_tests_for_model("model_1.onnx")
    """

    DATA_PATH = "./data/translations.csv"
    TARGET = "checked"
    CHILDREN_COL = "relationship_child_current_number"

    # ------------------------------------------------------------
    # LOAD + PREPARE DATA
    # ------------------------------------------------------------

    df_raw = pd.read_csv(DATA_PATH, header=None)
    colnames = df_raw.iloc[0].tolist()
    df = pd.read_csv(DATA_PATH, skiprows=1, names=colnames)

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    X = df.drop(columns=[TARGET]).apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df[TARGET]

    # ALWAYS SAME SPLIT
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # ------------------------------------------------------------
    # DEFINE PARTITIONS
    # ------------------------------------------------------------
    partitions = [
        {"name": "No children", "condition": lambda df: df[CHILDREN_COL] == 0},
        {"name": "One child", "condition": lambda df: df[CHILDREN_COL] == 1},
        {"name": "Two or more children", "condition": lambda df: df[CHILDREN_COL] >= 2},
    ]

    # ------------------------------------------------------------
    # LOAD MODEL (sklearn or ONNX)
    # ------------------------------------------------------------
    def load_model(m):
        if isinstance(m, str):
            return ort.InferenceSession(m, providers=["CPUExecutionProvider"])
        elif isinstance(m, ort.InferenceSession):
            return m
        elif hasattr(m, "predict"):
            return m
        else:
            raise TypeError("Unsupported model type")

    model = load_model(model_or_path)

    # ------------------------------------------------------------
    # PREDICT HANDLER
    # ------------------------------------------------------------
    def run_predict(model, X_part):
        if hasattr(model, "predict"):
            return model.predict(X_part)

        elif isinstance(model, ort.InferenceSession):
            input_name = model.get_inputs()[0].name
            X_np = X_part.to_numpy().astype(np.float32)
            outputs = model.run(None, {input_name: X_np})

            # find label output
            label_index = None
            for i, o in enumerate(model.get_outputs()):
                if "label" in o.name.lower():
                    label_index = i
                    break

            if label_index is None:
                raise ValueError("ONNX model missing label output.")

            return np.array(outputs[label_index]).astype(int).flatten()

    # ------------------------------------------------------------
    # RUN PARTITION TESTS
    # ------------------------------------------------------------

    print("\n=========================================")
    print("      PARTITION TEST RESULTS")
    print("=========================================\n")

    for part in partitions:
        name = part["name"]
        cond = part["condition"]

        df_part = X_test[cond(X_test)]
        idx = df_part.index
        true_labels = y_test.loc[idx].astype(int)

        if df_part.empty:
            print(f"Skipping {name}: no rows.\n")
            continue

        preds = run_predict(model, df_part)

        TP = np.sum((preds == 1) & (true_labels == 1))
        TN = np.sum((preds == 0) & (true_labels == 0))
        FP = np.sum((preds == 1) & (true_labels == 0))
        FN = np.sum((preds == 0) & (true_labels == 1))

        N = len(df_part)

        fraud_rate = true_labels.mean()
        pred_rate = preds.mean()
        acc = (TP + TN) / N
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        TNR = TN / (TN + FP) if (TN + FP) > 0 else 0

        print("===================================")
        print(f"Partition: {name}")
        print("===================================")
        print(f"Data points: {N}")
        print(f"Actual fraud rate:   {fraud_rate*100:.2f}%")
        print(f"Predicted fraud rate:{pred_rate*100:.2f}%")
        print("\n--- Confusion Matrix ---")
        print(f"TP={TP}  TN={TN}  FP={FP}  FN={FN}")
        print("\n--- Metrics ---")
        print(f"Accuracy: {acc*100:.2f}%")
        print(f"FPR: {FPR*100:.2f}%")
        print(f"FNR: {FNR*100:.2f}%")
        print(f"TPR/Recall: {TPR*100:.2f}%")
        print(f"TNR: {TNR*100:.2f}%\n")
