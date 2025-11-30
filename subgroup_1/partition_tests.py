import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.model_selection import train_test_split


class PartitionTester:
    """
    Self-contained partition testing utility.
    
    Usage:
        tester = PartitionTester("../data/translations.csv")
        tester.run("model_1.onnx")   # or a sklearn model
    """

    # -------------------------------------------------------------
    # INITIALIZER: loads dataset, cleans it, prepares test split
    # -------------------------------------------------------------
    def __init__(self, data_path):
        self.DATA_PATH = data_path
        self.TARGET = "checked"
        self.CHILDREN_COL = "relationship_child_current_number"

        # ---------- LOAD & PREPARE DATA ----------
        df_raw = pd.read_csv(self.DATA_PATH, header=None)
        colnames = df_raw.iloc[0].tolist()
        df = pd.read_csv(self.DATA_PATH, skiprows=1, names=colnames)

        df[self.TARGET] = pd.to_numeric(df[self.TARGET], errors="coerce")
        df = df.dropna(subset=[self.TARGET]).copy()
        df[self.TARGET] = df[self.TARGET].astype(int)

        # X, y
        X = df.drop(columns=[self.TARGET]).apply(pd.to_numeric, errors="coerce").fillna(0)
        y = df[self.TARGET]

        # ALWAYS same test set (deterministic)
        _, self.X_test, _, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # ---------- DEFINE PARTITIONS ----------
        self.partitions = [
            {"name": "No children",         "condition": lambda df: df[self.CHILDREN_COL] == 0},
            {"name": "One child",           "condition": lambda df: df[self.CHILDREN_COL] == 1},
            {"name": "Two or more children","condition": lambda df: df[self.CHILDREN_COL] >= 2},
        ]

    # -------------------------------------------------------------
    # MODEL LOADER (sklearn or ONNX)
    # -------------------------------------------------------------
    def _load_model(self, m):
        if isinstance(m, str):
            # string path → assume ONNX
            return ort.InferenceSession(m, providers=["CPUExecutionProvider"])
        elif isinstance(m, ort.InferenceSession):
            return m
        elif hasattr(m, "predict"):
            return m
        else:
            raise TypeError("Unsupported model type (must be sklearn model or ONNX path).")

    # -------------------------------------------------------------
    # UNIFIED PREDICT FUNCTION
    # -------------------------------------------------------------
    def _predict(self, model, X_part):
        # sklearn
        if hasattr(model, "predict"):
            return model.predict(X_part)

        # ONNX
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

    # -------------------------------------------------------------
    # MAIN PUBLIC METHOD
    # -------------------------------------------------------------
    def run(self, model_or_path):
        """
        Execute full partition tests on the given model.
        """
        model = self._load_model(model_or_path)

        print("\n=========================================")
        print("      PARTITION TEST RESULTS")
        print("=========================================\n")

        for part in self.partitions:
            name = part["name"]
            cond = part["condition"]

            df_part = self.X_test[cond(self.X_test)]
            idx = df_part.index
            true_labels = self.y_test.loc[idx].astype(int)

            if df_part.empty:
                print(f"Skipping {name}: no rows.\n")
                continue

            preds = self._predict(model, df_part)

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
