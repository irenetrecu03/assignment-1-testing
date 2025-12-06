import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.model_selection import train_test_split

class PartitionTester:
    def __init__(self, data_path):
        self.DATA_PATH = data_path
        self.TARGET = "checked"
        self.CHILDREN_COL = "relatie_kind_huidige_aantal"

        # ---------- LOAD DATA (OPTIMIZED) ----------
        # Fix: Read only the first row to get headers (avoids DtypeWarning & reading file twice)
        df_header = pd.read_csv(self.DATA_PATH, nrows=0)
        colnames = df_header.columns.tolist()
        
        # Load actual data
        df = pd.read_csv(self.DATA_PATH, skiprows=1, names=colnames, low_memory=False)

        # Convert target
        df[self.TARGET] = pd.to_numeric(df[self.TARGET], errors="coerce")
        df = df.dropna(subset=[self.TARGET]).copy()
        df[self.TARGET] = df[self.TARGET].astype(int)

        # Prepare X, y
        X = df.drop(columns=[self.TARGET]).apply(pd.to_numeric, errors="coerce").fillna(0)
        y = df[self.TARGET]

        # Deterministic test split
        _, self.X_test, _, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # ---------- DEFINE PARTITIONS ----------
        self.partitions = [
            {"name": "No children",         "condition": lambda df: df[self.CHILDREN_COL] == 0},
            {"name": "One child",           "condition": lambda df: df[self.CHILDREN_COL] == 1},
            {"name": "Two or more children","condition": lambda df: df[self.CHILDREN_COL] >= 2},
        ]

    # ... (Keep _load_model and _predict exactly as you had them) ...
    def _load_model(self, m):
        if isinstance(m, str):
            return ort.InferenceSession(m, providers=["CPUExecutionProvider"])
        elif isinstance(m, ort.InferenceSession):
            return m
        elif hasattr(m, "predict"):
            return m
        else:
            raise TypeError("Unsupported model type (must be sklearn model or ONNX path).")

    def _predict(self, model, X_part):
        if hasattr(model, "predict"):
            return model.predict(X_part)
        elif isinstance(model, ort.InferenceSession):
            input_name = model.get_inputs()[0].name
            X_np = X_part.to_numpy().astype(np.float32)
            outputs = model.run(None, {input_name: X_np})
            label_index = None
            for i, o in enumerate(model.get_outputs()):
                if "label" in o.name.lower():
                    label_index = i
                    break
            if label_index is None: label_index = 0 # Fallback
            return np.array(outputs[label_index]).astype(int).flatten()

    def run(self, model_or_path):
        model = self._load_model(model_or_path)

        print("\n=========================================")
        print("      PARTITION TEST RESULTS")
        print("=========================================\n")

        accuracies = []
        fraud_rate_diffs = []
        detailed_stats = [] 

        for part in self.partitions:
            name = part["name"]
            cond = part["condition"]
            df_part = self.X_test[cond(self.X_test)]
            
            if df_part.empty:
                print(f"Skipping {name}: no rows.\n")
                continue

            true_labels = self.y_test.loc[df_part.index].astype(int)
            preds = self._predict(model, df_part)

            TP = np.sum((preds == 1) & (true_labels == 1))
            TN = np.sum((preds == 0) & (true_labels == 0))
            N = len(df_part)
            
            fraud_rate = true_labels.mean()
            pred_rate = preds.mean()
            acc = (TP + TN) / N
            gap = abs(pred_rate - fraud_rate)

            accuracies.append(acc)
            fraud_rate_diffs.append(gap)
            detailed_stats.append((name, acc, gap))

            print(f"--- Partition: {name} ---")
            print(f"Counts: N={N} (TP={TP}, TN={TN})")
            print(f"Acc: {acc:.4f} | Gap: {gap:.4f}")
            print("")

        # ---------------------------------------------------------
        # ASSERTION SECTION
        # ---------------------------------------------------------
        print("==========================================")
        print("          FAIRNESS CHECKS")
        print("==========================================")

        failed = False

        # 1. Accuracy Check
        acc_range = max(accuracies) - min(accuracies)
        print(f"1. Accuracy Range Check (Threshold 0.07):")
        print(f"   Calculated Range: {acc_range:.4f}")
        
        print("   Breakdown:")
        for name, acc, _ in detailed_stats:
            print(f"     - {name:<22}: {acc:.4f}")

        if acc_range <= 0.07:
            print("   Status: PASS")
        else:
            print("   Status: FAIL")
            failed = True

        print("-" * 30)

        # 2. Fraud Rate Gap Check
        max_gap = max(fraud_rate_diffs)
        print(f"2. Prediction Gap Check (Threshold 0.10):")
        print(f"   Max Gap Found:    {max_gap:.4f}")
        
        if max_gap <= 0.10:
            print("   Status: PASS")
        else:
            print("   Status: FAIL")
            for name, _, gap in detailed_stats:
                if gap > 0.10:
                    print(f"     -> {name} gap is {gap:.4f}")
            failed = True

        print("\n==========================================")
        if failed:
            print("BAD MODEL")
        else:
            print("GOOD MODEL")
        print("==========================================\n")